#!/usr/bin/env python3
"""
pretrain.py -- Supervised warm-start of the AlphaZero-chess value/policy
network on real human games (e.g. Lichess broadcast PGNs), before handing
off to train.py's self-play loop.

Why this exists: train.py's self-play starts from a randomly initialized
network, which has to bootstrap both material/tactical understanding and
decent play from nothing -- slow, and prone to the value-head collapse
this project has repeatedly hit under high-draw self-play conditions.
This script instead trains the *same* network architecture on real game
data first: policy target = the move actually played, value target =
the game's real final result (from the mover's own perspective, same
convention train.py's self-play buffer already uses). The output is a
plain state_dict checkpoint that train.py can load directly as
best_model.pt and continue self-play from.

Correctness notes (read before changing anything):
  - Reuses train.board_to_tensor / train.move_policy_index / and
    train.DualHeadResNet directly, rather than reimplementing the
    canonical-mirroring encoding. Do not duplicate that logic here --
    any divergence would silently produce a mismatched checkpoint.
  - board_to_tensor()'s repetition-count channel (channel 12) needs a
    board with real move history to be meaningful. We get this for free
    here because each game is replayed move-by-move from game.board()
    with board.push(move) -- unlike a bare FEN, this board's repetition
    state is genuine.
  - Value target is derived from the actual PGN Result tag, not a
    centipawn evaluation, so there is no White-POV-vs-mover-POV
    conversion step to get wrong: z_white in {+1, -1, 0}, flipped to
    the mover's own perspective at each position exactly like
    train.py's self-play buffer does.
  - Policy target is a single played move (one-hot), which is peakier
    than train.py's soft MCTS visit-count targets. Expect the policy
    head to look a little overconfident right after pretraining, and
    expect early self-play generations to soften it back down -- this
    is normal, not a regression. --label-smoothing is provided as a
    cheap lever to soften this if it looks like a problem in practice.
  - --value-target eval/blend reads PGN %eval comments as an additional
    (denser, less noisy) value signal, ASSUMING they are signed from
    White's perspective -- the standard Lichess broadcast convention
    for both centipawn and mate ("#N"/"#-N") scores. If your source
    uses side-to-move-relative signs instead, every eval-derived target
    on a Black move would come out inverted. Sanity-check a handful of
    known positions from your actual PGNs before trusting a long run in
    "eval" or "blend" mode. This also changes what the checkpoint is
    optimizing for: outcome-only targets are the faithful AlphaZero
    objective (learn from self-play-style final results); eval-based
    targets partially imitate whatever engine annotated the PGNs, which
    trains faster/less noisily per position but inherits that engine's
    biases. "outcome" remains the default for this reason.

Usage:
    # single file
    python3 pretrain.py --pgn warm_train/lichess_db_broadcast_2025-01.pgn \
        --output best_model.pt

    # a whole directory of monthly PGNs, or a glob -- all games across all
    # files are streamed into one continuous reservoir-sampled run
    python3 pretrain.py --pgn warm_train/ --output best_model.pt
    python3 pretrain.py --pgn "warm_train/lichess_db_broadcast_2025-*.pgn" \
        --output best_model.pt

See --help for all tunable knobs.
"""

import argparse
import glob
import os
import random
import re

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import train  # reuse board_to_tensor, move_policy_index, DualHeadResNet exactly

RESULT_TO_Z_WHITE = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

# Matches Lichess-style PGN eval annotations embedded in move comments, e.g.
# "{[%eval 0.34]}" (centipawns, in pawns) or "{[%eval #-3]}" (mate score).
# Also tolerates a clock annotation sharing the same comment, e.g.
# "{[%eval 0.34] [%clk 0:05:00]}".
EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")

# ASSUMPTION -- verify against your own data before trusting --value-target
# eval/blend: eval annotations are assumed to be signed from White's
# perspective (positive = good for White), which is the standard Lichess
# broadcast convention for both centipawn and mate ("#N"/"#-N") scores.
# If your PGNs turn out to use side-to-move-relative signs instead, every
# eval-derived target below would be inverted on Black's moves -- flip the
# sign in parse_eval_to_z if you confirm that's the case for your source.
MATE_SATURATION_Z = 0.99  # target used for mate scores, kept off exactly +-1
                           # so it doesn't push tanh's pre-activation to +-inf


def parse_eval_to_z(comment: str, eval_scale: float) -> float | None:
    """Extracts a PGN %eval annotation from a move comment and squashes it
    to a bounded value in [-1, 1] from White's perspective, matching the
    scale of the outcome-based z the rest of this script uses. Returns
    None if the comment has no eval annotation (common for older/lower-
    quality broadcast games, or the very first ply before any move has
    been played and analyzed).

    Centipawn scores are squashed with tanh(pawns / eval_scale) -- a
    simple monotonic approximation of engine-eval-to-win-probability, not
    a calibrated one. Mate scores saturate to +-MATE_SATURATION_Z rather
    than +-1.0 exactly, since a value head trained on exact +-1 targets
    (unreachable by tanh) sees needlessly large gradients trying to
    approach them.
    """
    if not comment:
        return None
    m = EVAL_RE.search(comment)
    if not m:
        return None
    raw = m.group(1)
    if raw.startswith("#"):
        mate_in = raw[1:]
        sign = -1.0 if mate_in.startswith("-") else 1.0
        return sign * MATE_SATURATION_Z
    pawns = float(raw)
    return float(np.tanh(pawns / eval_scale))


# NAG (Numeric Annotation Glyph) codes for move-quality suffixes, per the
# PGN standard: $1 "!" good, $2 "?" mistake, $3 "!!" brilliant,
# $4 "??" blunder, $5 "!?" speculative, $6 "?!" dubious/inaccuracy.
# python-chess parses suffixes like "h6??" straight from SAN into
# node.nags, independent of language/broadcast, so this is the primary,
# more robust signal.
NAG_SEVERITY = {4: 1.0, 2: 0.5, 6: 0.25}

# Fallback: Lichess-style analysis comments spell it out in English
# regardless of the broadcast's own language (see the Chinese-titled
# broadcast in this project's sample PGN, which still says "Blunder."/
# "Mistake."/"Inaccuracy." in the comment text). Used when a source
# doesn't carry NAGs but does carry this commentary.
BLUNDER_TEXT_RE = re.compile(r"\b(Blunder|Mistake|Inaccuracy)\b")
TEXT_SEVERITY = {"Blunder": 1.0, "Mistake": 0.5, "Inaccuracy": 0.25}


def move_quality_penalty(node) -> float:
    """Returns a severity in [0, 1] if `node` -- the PGN node for the move
    that was just played -- was itself flagged as an inaccuracy/mistake/
    blunder by the annotator, else 0.0.

    0.0 means "ordinary example, imitate normally." A value > 0.0 means
    the annotator judged this exact move (not the resulting position) to
    be bad; the caller uses this to train the policy head away from that
    move instead of imitating it, scaled by severity. This deliberately
    doesn't drop the example -- the position and its value target are
    still useful training signal (see --value-target eval/blend for how
    the value head learns from it) -- it only changes how the *policy*
    head treats the move that was played there.
    """
    if node is None:
        return 0.0
    severity = 0.0
    for nag in node.nags:
        if nag in NAG_SEVERITY:
            severity = max(severity, NAG_SEVERITY[nag])
    if node.comment:
        m = BLUNDER_TEXT_RE.search(node.comment)
        if m:
            severity = max(severity, TEXT_SEVERITY[m.group(1)])
    return severity


# ----------------------------------------------------------------------
# Streaming example generator
# ----------------------------------------------------------------------

def resolve_pgn_paths(pgn_arg: list[str]) -> list[str]:
    """Expands --pgn args (which may each be a file, a directory, or a
    glob pattern) into a flat, de-duplicated, sorted list of .pgn files.
    Sorting is by filename so that e.g. lichess_db_broadcast_2025-07.pgn,
    ..._2025-08.pgn, ..., ..._2026-06.pgn stream in chronological order
    when the files follow that naming convention -- purely cosmetic
    (reservoir sampling doesn't care about order) but makes progress
    logs easier to reason about.
    """
    paths = set()
    for arg in pgn_arg:
        if os.path.isdir(arg):
            paths.update(glob.glob(os.path.join(arg, "*.pgn")))
        elif any(ch in arg for ch in "*?["):
            paths.update(glob.glob(arg))
        else:
            paths.add(arg)

    resolved = sorted(paths)
    if not resolved:
        raise FileNotFoundError(f"No .pgn files found for --pgn {pgn_arg!r}")
    return resolved


def iter_examples(pgn_paths: list[str], skip_bots: bool, min_elo: int | None,
                   value_target: str = "outcome", eval_weight: float = 0.5,
                   eval_scale: float = 400.0):
    """Yields (state, policy_idx, z, quality_penalty) for every position in
    every game across every file in pgn_paths, streaming game-by-game (and
    file-by-file) so nothing ever has to sit fully in memory. `state` is a
    (13, 8, 8) float32 array from train.board_to_tensor; `policy_idx` is
    an int in [0, ACTION_SIZE); `z` is a float in {+1, -1, 0} (or, in
    "eval"/"blend" mode, a continuous value in [-1, 1]) from the mover's
    own perspective at that position, matching train.py's self-play
    buffer convention. `quality_penalty` is 0.0 for an ordinary move, or
    in (0, 1] if the move actually played was itself flagged as an
    inaccuracy/mistake/blunder by the PGN annotator -- see
    move_quality_penalty() for how the caller should use this (train the
    policy head away from that move rather than imitating it).

    value_target controls where z comes from:
      - "outcome": always the game's final result (original behavior).
      - "eval": the position's PGN %eval annotation, squashed to [-1, 1]
        via parse_eval_to_z, falling back to the game outcome for plies
        that have no eval annotation attached.
      - "blend": eval_weight * eval_z + (1 - eval_weight) * outcome_z,
        also falling back to pure outcome_z when no eval is present.
    """
    use_eval = value_target in ("eval", "blend")

    for pgn_path in pgn_paths:
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                headers = game.headers
                if skip_bots and (headers.get("WhiteTitle") == "BOT"
                                   or headers.get("BlackTitle") == "BOT"):
                    continue

                result = headers.get("Result", "*")
                if result not in RESULT_TO_Z_WHITE:
                    continue  # unfinished/unknown-result games have no value target

                if min_elo is not None:
                    try:
                        w_elo = int(headers.get("WhiteElo", ""))
                        b_elo = int(headers.get("BlackElo", ""))
                    except ValueError:
                        continue  # missing/unparseable Elo -- skip under a filter
                    if w_elo < min_elo or b_elo < min_elo:
                        continue

                z_white = RESULT_TO_Z_WHITE[result]
                board = game.board()
                # A comment on a mainline node describes the position that
                # results from that node's move -- i.e. it annotates the
                # *next* ply's starting position, not the move that
                # produced it. So we track "the comment that describes the
                # board we're currently sitting at" one step behind the
                # move-iteration below. The root's comment (rarely an
                # eval) covers the starting position.
                pending_comment = game.comment
                node = game
                for move in game.mainline_moves():
                    if not board.is_legal(move):
                        break  # malformed game data -- stop rather than crash
                    state = train.board_to_tensor(board)
                    mirror = board.turn == chess.BLACK
                    policy_idx = train.move_policy_index(move, mirror=mirror)
                    mover_is_white = board.turn == chess.WHITE
                    outcome_z = z_white if mover_is_white else -z_white

                    if use_eval:
                        eval_z_white = parse_eval_to_z(pending_comment, eval_scale)
                        if eval_z_white is None:
                            z = outcome_z  # no eval on this ply -- fall back
                        else:
                            eval_z = eval_z_white if mover_is_white else -eval_z_white
                            if value_target == "eval":
                                z = eval_z
                            else:  # "blend"
                                z = eval_weight * eval_z + (1 - eval_weight) * outcome_z
                    else:
                        z = outcome_z

                    board.push(move)
                    node = node.next()
                    # `node` now IS the node for the move we just played --
                    # its own nags/comment (e.g. "??" / "Blunder. Nxd3 was
                    # best.") annotate the quality of THIS move, unlike
                    # pending_comment above which describes the *resulting*
                    # position and is used one ply later for eval.
                    quality_penalty = move_quality_penalty(node)
                    pending_comment = node.comment if node is not None else ""

                    yield state, policy_idx, z, quality_penalty


# ----------------------------------------------------------------------
# Fixed-capacity reservoir (uniform random sample over the whole stream
# seen so far, memory-bounded regardless of how large the PGN file is)
# ----------------------------------------------------------------------

class Reservoir:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = []
        self.total_seen = 0

    def add(self, item):
        self.total_seen += 1
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            j = random.randrange(self.total_seen)
            if j < self.capacity:
                self.buf[j] = item

    def sample_batch(self, batch_size: int):
        idxs = [random.randrange(len(self.buf)) for _ in range(batch_size)]
        states = np.stack([self.buf[i][0] for i in idxs])
        policy_idx = np.array([self.buf[i][1] for i in idxs], dtype=np.int64)
        z = np.array([self.buf[i][2] for i in idxs], dtype=np.float32)
        quality_penalty = np.array([self.buf[i][3] for i in idxs], dtype=np.float32)
        return (torch.from_numpy(states),
                torch.from_numpy(policy_idx),
                torch.from_numpy(z).unsqueeze(1),
                torch.from_numpy(quality_penalty))

    def __len__(self):
        return len(self.buf)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_chunk(model, optimizer, reservoir, steps, batch_size, device,
                 value_loss_weight, label_smoothing, blunder_penalty_weight):
    """Ordinary examples (quality_penalty == 0) train the policy head as
    before: label-smoothed cross-entropy imitating the move played.

    Examples flagged by the PGN annotator as an inaccuracy/mistake/blunder
    (quality_penalty > 0, from move_quality_penalty()) are NOT imitated.
    Instead they go through an unlikelihood loss that pushes probability
    mass on that specific move DOWN, scaled by severity and
    blunder_penalty_weight:

        -log(1 - p_flagged_move)

    This is bounded and well-behaved (unlike naively negating the usual
    cross-entropy, -log(p), which blows up toward -inf as p->0 and can
    destabilize training) -- it's the standard "unlikelihood training"
    trick for teaching a model to avoid a specific output without
    removing the example (the position/state is still useful training
    data, e.g. for the value head via --value-target eval/blend).
    """
    model.train()
    total_p, total_v, total_b, bad_seen = 0.0, 0.0, 0.0, 0
    for _ in range(steps):
        states, policy_idx, z, quality_penalty = reservoir.sample_batch(batch_size)
        states = states.to(device)
        policy_idx = policy_idx.to(device)
        z = z.to(device)
        quality_penalty = quality_penalty.to(device)

        logits, pred_value = model(states)
        value_loss = F.mse_loss(pred_value, z)

        is_bad = quality_penalty > 0
        is_good = ~is_bad

        policy_loss = torch.zeros((), device=device)
        if is_good.any():
            policy_loss = policy_loss + F.cross_entropy(
                logits[is_good], policy_idx[is_good], label_smoothing=label_smoothing)

        blunder_loss = torch.zeros((), device=device)
        if is_bad.any():
            probs = F.softmax(logits[is_bad], dim=1)
            p_flagged = probs.gather(1, policy_idx[is_bad].unsqueeze(1)).squeeze(1)
            severity = quality_penalty[is_bad]
            blunder_loss = (-torch.log(1.0 - p_flagged + 1e-6) * severity).mean()
            policy_loss = policy_loss + blunder_penalty_weight * blunder_loss
            bad_seen += 1

        loss = policy_loss + value_loss_weight * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_p += policy_loss.item()
        total_v += value_loss.item()
        total_b += blunder_loss.item()
    avg_b = total_b / bad_seen if bad_seen else 0.0
    return total_p / steps, total_v / steps, avg_b


def main():
    parser = argparse.ArgumentParser(description="Supervised pretrain on real PGN games.")
    parser.add_argument("--pgn", type=str, required=True, nargs="+",
                         help="One or more (decompressed) PGN sources. Each can be a single "
                              "file, a directory (all *.pgn inside are used), or a glob "
                              "pattern (quote it so your shell doesn't expand it first), "
                              "e.g. --pgn warm_train/ or "
                              "--pgn \"warm_train/lichess_db_broadcast_2025-*.pgn\" or "
                              "multiple explicit files.")
    parser.add_argument("--output", type=str, default=train.BEST_MODEL_PATH,
                         help="Where to save the resulting checkpoint (default: best_model.pt).")
    parser.add_argument("--init-from", type=str, default=None,
                         help="Optional existing checkpoint to continue pretraining from "
                              "(default: fresh random init).")
    parser.add_argument("--buffer-size", type=int, default=200_000,
                         help="Reservoir capacity in positions. ~3.3KB/position "
                              "(default 200k ~ 660MB). Lower this if memory-constrained.")
    parser.add_argument("--examples-per-chunk", type=int, default=5_000,
                         help="New positions streamed in between training chunks.")
    parser.add_argument("--steps-per-chunk", type=int, default=200,
                         help="Gradient steps run per chunk.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=1.0,
                         help="Weight on value_loss relative to policy_loss. Raise this "
                              "if the value head still looks undertrained afterward.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                         help="Softens the one-hot policy target (0.0-0.2 typical). "
                              "Use if post-pretrain policy looks overconfident.")
    parser.add_argument("--blunder-penalty-weight", type=float, default=1.0,
                         help="Weight on the unlikelihood loss applied to moves the "
                              "PGN annotator flagged as an inaccuracy/mistake/blunder "
                              "(see move_quality_penalty()). These moves are never "
                              "imitated by the policy head regardless of this value; "
                              "raising it pushes probability on them down harder. "
                              "Set to 0.0 to disable (flagged moves are then simply "
                              "excluded from the policy loss, still contribute to the "
                              "value loss). Only affects sources with NAGs ('??'/'?'/"
                              "'?!') or Lichess-style 'Blunder.'/'Mistake.'/"
                              "'Inaccuracy.' comments -- no effect on PGNs without them.")
    parser.add_argument("--min-elo", type=int, default=None,
                         help="Skip games where either player's Elo is below this "
                              "(default: no filter -- use the full strength spread).")
    parser.add_argument("--value-target", type=str, default="outcome",
                         choices=["outcome", "eval", "blend"],
                         help="Where the value-head training target comes from. "
                              "'outcome' (default): the game's final result, same "
                              "value for every position in the game (original "
                              "AlphaZero-style behavior). 'eval': the position's PGN "
                              "%%eval annotation, falling back to outcome when a ply "
                              "has none. 'blend': a weighted mix of both (see "
                              "--eval-weight). NOTE: assumes eval annotations are "
                              "signed from White's perspective (standard Lichess "
                              "convention) -- verify this holds for your PGN source.")
    parser.add_argument("--eval-weight", type=float, default=0.5,
                         help="Weight on the eval-derived target vs. the outcome-"
                              "derived target when --value-target=blend (0=pure "
                              "outcome, 1=pure eval). Ignored otherwise.")
    parser.add_argument("--eval-scale", type=float, default=400.0,
                         help="Divisor (in pawns) used to squash centipawn evals to "
                              "[-1, 1] via tanh(pawns / eval_scale) for --value-target "
                              "eval/blend. Lower = more saturated/confident targets "
                              "from smaller advantages; higher = flatter/gentler.")
    parser.add_argument("--include-bots", action="store_true",
                         help="Include engine-vs-human games (skipped by default).")
    parser.add_argument("--max-examples", type=int, default=None,
                         help="Stop after streaming this many positions (for a quick test run).")
    parser.add_argument("--save-every-chunks", type=int, default=5,
                         help="Checkpoint save cadence, in chunks.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train.setup_logging()
    logger = __import__("logging").getLogger()
    device = torch.device(args.device)

    model = train.DualHeadResNet().to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        logger.info(f"Continuing pretraining from {args.init_from}.")
    else:
        logger.info("Starting from a fresh random init.")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    pgn_paths = resolve_pgn_paths(args.pgn)
    logger.info(f"Found {len(pgn_paths)} PGN file(s) to stream: "
                f"{', '.join(os.path.basename(p) for p in pgn_paths)}")
    if args.value_target == "outcome":
        logger.info("Value target: game outcome only.")
    elif args.value_target == "eval":
        logger.info(f"Value target: PGN %eval (scale={args.eval_scale}), "
                     "falling back to outcome where no eval is present.")
    else:
        logger.info(f"Value target: blend of eval and outcome "
                     f"(eval_weight={args.eval_weight}, scale={args.eval_scale}).")

    reservoir = Reservoir(args.buffer_size)
    chunk_count = 0
    since_last_chunk = 0

    flagged_seen = 0
    pbar = tqdm(desc="Streaming positions", unit="pos")
    for state, policy_idx, z, quality_penalty in iter_examples(
            pgn_paths, skip_bots=not args.include_bots,
            min_elo=args.min_elo,
            value_target=args.value_target,
            eval_weight=args.eval_weight,
            eval_scale=args.eval_scale):
        reservoir.add((state, policy_idx, z, quality_penalty))
        since_last_chunk += 1
        if quality_penalty > 0:
            flagged_seen += 1
        pbar.update(1)

        if args.max_examples and reservoir.total_seen >= args.max_examples:
            break

        if since_last_chunk >= args.examples_per_chunk and len(reservoir) >= args.batch_size:
            since_last_chunk = 0
            chunk_count += 1
            p_loss, v_loss, b_loss = train_chunk(model, optimizer, reservoir, args.steps_per_chunk,
                                                  args.batch_size, device, args.value_loss_weight,
                                                  args.label_smoothing, args.blunder_penalty_weight)
            logger.info(f"Chunk {chunk_count}: seen={reservoir.total_seen} "
                        f"buffer={len(reservoir)} flagged={flagged_seen} "
                        f"avg policy_loss={p_loss:.4f} avg value_loss={v_loss:.4f} "
                        f"avg blunder_loss={b_loss:.4f}")
            if chunk_count % args.save_every_chunks == 0:
                torch.save(model.state_dict(), args.output)
                logger.info(f"Saved checkpoint to {args.output}.")
    pbar.close()

    # Final polish pass + guaranteed save, even if the stream ended mid-chunk.
    if len(reservoir) >= args.batch_size:
        p_loss, v_loss, b_loss = train_chunk(model, optimizer, reservoir, args.steps_per_chunk,
                                              args.batch_size, device, args.value_loss_weight,
                                              args.label_smoothing, args.blunder_penalty_weight)
        logger.info(f"Final chunk: avg policy_loss={p_loss:.4f} avg value_loss={v_loss:.4f} "
                    f"avg blunder_loss={b_loss:.4f}")

    torch.save(model.state_dict(), args.output)
    logger.info(f"Done. Streamed {reservoir.total_seen} positions "
                f"({flagged_seen} flagged as inaccuracy/mistake/blunder, "
                f"policy-penalized rather than imitated) across {chunk_count} chunks. "
                f"Saved final checkpoint to {args.output}.")


if __name__ == "__main__":
    main()
