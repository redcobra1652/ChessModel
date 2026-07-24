#!/usr/bin/env python3
"""
stockfish_train.py -- Learn directly from Stockfish, adaptively.

Unlike self-play (train.py), the model's OWN moves during data
generation are never chosen by the model itself -- the policy target
always comes from Stockfish. This sidesteps the cold-start loop where a
weak model's self-play just teaches it its own bad habits -- it gets
real tactical signal from move one.

-- Data-generation design (competitive games + separate annotator) --
The GAME ITSELF is advanced by two EVENLY MATCHED Stockfish instances
(Stockfish-at-current-elo vs a second, independent Stockfish-at-current-elo
instance) playing each other. Because neither side outclasses the other,
outcomes vary naturally across wins, losses, and draws, so the value
target `z` spans the full [-1, 1] range instead of collapsing to a
constant. (Earlier versions of this script had the full-strength teacher
physically play one side of the board against the weak adaptive-Elo
opponent; the teacher won essentially every game, so every training
example had z ~= +1.0 and the value head learned to ignore the board
and just output a constant -- this file's current design specifically
fixes that failure mode.)

A SEPARATE, full-strength, never-rate-limited Stockfish instance
(`sf_teacher`) never plays a move on the board. It is used purely as an
external annotator: at EVERY ply, for BOTH sides, it analyses the
current position and its top move / evaluation become that ply's policy
target and eval-based value-target component. This keeps the policy
signal at full teacher strength (3200+ Elo judgement) even though the
physical game is played out by weaker, evenly-matched engines.

Stockfish's own strength is still adapted automatically, exactly as
before (calibration -> initial Elo -> promote on --promotion-threshold),
but now that ratchet controls the Elo of BOTH competitive-game engines
(kept in lockstep with each other) -- i.e. how strong the games are that
the teacher annotates. Progress is gated by a held-out deterministic
eval batch where the model's OWN weights (via train.search/MCTS), not
Stockfish, choose the moves.

Concretely, each generation:
  1. DATA GENERATION: play --batch-games games of
     (Stockfish@current_elo vs Stockfish@current_elo, two separate
     instances). Every ply of every game becomes a training example:
     (board_to_tensor(board) BEFORE the move, one-hot policy on the
     move the full-strength teacher would play from that position, a
     value target blending the teacher's per-move eval with the game's
     real final outcome -- both expressed from the side-to-move's
     perspective for that ply).
  2. TRAIN: push those examples into a ReplayBuffer (same class as
     train.py) and run --train-steps-per-gen optimizer steps.
  3. EVAL: the model's OWN weights (real MCTS via train.search, no
     Stockfish involved) play a small deterministic batch against the
     current-Elo Stockfish. This is the actual progress signal.
  4. PROMOTE: if the eval score clears --promotion-threshold, ratchet
     Stockfish's Elo target up by --elo-step and keep going.

No PGN, no interaction with pretrain.py. Checkpoints save to
best_model.pt (or --output) periodically and at the end.

-- UCI_Elo vs Skill Level --
UCI_LimitStrength + UCI_Elo is used throughout (not Skill Level, which
has no calibrated real-world unit and can't be stepped by "+100"). Every
full-strength instance has UCI_LimitStrength explicitly off.

Usage:
    python3 stockfish_train.py --stockfish-dir stockfish
"""

import argparse
import glob
import json
import logging
import os
import platform
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pickle
import random
import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import train  # reuse DualHeadResNet, ReplayBuffer, train_step, search, board/move helpers
from eval_game_logger import run_eval_batch_with_pgn  # noqa: E402 -- adds PGN sample saving on top of run_eval_batch
from endgame_data import generate_endgame_batch  # noqa: E402 -- synthetic K+Q/K+R/K+2R vs K mating-technique data

FALLBACK_MIN_ELO = 1320
FALLBACK_MAX_ELO = 3190
TRAJECTORY_LOG_DEFAULT = "stockfish_curriculum_log.jsonl"

log = logging.getLogger("stockfish_train")


def _is_executable(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    st = os.stat(path)
    return bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def _candidate_score(filename: str) -> int:
    name = filename.lower()
    score = 0
    machine = platform.machine().lower()
    is_apple_silicon_build = "apple-silicon" in name or "m1" in name or "arm64" in name
    is_x86_build = "x86-64" in name or "x86_64" in name
    if machine in ("arm64", "aarch64"):
        if is_apple_silicon_build:
            score += 100
        elif is_x86_build:
            score -= 50
    else:
        if is_x86_build:
            score += 100
        elif is_apple_silicon_build:
            score -= 100
    for feature, bonus in [("avx512", 40), ("vnni", 35), ("bmi2", 30), ("avx2", 25),
                            ("sse41-popcnt", 10), ("sse41", 5)]:
        if feature in name:
            score += bonus
            break
    if name == "stockfish":
        score += 15
    if name.endswith(".nnue") or name.endswith(".zip") or name.endswith(".tar"):
        score -= 1000
    return score


def find_stockfish_binary(stockfish_dir: str = None, explicit_path: str = None) -> str:
    if explicit_path:
        if not _is_executable(explicit_path):
            raise FileNotFoundError(
                f"--stockfish-path '{explicit_path}' doesn't exist or isn't executable. "
                f"Try: chmod +x '{explicit_path}' and xattr -d com.apple.quarantine '{explicit_path}'"
            )
        return explicit_path
    search_dir = stockfish_dir or "stockfish"
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(
            f"Stockfish directory '{search_dir}' not found. Pass --stockfish-dir or --stockfish-path."
        )
    candidates = [p for p in glob.glob(os.path.join(search_dir, "**", "stockfish*"), recursive=True)
                  if _is_executable(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No executable 'stockfish*' binary found under '{search_dir}'. Try: "
            f"chmod +x {search_dir}/<binary-name>  (and possibly "
            f"xattr -d com.apple.quarantine {search_dir}/<binary-name>). Or pass --stockfish-path."
        )
    candidates.sort(key=_candidate_score, reverse=True)
    chosen = candidates[0]
    log.info(f"Auto-detected Stockfish binary: {chosen}"
             + (f"  (others: {[os.path.basename(c) for c in candidates[1:]]})"
                if len(candidates) > 1 else ""))
    return chosen


def start_stockfish(path: str, threads: int, hash_mb: int) -> chess.engine.SimpleEngine:
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied launching Stockfish at '{path}'. Try "
            f"chmod +x '{path}' and xattr -d com.apple.quarantine '{path}', then retry."
        ) from e
    engine.configure({"Threads": threads, "Hash": hash_mb})
    log.info(f"Started Stockfish subprocess: {path}  (Threads={threads}, Hash={hash_mb}MB)")
    return engine


def get_elo_range(engine: chess.engine.SimpleEngine, min_override, max_override):
    opt = engine.options.get("UCI_Elo")
    if opt is not None and opt.min is not None and opt.max is not None:
        engine_min, engine_max = int(opt.min), int(opt.max)
    else:
        log.warning(f"UCI_Elo min/max not reported; falling back to [{FALLBACK_MIN_ELO}, {FALLBACK_MAX_ELO}].")
        engine_min, engine_max = FALLBACK_MIN_ELO, FALLBACK_MAX_ELO
    min_elo = int(min_override) if min_override is not None else engine_min
    max_elo = int(max_override) if max_override is not None else engine_max
    min_elo, max_elo = max(min_elo, engine_min), min(max_elo, engine_max)
    if min_elo >= max_elo:
        raise ValueError(f"Resolved Elo range invalid ({min_elo}-{max_elo}).")
    return min_elo, max_elo


def map_winrate_to_initial_elo(score: float, min_elo: int, max_elo: int) -> int:
    score = min(max(score, 0.0), 1.0)
    return int(round(min_elo + score * (max_elo - min_elo)))


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def eval_to_value(score: chess.engine.PovScore, mate_score_cp: int = 1000, cp_scale: float = 200.0) -> float:
    """Convert a Stockfish PovScore (from the side to move's perspective) into a
    scalar in [-1, 1] from White's perspective, using tanh(cp/cp_scale). Mates are
    clamped to a large-but-finite cp equivalent before scaling."""
    pov = score.white()
    if pov.is_mate():
        mate_in = pov.mate()
        cp = mate_score_cp if mate_in > 0 else -mate_score_cp
    else:
        cp = pov.score()
        if cp is None:
            cp = 0
    return float(np.tanh(cp / cp_scale))


def make_limit(movetime_ms, depth) -> chess.engine.Limit:
    if depth is not None:
        return chess.engine.Limit(depth=depth)
    return chess.engine.Limit(time=movetime_ms / 1000.0)


def log_trajectory(fh, record: dict):
    if fh is None:
        return
    record = dict(record)
    record["timestamp"] = time.time()
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def play_one_imitation_game(sf_teacher, sf_white, sf_black, opponent_limit, teacher_limit,
                             max_moves, eval_blend: float = 0.5,
                             eval_depth: int = None, eval_movetime_ms: int = None):
    """Play one data-generation game with two EVENLY MATCHED engines
    (sf_white vs sf_black, both at the current adaptive Elo) actually
    advancing the board. This produces a natural, varied distribution of
    wins/losses/draws -- unlike teacher-vs-weak-opponent, where the
    teacher wins ~100% of the time and every value target collapses to
    +1.0.

    The full-strength, unthrottled `sf_teacher` never plays a move. It is
    used purely as an external annotator: at EVERY ply (both sides), it
    is asked to analyse the current position, and its top move/eval
    become that ply's policy target and eval-based value target. This
    keeps the policy signal at full teacher strength even though the
    game itself is played by weaker, evenly-matched engines -- decoupling
    "whose move reaches the board" from "whose judgement the network is
    trained to imitate".

    Sign convention: every stored (state, policy_target, z) tuple is from
    the perspective of the side to move at that ply (standard AlphaZero
    convention: z > 0 means "good for whoever's turn it is"). Both the
    per-move teacher eval and the final game outcome are flipped onto
    that same side-to-move perspective before blending, so the two
    contributions to z can never point in inconsistent directions.
    """
    board = chess.Board()
    examples = []
    ply = 0
    eval_limit = None
    if eval_blend > 0.0:
        eval_limit = make_limit(eval_movetime_ms, eval_depth)

    # A single 2-thread executor is reused across all plies so we don't pay
    # thread-creation overhead every ply. Both Stockfish processes are
    # completely independent (separate binaries, separate pipes, no shared
    # state), so running them concurrently is safe. board is passed as a
    # copy to each call so neither future can observe a mutation made by
    # the other before it has snapshotted the position.
    with ThreadPoolExecutor(max_workers=2) as ex:
        while not board.is_game_over(claim_draw=False) and not board.is_repetition(3) and not board.can_claim_fifty_moves() and ply < max_moves:
            side_to_move = board.turn  # capture BEFORE any mutation of `board`
            mover = sf_white if side_to_move == chess.WHITE else sf_black

            # --- Teacher annotation and game advancement run concurrently.
            # sf_teacher.analyse() and mover.play() are completely independent
            # -- different engine processes analysing the same position -- so
            # both can run simultaneously, cutting per-ply wall time roughly
            # in half compared to the previous sequential approach.
            # board is safe to pass directly: board.push() only happens after
            # both futures have returned, so there is no concurrent mutation. ---
            if eval_blend > 0.0:
                fut_teacher = ex.submit(sf_teacher.analyse, board, eval_limit)
                fut_mover   = ex.submit(mover.play, board, opponent_limit)
                info        = fut_teacher.result()
                teacher_move    = info["pv"][0]
                move_eval_white = eval_to_value(info["score"])
                played_move     = fut_mover.result().move
            else:
                # eval_blend == 0: no analyse() needed, just play() sequentially
                # (teacher only needs to return a move, not a score).
                teacher_move = sf_teacher.play(board, teacher_limit).move
                move_eval_white = None
                played_move = mover.play(board, opponent_limit).move

            mirror = side_to_move == chess.BLACK
            policy_target = np.zeros(train.ACTION_SIZE, dtype=np.float32)
            policy_target[train.move_policy_index(teacher_move, mirror)] = 1.0
            # Tensorize BEFORE pushing any move, so the state exactly matches
            # the position the policy/value targets were computed for -- never
            # the post-move position (no state/label leakage or off-by-one).
            state = train.board_to_tensor(board)
            examples.append([state, policy_target, side_to_move, move_eval_white])

            # --- Actual game advancement: evenly matched engines play each
            # other. This is what makes win/loss/draw outcomes vary, instead
            # of the teacher steamrolling a low-Elo opponent every game. ---
            assert played_move in board.legal_moves, \
                f"ILLEGAL MOVE '{played_move}' at FEN '{board.fen()}'"
            board.push(played_move)
            ply += 1

    terminal, result_for_mover = train.position_outcome(board)
    # result_for_mover is from the perspective of the side to move in the
    # FINAL (terminal or move-limit) position. Convert to a White-relative
    # scalar so it can be uniformly re-projected onto each example's own
    # side-to-move below.
    z_white = 0.0 if not terminal else (result_for_mover if board.turn == chess.WHITE else -result_for_mover)

    finished = []
    for state, policy_target, side_to_move, move_eval_white in examples:
        # Flip the White-relative outcome onto this ply's side-to-move.
        outcome_z = z_white if side_to_move == chess.WHITE else -z_white
        if move_eval_white is None:
            z = outcome_z
        else:
            # Flip the White-relative teacher eval onto this ply's
            # side-to-move using the exact same convention as outcome_z,
            # so eval and outcome are never combined with mismatched signs.
            eval_z = move_eval_white if side_to_move == chess.WHITE else -move_eval_white
            z = eval_blend * eval_z + (1.0 - eval_blend) * outcome_z
        z = float(np.clip(z, -1.0, 1.0))
        finished.append((state, policy_target, z))
    return finished, ply, z_white


def generate_batch(sf_teacher, sf_opponent_a, sf_opponent_b, opponent_limit, teacher_limit, max_moves, n_games, desc,
                    eval_blend: float = 0.5, eval_depth: int = None, eval_movetime_ms: int = None):
    """Generate a batch of data-gen games.

    sf_opponent_a / sf_opponent_b are two evenly-matched engine instances
    (both configured to the same adaptive Elo by the caller) that
    alternate colours every game so there's no systematic White/Black
    bias in the resulting dataset. sf_teacher annotates every ply of
    every game (see play_one_imitation_game) but never moves a piece.
    """
    all_examples = []
    wins = losses = draws = 0  # tracked from White's perspective, purely for logging/diagnostics
    for g in tqdm(range(n_games), desc=desc, leave=False):
        # Alternate which physical engine instance plays White so that any
        # subtle asymmetry between the two SimpleEngine processes (e.g.
        # hash table warmth) doesn't correlate with colour.
        a_is_white = (g % 2 == 0)
        sf_white, sf_black = (sf_opponent_a, sf_opponent_b) if a_is_white else (sf_opponent_b, sf_opponent_a)
        examples, plies, z_white = play_one_imitation_game(
            sf_teacher, sf_white, sf_black, opponent_limit, teacher_limit, max_moves,
            eval_blend=eval_blend, eval_depth=eval_depth, eval_movetime_ms=eval_movetime_ms)
        all_examples.extend(examples)
        if z_white > 0:
            wins += 1
        elif z_white < 0:
            losses += 1
        else:
            draws += 1
        log.info(f"  game {g + 1}/{n_games}: {plies} plies, result(white)={z_white:+.1f}, "
                 f"{len(examples)} examples")
    total = wins + losses + draws
    if total:
        log.info(f"  batch outcome distribution (White's perspective): "
                 f"{wins}W-{losses}L-{draws}D ({wins/total:.1%} / {losses/total:.1%} / {draws/total:.1%})")
        if wins / total > 0.95 or losses / total > 0.95:
            log.warning("  batch is >95% one-sided by outcome -- value targets may still be "
                        "poorly balanced; check that the two engine instances are truly evenly "
                        "matched (same Elo config) rather than one out-strengthening the other.")
    return all_examples


def play_one_eval_game(mcts_proc, sf_engine, model, device, sims, threads, max_moves,
                        model_is_white, sf_limit, game_index):
    """Play one eval game (model MCTS vs Stockfish).

    Returns:
        outcome:        "win" | "loss" | "draw"
        ply:            total plies played
        model_records:  list of (fen_before_move, move_played_uci) for every
                        move the model made -- used by analyse_eval_game to
                        build corrective training examples when the game is lost.
    """
    train.SELF_PLAY_MODE = False
    train.CURRENT_DEVICE = device
    train.CURRENT_MODEL = model
    model_color = chess.WHITE if model_is_white else chess.BLACK
    board = chess.Board()
    ply = 0
    model_records = []  # (fen_before, move_uci) for each model move
    while not board.is_game_over(claim_draw=False) and not board.is_repetition(3) and not board.can_claim_fifty_moves() and ply < max_moves:
        if board.turn == model_color:
            mate_move = train.find_immediate_mate(board)
            if mate_move is not None:
                move = mate_move
            else:
                visits, _ = train.search(mcts_proc, board, sims=sims, threads=threads)
                best_uci = train.pick_safe_move_from_visits(board, visits, temperature=0.0)
                move = chess.Move.from_uci(best_uci)
            assert move in board.legal_moves
            model_records.append((board.fen(), move.uci()))
            board.push(move)
        else:
            sf_move = sf_engine.play(board, sf_limit).move
            assert sf_move in board.legal_moves
            board.push(sf_move)
        ply += 1
    terminal, result_for_mover = train.position_outcome(board)
    if not terminal or result_for_mover == 0.0:
        return "draw", ply, model_records
    winner_is_white = (result_for_mover == 1.0) == (board.turn == chess.WHITE)
    model_won = winner_is_white == model_is_white
    outcome = "win" if model_won else "loss"
    return outcome, ply, model_records


def analyse_eval_game(sf_teacher, model_records, model_color, z_outcome,
                       corrective_limit, cp_threshold: float = 30.0,
                       eval_blend: float = 0.5):
    # NOTE: z_outcome / eval_blend are accepted for call-site / CLI
    # stability (--corrective-eval-blend) but are not currently used to
    # compute a value target: corrective examples carry no `z` because
    # corrective_train_step only ever trains the policy head (see the
    # comment above `examples.append` below for the rationale).
    """Analyse every model move from a lost eval game and return corrective
    training examples for moves that significantly hurt the model's position.

    For each model move we:
      1. Evaluate the position BEFORE the move (Stockfish's view of what the
         model *should* have done) -- this gives us sf_best_move and eval_before.
      2. Evaluate the position AFTER the model's move -- eval_after.
      3. Compute cp_drop = (eval_before - eval_after) from the model's own
         perspective (positive = the move hurt the model, regardless of colour).
         White perspective: drop = eval_before_white - eval_after_white
         Black perspective: drop = eval_after_white - eval_before_white
         (because a drop in White's score is good for Black)
      4. If cp_drop >= cp_threshold, emit a corrective example:
           policy_target = one-hot on sf_best_move  (teach "play this instead")
           z = eval_blend * eval_z + (1 - eval_blend) * outcome_z
             where eval_z is derived from eval_before (what the position was
             *worth* before the blunder, from the model's perspective) and
             outcome_z is the actual game result.

    All cp values are in centipawns from White's perspective (as Stockfish
    reports them). We convert to the model-side perspective when computing
    the drop so that the threshold is colour-agnostic.

    Args:
        sf_teacher:       full-strength Stockfish engine (SimpleEngine)
        model_records:    list of (fen_before, move_played_uci) from play_one_eval_game
        model_color:      chess.WHITE or chess.BLACK
        z_outcome:        game outcome from the model's perspective (+1 win, -1 loss)
        corrective_limit: chess.engine.Limit for each analyse() call
        cp_threshold:     centipawn drop that triggers a corrective example (default 30)
        eval_blend:       weight on per-move eval in the value target (same as --eval-blend)

    Returns:
        list of (state_tensor, policy_target, z) corrective training examples
    """
    examples = []
    for fen_before, move_played_uci in model_records:
        board_before = chess.Board(fen_before)

        # --- eval BEFORE the model's move (gives sf_best_move + score) ---
        try:
            info_before = sf_teacher.analyse(board_before, corrective_limit)
        except Exception as e:
            log.warning(f"analyse_eval_game: analyse before move failed ({e}); skipping position.")
            continue

        score_before = info_before["score"]  # PovScore from side-to-move's view
        # Convert to centipawns from White's perspective for consistent sign convention
        if score_before.white().is_mate():
            mate_in = score_before.white().mate()
            cp_before_white = 10000 if mate_in > 0 else -10000
        else:
            cp_before_white = score_before.white().score() or 0

        sf_best_move = info_before.get("pv", [None])[0]
        if sf_best_move is None:
            continue  # no legal moves (shouldn't happen mid-game, skip)

        # --- eval AFTER the model's move ---
        board_after = board_before.copy()
        move_played = chess.Move.from_uci(move_played_uci)
        board_after.push(move_played)
        try:
            info_after = sf_teacher.analyse(board_after, corrective_limit)
        except Exception as e:
            log.warning(f"analyse_eval_game: analyse after move failed ({e}); skipping position.")
            continue

        score_after = info_after["score"]
        if score_after.white().is_mate():
            mate_in = score_after.white().mate()
            cp_after_white = 10000 if mate_in > 0 else -10000
        else:
            cp_after_white = score_after.white().score() or 0

        # --- cp drop from the model's perspective ---
        # White wants higher scores, Black wants lower scores.
        # A positive cp_drop means the move hurt the model.
        if model_color == chess.WHITE:
            cp_drop = cp_before_white - cp_after_white
        else:
            cp_drop = cp_after_white - cp_before_white

        if cp_drop < cp_threshold:
            continue  # move wasn't bad enough; skip

        # --- build corrective training example (4-tuple) ---
        # good_move_idx: Stockfish's best move index  -> loss will pull this UP
        # bad_move_idx:  the blunder the model played -> loss will push this DOWN
        # cp_drop:       blunder magnitude in centipawns, used as per-example loss weight
        #
        # Note: no value target is computed here. corrective_train_step
        # intentionally only trains the policy head (see its docstring) --
        # blending a losing game's outcome into the value target for a
        # position that was objectively fine before the blunder would
        # inject a noisy, self-contradictory signal into the value head,
        # so we deliberately don't touch it in corrective training.
        mirror = board_before.turn == chess.BLACK
        good_move_idx = train.move_policy_index(sf_best_move, mirror)
        bad_move_idx  = train.move_policy_index(move_played, mirror)
        state = train.board_to_tensor(board_before)

        examples.append((state, good_move_idx, bad_move_idx, cp_drop))

    return examples


def corrective_train_step(model, optimizer, corrective_buffer: list,
                           batch_size: int, device,
                           penalty_weight: float = 1.0):
    """One gradient step using corrective examples from lost eval games.

    Each example is a 4-tuple: (state, good_move_idx, bad_move_idx, cp_drop).

    The loss has three terms:
      1. POSITIVE policy loss: cross-entropy toward good_move_idx (Stockfish's
         best move). This is the standard imitation signal -- pull the good move up.
      2. NEGATIVE policy penalty: -log(1 - p_bad + eps) on bad_move_idx (the
         blunder). This directly pushes the probability of the blunder move DOWN,
         which the normal buffer never does. Each example is weighted by its
         cp_drop normalised within the batch, so a 200cp blunder gets proportionally
         more gradient than a 35cp one (Option C).

    Args:
        corrective_buffer: list of (state, good_idx, bad_idx, cp_drop) 4-tuples
        penalty_weight:    scaling factor on the negative penalty term (default 1.0,
                           i.e. equal weight to the positive cross-entropy term).

    Returns:
        (policy_loss, penalty_loss) floats -- value head is intentionally not trained here
    """
    n = min(batch_size, len(corrective_buffer))
    batch = random.sample(corrective_buffer, n)
    states, good_idxs, bad_idxs, cp_drops = zip(*batch)

    states_t    = torch.from_numpy(np.stack(states)).float().to(device)
    good_t      = torch.tensor(good_idxs, dtype=torch.long).to(device)
    bad_t       = torch.tensor(bad_idxs,  dtype=torch.long).to(device)
    cp_drops_t  = torch.tensor(cp_drops,  dtype=torch.float32).to(device)

    # Normalise cp_drop within this batch to [0, 1] so it acts as a relative
    # weight rather than an absolute scale.  Add a small floor so every example
    # contributes at least a little gradient even if cp_drops are all equal.
    cp_min, cp_max = cp_drops_t.min(), cp_drops_t.max()
    if cp_max > cp_min:
        weights = (cp_drops_t - cp_min) / (cp_max - cp_min + 1e-8)
    else:
        weights = torch.ones_like(cp_drops_t)
    weights = weights + 0.1          # floor: even smallest blunder gets 10% weight
    weights = weights / weights.sum() * n  # re-normalise so mean weight == 1

    # Only run the policy head through the loss -- corrective training is purely
    # about fixing move selection, not rewriting value estimates.  The value head
    # already has a consistent estimate of these positions from imitation training;
    # feeding it contradictory z targets (blended from a losing outcome against
    # a position that was objectively fine before the blunder) would just corrupt it.
    logits, _ = model(states_t)
    log_probs = F.log_softmax(logits, dim=1)   # (N, ACTION_SIZE)
    probs     = log_probs.exp()

    # 1. Positive: pull good move up (standard cross-entropy)
    pos_loss_per = -log_probs.gather(1, good_t.unsqueeze(1)).squeeze(1)  # (N,)
    policy_loss  = (weights * pos_loss_per).mean()

    # 2. Negative: push bad move down
    # -log(1 - p_bad) approaches inf as p_bad -> 1, giving a strong gradient
    # when the model is confidently wrong.
    p_bad         = probs.gather(1, bad_t.unsqueeze(1)).squeeze(1).clamp(max=1 - 1e-6)
    neg_loss_per  = -torch.log(1.0 - p_bad + 1e-8)
    penalty_loss  = penalty_weight * (weights * neg_loss_per).mean()

    loss = policy_loss + penalty_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), penalty_loss.item()


def run_eval_batch(mcts_proc, sf_engine, model, device, sims, threads, max_moves,
                    n_games, sf_limit, elo, desc,
                    sf_teacher=None, corrective_limit=None,
                    corrective_cp_threshold: float = 30.0,
                    corrective_eval_blend: float = 0.5):
    """Run n_games eval games (model MCTS vs Stockfish) and return win/loss/draw
    stats plus corrective training examples extracted from lost games.

    Corrective examples are only generated when sf_teacher and corrective_limit
    are both provided (i.e. --corrective-analysis is enabled).  For every game
    the model loses, every model move is analysed by sf_teacher; moves where the
    cp dropped by >= corrective_cp_threshold (from the model's perspective) yield
    a training example with sf_teacher's best move as the policy target.

    Args:
        sf_teacher:               full-strength Stockfish used for per-move analysis.
                                  Pass None to disable corrective analysis entirely.
        corrective_limit:         chess.engine.Limit for each sf_teacher.analyse() call.
        corrective_cp_threshold:  centipawn drop that triggers a corrective example.
        corrective_eval_blend:    blend weight for eval vs outcome in the value target.

    Returns:
        dict with keys: wins, losses, draws, total, score, corrective_examples
    """
    wins = losses = draws = 0
    all_corrective = []
    do_corrective = sf_teacher is not None and corrective_limit is not None
    for g in tqdm(range(n_games), desc=desc, leave=False):
        model_is_white = (g % 2 == 0)
        outcome, plies, model_records = play_one_eval_game(
            mcts_proc, sf_engine, model, device, sims, threads,
            max_moves, model_is_white, sf_limit, g)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
            if do_corrective:
                model_color = chess.WHITE if model_is_white else chess.BLACK
                corrective = analyse_eval_game(
                    sf_teacher, model_records, model_color,
                    z_outcome=-1.0,  # model lost
                    corrective_limit=corrective_limit,
                    cp_threshold=corrective_cp_threshold,
                    eval_blend=corrective_eval_blend,
                )
                log.info(f"  corrective analysis: {len(corrective)} examples from lost game {g + 1}")
                all_corrective.extend(corrective)
        else:
            draws += 1
        log.info(f"  eval game {g + 1}/{n_games} (model {'White' if model_is_white else 'Black'}, "
                 f"Elo {elo}): {outcome}  ({plies} plies)")
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total else 0.0
    return {"wins": wins, "losses": losses, "draws": draws, "total": total, "score": score,
            "corrective_examples": all_corrective}


def _save_checkpoint(path: str, model, optimizer, current_lr: float,
                     global_gen: int, num_promotions: int):
    """Save full training state: model weights, optimizer state (Adam momentum
    buffers), current LR, global generation count, and promotion count."""
    torch.save({
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "lr":             current_lr,
        "global_gen":     global_gen,
        "num_promotions": num_promotions,
    }, path)


def run_promotion_probe(mcts_proc, sf_probe, model, device, sims, threads, max_moves,
                        n_games, sf_limit_fn, probe_elo: int,
                        threshold: float, desc: str) -> float:
    """Run a short eval batch against `probe_elo` Stockfish.

    `sf_limit_fn` is a callable(elo) -> chess.engine.Limit (so the caller
    can reuse the same engine with a fresh configure() call each time).
    Returns the model's score (wins + 0.5*draws) / total.
    """
    sf_probe.configure({"UCI_LimitStrength": True, "UCI_Elo": probe_elo})
    result = run_eval_batch(mcts_proc, sf_probe, model, device, sims, threads,
                            max_moves, n_games, sf_limit_fn(probe_elo), elo=probe_elo,
                            desc=desc)
    result.pop("corrective_examples", None)
    return result["score"]


def main():
    parser = argparse.ArgumentParser(
        description="Train a DualHeadResNet to imitate Stockfish, with Stockfish's own "
                    "strength adapted to the model's progress.")
    parser.add_argument("--model", type=str, default=train.BEST_MODEL_PATH)
    parser.add_argument("--output", type=str, default=train.BEST_MODEL_PATH)
    parser.add_argument("--save-every-gens", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--endgame-positions-per-gen", type=int, default=40,
                         help="Synthetic K+Q/K+R/K+2R vs K positions to generate and train on each generation. 0 disables.")
    parser.add_argument("--endgame-max-moves", type=int, default=40,
                         help="Move cap for synthetic endgame games (trivial for full-strength Stockfish; should resolve fast).")
    parser.add_argument("--endgame-teacher-movetime-ms", type=int, default=None,
                         help="Per-move time limit for the full-strength Stockfish teacher during synthetic endgame generation. If unset, uses --endgame-teacher-depth instead.")
    parser.add_argument("--endgame-teacher-depth", type=int, default=None,
                         help="Search depth for the full-strength Stockfish teacher during synthetic endgame generation. If both this and --endgame-teacher-movetime-ms are set, depth wins (see make_limit). If neither is set, defaults to depth=12.")
    parser.add_argument("--stockfish-dir", type=str, default="stockfish")
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--stockfish-hash", type=int, default=16)
    parser.add_argument("--teacher-movetime-ms", type=int, default=200)
    parser.add_argument("--teacher-depth", type=int, default=None)
    parser.add_argument("--eval-blend", type=float, default=0.5,
                         help="Weight on per-move Stockfish eval in the value target, "
                              "blended as eval_blend*eval + (1-eval_blend)*outcome. "
                              "0.0 disables per-move eval and uses pure game outcome (old behavior).")
    parser.add_argument("--eval-movetime-ms", type=int, default=None,
                         help="Movetime for the teacher's per-move analyse() call used for eval-blending. "
                              "Defaults to --teacher-movetime-ms if not set.")
    parser.add_argument("--eval-depth", type=int, default=None,
                         help="Depth for the teacher's per-move analyse() call used for eval-blending. "
                              "Defaults to --teacher-depth if not set.")
    parser.add_argument("--calibration-games", type=int, default=16)
    parser.add_argument("--calibration-movetime-ms", type=int, default=300)
    parser.add_argument("--calibration-depth", type=int, default=None)
    parser.add_argument("--initial-elo", type=int, default=None)
    parser.add_argument("--batch-games", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=50)
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--elo-step", type=int, default=100)
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--max-elo", type=int, default=None)
    parser.add_argument("--max-batches-per-level", type=int, default=10)
    parser.add_argument("--batch-games-growth", type=int, default=10,
                         help="Amount to add to --batch-games for each consecutive "
                              "generation that fails to clear --promotion-threshold.")
    parser.add_argument("--max-batch-games", type=int, default=100,
                         help="Ceiling on how large batch-games can grow after repeated stalls.")
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--train-steps-per-gen", type=int, default=200)
    parser.add_argument("--steps-per-buffer-example", type=float, default=None,
                         help="If set, train_steps_this_gen = max(train_steps_per_gen, "
                              "steps_per_buffer_example * len(buffer)). Scales training "
                              "effort with buffer size instead of using a fixed step count "
                              "every generation. Note: 1 full epoch over the buffer "
                              "corresponds to steps_per_buffer_example = 1/batch_size "
                              "(~0.0078 at batch_size=128); e.g. 0.06 means ~8 passes "
                              "over each example per generation.")
    parser.add_argument("--max-train-steps-per-gen", type=int, default=2000,
                         help="Ceiling on train steps per generation when using "
                              "--steps-per-buffer-example, to bound wall-clock time.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None,
                         help="Learning rate. When resuming from a checkpoint, the saved LR is "
                              "used by default; pass --lr to override it explicitly. "
                              "When starting fresh (no checkpoint), defaults to 1e-3.")
    parser.add_argument("--lr-decay", type=float, default=0.97,
                         help="Per-generation exponential LR decay factor applied every generation "
                              "before training: lr = max(lr_floor, initial_lr * decay**global_gen). "
                              "0.97-0.98 for slow decay, 1.0 to disable. (default: 0.97)")
    parser.add_argument("--lr-floor", type=float, default=1e-5,
                         help="Hard lower bound on LR. Neither exponential decay nor promotion "
                              "multipliers can push LR below this value. (default: 1e-5)")
    parser.add_argument("--promotion-lr-multiplier", type=float, default=0.6,
                         help="LR is multiplied by this factor on each promotion event. "
                              "Compounds across promotions (3 promotions = 0.6^3 = 0.216x). "
                              "(default: 0.6)")
    parser.add_argument("--probe-games", type=int, default=10,
                         help="Number of eval games to play against each successive probe Elo "
                              "after a promotion (probe starts at promoted_elo + elo_step). "
                              "(default: 10)")
    parser.add_argument("--probe-threshold", type=float, default=None,
                         help="Win-rate threshold to clear a probe level and advance to the next. "
                              "Defaults to --promotion-threshold if not set.")
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--trajectory-log", type=str, default=TRAJECTORY_LOG_DEFAULT)
    parser.add_argument("--buffer-path", type=str, default="replay_buffer.pkl",
                         help="Path to persist/restore the replay buffer across restarts. "
                              "Saved after every generation. Set to '' to disable. (default: replay_buffer.pkl)")

    # --- corrective analysis: learn from eval-game losses ---
    parser.add_argument("--corrective-analysis", action="store_true", default=False,
                         help="After each eval game the model loses, analyse every model "
                              "move with the full-strength teacher Stockfish. Moves where "
                              "the eval dropped by >= --corrective-cp-threshold (from the "
                              "model's perspective, colour-adjusted) produce a corrective "
                              "training example: policy target = Stockfish's best move at "
                              "that position, value target blended per --corrective-eval-blend. "
                              "Examples are pushed into the replay buffer alongside normal "
                              "imitation examples and trained on in the same generation.")
    parser.add_argument("--corrective-cp-threshold", type=float, default=30.0,
                         help="Centipawn drop (from model's perspective) that triggers a "
                              "corrective example. 30 cp is roughly one tempo / minor inaccuracy. "
                              "Lower = more examples but noisier; higher = fewer but more "
                              "egregious mistakes only. (default: 30)")
    parser.add_argument("--corrective-eval-movetime-ms", type=int, default=None,
                         help="Movetime in ms for each Stockfish analyse() call during "
                              "corrective analysis. Defaults to --teacher-movetime-ms.")
    parser.add_argument("--corrective-eval-depth", type=int, default=None,
                         help="Depth for each Stockfish analyse() call during corrective "
                              "analysis. Overrides --corrective-eval-movetime-ms if set.")
    parser.add_argument("--corrective-steps-per-gen", type=int, default=50,
                         help="Number of corrective gradient steps per generation. "
                              "Each step samples --batch-size examples from the corrective "
                              "buffer and applies both positive (good move up) and negative "
                              "(bad move down) policy losses. (default: 50)")
    parser.add_argument("--corrective-penalty-weight", type=float, default=1.0,
                         help="Scaling factor on the negative penalty term in corrective "
                              "training. 1.0 = equal weight to the positive cross-entropy "
                              "term. Higher = push bad moves down harder. (default: 1.0)")
    parser.add_argument("--corrective-buffer-size", type=int, default=5000,
                         help="Maximum number of corrective examples to retain across "
                              "generations. Oldest examples are dropped when full. (default: 5000)")
    parser.add_argument("--corrective-eval-blend", type=float, default=0.5,
                         help="Blend weight for the value target in corrective examples: "
                              "blend*eval_z + (1-blend)*outcome_z. eval_z is derived from "
                              "Stockfish's position eval BEFORE the blunder (what the position "
                              "was worth before the mistake); outcome_z is -1 (the model lost). "
                              "(default: 0.5)")

    args = parser.parse_args()

    train.setup_logging()
    device = torch.device(args.device)
    model = train.DualHeadResNet().to(device)

    # --- Checkpoint load: supports both old format (raw state_dict) and new
    # format (dict with model/optimizer/lr/global_gen/num_promotions). ---
    restored_lr = None
    global_gen = 0
    num_promotions = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or 1e-3, weight_decay=1e-4)

    try:
        raw = torch.load(args.model, map_location=device)
        if isinstance(raw, dict) and "model" in raw:
            # New-format checkpoint
            model.load_state_dict(raw["model"])
            opt_state = raw.get("optimizer")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
                # Move optimizer state tensors to the right device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                log.info("Restored optimizer state from checkpoint.")
            else:
                log.warning(
                    "Checkpoint has no optimizer state (was cleared during migration). "
                    "Adam momentum buffers will reinitialize from scratch -- expect a "
                    "brief update spike for the first 1-2 generations."
                )
            restored_lr    = raw.get("lr", 1e-3)
            global_gen     = raw.get("global_gen", 0)
            num_promotions = raw.get("num_promotions", 0)
            if args.lr is not None:
                log.info(
                    f"--lr {args.lr:.2e} overrides checkpoint LR {restored_lr:.2e}."
                )
                restored_lr = args.lr
            log.info(
                f"Loaded checkpoint from '{args.model}': "
                f"global_gen={global_gen}, num_promotions={num_promotions}, lr={restored_lr:.2e}."
            )
        else:
            # Old-format checkpoint: raw state_dict only
            model.load_state_dict(raw)
            log.info(
                f"Loaded legacy weights-only checkpoint from '{args.model}'. "
                f"Optimizer state and global_gen not available; starting fresh for those."
            )
    except FileNotFoundError:
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr": args.lr or 1e-3,
                    "global_gen": 0,
                    "num_promotions": 0}, args.model)
        log.info(f"No existing '{args.model}' found; initialized and saved a fresh checkpoint.")

    model.eval()

    # current_lr: use restored value if checkpoint had one (may already be
    # overridden by --lr above), else fall back to --lr or the default 1e-3.
    current_lr = restored_lr if restored_lr is not None else (args.lr or 1e-3)
    # Apply it to the optimizer param groups (in case we loaded old-format
    # and the optimizer was freshly constructed).
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr
    buffer = train.ReplayBuffer(args.buffer_size)

    if args.buffer_path and os.path.exists(args.buffer_path):
        try:
            with open(args.buffer_path, "rb") as f:
                saved = pickle.load(f)
            if "decisive" in saved or "draws" in saved:
                # Backward-compatible load: old buffer format from before
                # the decisive/draws split was removed. Merge both pools
                # into the new unified deque (order doesn't matter --
                # sampling is uniform now).
                n_loaded = 0
                for item in saved.get("decisive", []):
                    buffer.buffer.append(item)
                    n_loaded += 1
                for item in saved.get("draws", []):
                    buffer.buffer.append(item)
                    n_loaded += 1
                log.info(f"Restored replay buffer from '{args.buffer_path}' (old decisive/draws "
                         f"format, merged into unified buffer): {len(buffer)} examples.")
            else:
                for item in saved.get("buffer", []):
                    buffer.buffer.append(item)
                log.info(f"Restored replay buffer from '{args.buffer_path}': {len(buffer)} examples.")
        except Exception as e:
            log.warning(f"Could not load replay buffer from '{args.buffer_path}': {e}. Starting fresh.")

    train.compile_engine()
    mcts_proc = train.start_engine()

    sf_path = find_stockfish_binary(args.stockfish_dir, args.stockfish_path)
    sf_teacher = start_stockfish(sf_path, args.stockfish_threads, args.stockfish_hash)
    # Two SEPARATE adaptive-strength instances that will play each other
    # during data generation (see generate_batch). Both are always kept
    # at the SAME Elo target as each other -- they exist as two engine
    # instances only so a real game can be played (one process can't play
    # both sides of a chess.engine.SimpleEngine game concurrently), not to
    # create any strength asymmetry.
    sf_adaptive = start_stockfish(sf_path, args.stockfish_threads, args.stockfish_hash)
    sf_adaptive_b = start_stockfish(sf_path, args.stockfish_threads, args.stockfish_hash)
    sf_teacher.configure({"UCI_LimitStrength": False})

    trajectory_fh = open(args.trajectory_log, "a") if args.trajectory_log else None

    try:
        min_elo, max_elo = get_elo_range(sf_adaptive, args.min_elo, args.max_elo)
        log.info(f"Stockfish adaptive Elo range in use: [{min_elo}, {max_elo}].")
        teacher_limit = make_limit(args.teacher_movetime_ms, args.teacher_depth)
        adaptive_limit = make_limit(args.stockfish_movetime_ms, args.stockfish_depth)
        eval_movetime_ms = args.eval_movetime_ms if args.eval_movetime_ms is not None else args.teacher_movetime_ms
        eval_depth = args.eval_depth if args.eval_depth is not None else args.teacher_depth
        if args.eval_blend > 0.0:
            log.info(f"Value target blend: {args.eval_blend:.2f}*eval + {1 - args.eval_blend:.2f}*outcome "
                     f"(teacher analyse limit: {'depth ' + str(eval_depth) if eval_depth is not None else str(eval_movetime_ms) + 'ms'}).")

        # Corrective analysis setup
        corrective_limit = None
        if args.corrective_analysis:
            corr_movetime = (args.corrective_eval_movetime_ms
                             if args.corrective_eval_movetime_ms is not None
                             else args.teacher_movetime_ms)
            corrective_limit = make_limit(corr_movetime, args.corrective_eval_depth)
            log.info(
                f"Corrective analysis ENABLED: threshold={args.corrective_cp_threshold:.0f}cp, "
                f"eval_blend={args.corrective_eval_blend:.2f}, "
                f"limit={'depth ' + str(args.corrective_eval_depth) if args.corrective_eval_depth is not None else str(corr_movetime) + 'ms'}."
            )

        if args.initial_elo is not None:
            current_elo = clamp(args.initial_elo, min_elo, max_elo)
            log.info(f"Skipping calibration; starting at Elo {current_elo}.")
        else:
            log.info(f"=== Calibration: {args.calibration_games} eval games vs full-strength Stockfish ===")
            sf_adaptive.configure({"UCI_LimitStrength": False})
            calib_limit = make_limit(args.calibration_movetime_ms, args.calibration_depth)
            calib = run_eval_batch(mcts_proc, sf_adaptive, model, device, args.sims, args.threads,
                                    args.max_moves, args.calibration_games, calib_limit,
                                    elo="full-strength", desc="Calibration")
            calib.pop("corrective_examples", None)  # not generated during calibration
            log.info(f"Calibration result: {calib['wins']}-{calib['losses']}-{calib['draws']} "
                     f"(score {calib['score']:.1%}).")
            log_trajectory(trajectory_fh, {"phase": "calibration", "elo": None, **calib})
            current_elo = clamp(map_winrate_to_initial_elo(calib["score"], min_elo, max_elo), min_elo, max_elo)
            log.info(f"Calibration score {calib['score']:.1%} -> initial Elo target {current_elo}.")

        generation = 0
        stalled_batches = 0
        batch_games = args.batch_games
        corrective_buffer = []  # separate from ReplayBuffer; holds 5-tuples for corrective_train_step
        probe_threshold = args.probe_threshold if args.probe_threshold is not None else args.promotion_threshold

        # A fifth Stockfish instance used exclusively for promotion probes so
        # we can reconfigure its Elo freely without touching the data-gen engines.
        sf_probe = start_stockfish(sf_path, args.stockfish_threads, args.stockfish_hash)
        sf_probe.configure({"UCI_LimitStrength": True, "UCI_Elo": current_elo})

        while True:
            generation += 1
            global_gen += 1  # persists across restarts via checkpoint

            # --- Exponential LR decay ---
            # Recompute from initial_lr and global_gen on every generation so
            # the schedule is deterministic and restart-stable. The promotion
            # multiplier is already baked into current_lr (it's saved/loaded),
            # so we apply decay on top of *current_lr* rather than args.lr to
            # avoid overwriting the compounded multipliers.
            decayed_lr = current_lr * (args.lr_decay ** 1)  # one step per generation
            current_lr = max(args.lr_floor, decayed_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            log.info(f"Generation {generation} (global {global_gen}): lr={current_lr:.2e}")

            # Both adaptive instances are always kept in lockstep at the
            # same Elo target -- they're playing EACH OTHER, not one
            # imitating a teacher, so any Elo mismatch between them would
            # silently reintroduce the original one-sided-outcome problem.
            sf_adaptive.configure({"UCI_LimitStrength": True, "UCI_Elo": current_elo})
            sf_adaptive_b.configure({"UCI_LimitStrength": True, "UCI_Elo": current_elo})

            log.info(f"=== Generation {generation}: generating {batch_games} data-gen "
                     f"games (Stockfish@{current_elo} vs itself, teacher annotating) ===")
            new_examples = generate_batch(sf_teacher, sf_adaptive, sf_adaptive_b, adaptive_limit,
                                           teacher_limit, args.max_moves, batch_games,
                                           desc=f"Gen {generation} data-gen",
                                           eval_blend=args.eval_blend,
                                           eval_depth=eval_depth,
                                           eval_movetime_ms=eval_movetime_ms)
            for state, policy_target, z in new_examples:
                buffer.push(state, policy_target, z)
            log.info(f"Generation {generation}: {len(new_examples)} new examples (buffer size {len(buffer)}).")

            if args.endgame_positions_per_gen > 0:
                endgame_depth = args.endgame_teacher_depth
                if endgame_depth is None and args.endgame_teacher_movetime_ms is None:
                    endgame_depth = 12  # preserve old default when neither flag is passed
                endgame_teacher_limit = make_limit(args.endgame_teacher_movetime_ms, endgame_depth)
                endgame_examples = generate_endgame_batch(
                    sf_teacher, endgame_teacher_limit,
                    n_positions=args.endgame_positions_per_gen,
                    max_moves=args.endgame_max_moves,
                    desc=f"Gen {generation} synthetic endgames")
                for state, policy_target, z in endgame_examples:
                    buffer.push(state, policy_target, z)
                log.info(f"Generation {generation}: +{len(endgame_examples)} synthetic endgame examples (buffer size {len(buffer)}).")

            if len(buffer) < args.batch_size:
                log.info("Replay buffer smaller than one batch; skipping training this generation.")
                model.eval()
            else:
                if args.steps_per_buffer_example is not None:
                    steps_this_gen = int(max(args.train_steps_per_gen,
                                              args.steps_per_buffer_example * len(buffer)))
                    steps_this_gen = min(steps_this_gen, args.max_train_steps_per_gen)
                else:
                    steps_this_gen = args.train_steps_per_gen
                model.train()
                total_pl = total_vl = 0.0
                for step in tqdm(range(steps_this_gen), desc=f"Gen {generation} training", leave=False):
                    pl, vl = train.train_step(model, optimizer, buffer, args.batch_size, device)
                    total_pl += pl
                    total_vl += vl
                    if (step + 1) % 50 == 0:
                        log.info(f"  step {step + 1}/{steps_this_gen}  policy_loss={pl:.4f}  value_loss={vl:.4f}")
                log.info(f"Generation {generation} training done. avg policy_loss="
                         f"{total_pl / steps_this_gen:.4f} avg value_loss="
                         f"{total_vl / steps_this_gen:.4f}")

                # --- Corrective training step ---
                # Run AFTER the normal imitation training, on the separate corrective
                # buffer.  This explicitly pushes blunder-move logits DOWN (penalty)
                # and pulls Stockfish's best move UP (positive), weighted by cp_drop.
                # Runs for corrective_steps_per_gen steps if there are enough examples.
                if args.corrective_analysis and len(corrective_buffer) >= args.batch_size:
                    corr_steps = min(args.corrective_steps_per_gen, len(corrective_buffer) // args.batch_size)
                    total_cpl = total_pen = 0.0
                    for _ in tqdm(range(corr_steps), desc=f"Gen {generation} corrective", leave=False):
                        cpl, pen = corrective_train_step(
                            model, optimizer, corrective_buffer, args.batch_size, device,
                            penalty_weight=args.corrective_penalty_weight)
                        total_cpl += cpl
                        total_pen += pen
                    log.info(
                        f"Generation {generation} corrective training done ({corr_steps} steps, "
                        f"{len(corrective_buffer)} examples). "
                        f"avg policy_loss={total_cpl/corr_steps:.4f}  "
                        f"avg penalty_loss={total_pen/corr_steps:.4f}  "
)

                model.eval()

            if generation % args.save_every_gens == 0:
                _save_checkpoint(args.output, model, optimizer, current_lr, global_gen, num_promotions)
                log.info(f"Saved checkpoint to '{args.output}' "
                         f"(global_gen={global_gen}, num_promotions={num_promotions}, lr={current_lr:.2e}).")
                if args.buffer_path:
                    try:
                        with open(args.buffer_path, "wb") as f:
                            pickle.dump({"buffer": list(buffer.buffer)}, f)
                        log.info(f"Saved replay buffer to '{args.buffer_path}' "
                                 f"({len(buffer)} examples).")
                    except Exception as e:
                        log.warning(f"Could not save replay buffer: {e}.")

            log.info(f"=== Generation {generation}: eval, {args.eval_games} games "
                     f"(model MCTS vs Stockfish@{current_elo}) ===")
            pgn_sample_path = f"eval_sample_gen{generation}.pgn"
            result = run_eval_batch_with_pgn(mcts_proc, sf_adaptive, model, device, args.sims, args.threads,
                                     args.max_moves, args.eval_games, adaptive_limit,
                                     elo=current_elo, desc=f"Gen {generation} eval",
                                     sf_teacher=sf_teacher if args.corrective_analysis else None,
                                     corrective_limit=corrective_limit,
                                     corrective_cp_threshold=args.corrective_cp_threshold,
                                     corrective_eval_blend=args.corrective_eval_blend,
                                     pgn_sample_path=pgn_sample_path, pgn_sample_size=5)
            log.info(f"Saved sample games (win/loss/draw mix) to '{pgn_sample_path}'.")
            log.info(f"Eval @ Elo {current_elo}: {result['wins']}-{result['losses']}-{result['draws']} "
                     f"(score {result['score']:.1%})")

            # Corrective examples go into a SEPARATE list, never into the main
            # replay buffer.  They are trained on with corrective_train_step which
            # applies both a positive pull (toward Stockfish's move) and an explicit
            # negative penalty (away from the blunder), weighted by cp_drop magnitude.
            new_corrective = result.pop("corrective_examples", [])
            corrective_buffer.extend(new_corrective)
            # Cap corrective buffer so old examples don't dominate forever
            max_corrective = args.corrective_buffer_size
            if len(corrective_buffer) > max_corrective:
                corrective_buffer = corrective_buffer[-max_corrective:]
            if new_corrective:
                log.info(f"Generation {generation}: {len(new_corrective)} new corrective examples "
                         f"(corrective buffer size: {len(corrective_buffer)}).")

            log_trajectory(trajectory_fh, {"phase": "eval", "generation": generation, "elo": current_elo,
                                            "corrective_examples": len(new_corrective), **result})

            if result["score"] >= args.promotion_threshold:
                stalled_batches = 0
                if current_elo >= max_elo:
                    log.info(f"Cleared promotion threshold at max Elo ({max_elo}) -- curriculum complete.")
                    break

                # --- Promotion + cascading probe ---
                # Each time the model clears a level, we run a short eval
                # against (new_elo + elo_step) -- one step ahead of the just-
                # promoted level. If it clears that too, we keep probing upward
                # until it fails or hits the ceiling. Every promotion event
                # (including each skipped level) applies the LR multiplier once.
                probe_elo = current_elo + args.elo_step  # first probe target
                levels_skipped = 0

                while True:
                    # Commit the promotion to the next level
                    current_elo = clamp(probe_elo, min_elo, max_elo)
                    num_promotions += 1
                    lr_before = current_lr
                    current_lr = max(args.lr_floor, current_lr * args.promotion_lr_multiplier)
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr
                    log.info(
                        f"PROMOTED: Stockfish Elo target -> {current_elo}  "
                        f"(promotion #{num_promotions}, lr {lr_before:.2e} -> {current_lr:.2e} "
                        f"[x{args.promotion_lr_multiplier}])."
                        + (f"  [{levels_skipped} level(s) skipped so far]" if levels_skipped else "")
                    )

                    if current_elo >= max_elo:
                        log.info(f"Reached max Elo ({max_elo}) during promotion probe chain -- stopping.")
                        break

                    # Probe one level above where we just landed
                    next_probe_elo = clamp(current_elo + args.elo_step, min_elo, max_elo)
                    if next_probe_elo <= current_elo:
                        break  # already at ceiling

                    probe_score = run_promotion_probe(
                        mcts_proc, sf_probe, model, device,
                        args.sims, args.threads, args.max_moves,
                        args.probe_games,
                        sf_limit_fn=lambda elo: make_limit(args.stockfish_movetime_ms, args.stockfish_depth),
                        probe_elo=next_probe_elo,
                        threshold=probe_threshold,
                        desc=f"Probe vs {next_probe_elo}",
                    )
                    log.info(
                        f"Probe vs {next_probe_elo}: score={probe_score:.1%} "
                        f"(threshold {probe_threshold:.0%}) -> "
                        + ("SKIP LEVEL -- probing higher." if probe_score >= probe_threshold
                           else f"stay at {current_elo}.")
                    )
                    log_trajectory(trajectory_fh, {
                        "phase": "probe", "generation": generation,
                        "probe_elo": next_probe_elo, "score": probe_score,
                        "passed": probe_score >= probe_threshold,
                        "current_elo_after": current_elo,
                        "lr_after": current_lr,
                    })

                    if probe_score < probe_threshold:
                        break  # stay at current_elo; stop cascading

                    # Passed probe -- cascade upward
                    probe_elo = next_probe_elo
                    levels_skipped += 1
            else:
                stalled_batches += 1
                batch_games = min(batch_games + args.batch_games_growth, args.max_batch_games)
                log.info(f"Eval score {result['score']:.1%} did not clear {args.promotion_threshold:.0%} "
                         f"at Elo {current_elo}; staying ({stalled_batches}/{args.max_batches_per_level}). "
                         f"Next generation will use {batch_games} games.")
                if stalled_batches >= args.max_batches_per_level:
                    log.warning(f"Stalled {stalled_batches} consecutive generations at Elo {current_elo}. Stopping.")
                    break

            if args.max_generations is not None and generation >= args.max_generations:
                log.info(f"Reached --max-generations budget ({args.max_generations}); stopping.")
                break

        _save_checkpoint(args.output, model, optimizer, current_lr, global_gen, num_promotions)
        log.info(f"Done. Final Elo target: {current_elo}. Saved final weights to '{args.output}' "
                 f"(global_gen={global_gen}, num_promotions={num_promotions}, lr={current_lr:.2e}).")

    finally:
        sf_teacher.quit()
        sf_adaptive.quit()
        sf_adaptive_b.quit()
        try:
            sf_probe.quit()
        except Exception:
            pass  # sf_probe may not exist if startup failed before it was created
        train.shutdown_engine(mcts_proc)
        if trajectory_fh:
            trajectory_fh.close()
        if args.buffer_path and len(buffer) > 0:
            try:
                with open(args.buffer_path, "wb") as f:
                    pickle.dump({"buffer": list(buffer.buffer)}, f)
                log.info(f"Saved replay buffer on exit to '{args.buffer_path}' "
                         f"({len(buffer)} examples).")
            except Exception as e:
                log.warning(f"Could not save replay buffer on exit: {e}.")


if __name__ == "__main__":
    main()
