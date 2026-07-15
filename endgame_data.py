"""
endgame_data.py -- Generate synthetic mating-technique + pawn-promotion
training data using FULL-STRENGTH STOCKFISH as both the mover and the
policy/value teacher.

Why not the model's own MCTS: early in training the model's search is weak
and would just be teaching itself its own current mistakes as "correct"
technique -- circular, and far slower (many sims per move) for zero benefit
when a much stronger, much faster oracle is already available (Stockfish
mates from these trivial endgame positions almost instantly even at a
shallow/fast analyse depth).

This mirrors play_one_imitation_game()'s teacher-annotation pattern: at
every ply, full-strength Stockfish is asked to (a) analyse the position for
the policy target (its top move) and (b) actually play that move to
advance the board. Both the mating side AND the lone-king side's best
defense get correct, full-strength labels. No MCTS calls, no dependency on
the current model at all -- this generator is completely decoupled from
however good or bad the model currently is.

Output format matches ReplayBuffer.push exactly: (state, policy_target, z)
tuples, using the same ACTION_SIZE/board_to_tensor/move_policy_index
encoding as the rest of the pipeline, so they push directly into the same
buffer as normal generate_batch() output.

-- K+P vs K positions --
These are generated separately from the major-piece endgames.  Two-stage
filter ensures every position is genuinely winning for the stronger side:

  Stage 1 (free, geometric): pawn rank >= 5 (rank index 4, 0-based),
  stronger king within 3 Chebyshev squares of the pawn, no obvious
  stalemate geometry.

  Stage 2 (cheap Stockfish WDL probe, depth 8): reject the position
  unless Stockfish's WDL win% for the stronger side >= KP_WIN_THRESHOLD
  (default 85%).  This catches fortress draws, opposition draws, and any
  edge cases the geometry filter misses, at minimal cost (depth-8 on a
  K+P vs K position is near-instant).

Policy target for promotion moves: always Stockfish's chosen move, which
is almost always queen promotion.  We are not trying to teach
underpromotion tricks -- we want the model to promote to a queen and win
cleanly.  Stockfish's move handles this correctly by construction.
"""

import random
import chess
import numpy as np
from tqdm import tqdm

import train
from train import ACTION_SIZE, board_to_tensor, move_policy_index, position_outcome, log

# ---------------------------------------------------------------------------
# Endgame type registry
# Each entry: (name, piece_letters_for_stronger_side, weight, generator_fn)
# Major-piece endings are weighted toward rook endings per observed model
# weakness.  K+P endings are added at lower weight -- supplementary signal
# for promotion technique specifically.
# ---------------------------------------------------------------------------

# Weight breakdown (out of 18 total):
#   KQvK   3  (~17%)  -- model already ~25% mate rate, still needs work
#   KRvK   5  (~28%)  -- near-0% mate rate, highest priority
#   KRRvK  5  (~28%)  -- near-0% mate rate, highest priority
#   KPvK   5  (~28%)  -- promotion failure; supplementary
ENDGAME_TYPES = [
    ("KQvK",  ["Q"],        3),
    ("KRvK",  ["R"],        5),
    ("KRRvK", ["R", "R"],   5),
    ("KPvK",  ["P"],        5),
]

# Stockfish WDL win% threshold for accepting a K+P vs K position.
KP_WIN_THRESHOLD = 85   # percent (0-100)

# Depth for the cheap WDL probe used to validate K+P positions.
# K+P vs K resolves trivially at depth 8; no need to go deeper.
KP_PROBE_DEPTH = 8


def _random_king_squares(min_dist=2):
    """Two random distinct squares at least min_dist (Chebyshev) apart, so
    kings don't start adjacent/illegal."""
    while True:
        a = random.randint(0, 63)
        b = random.randint(0, 63)
        if a == b:
            continue
        af, ar = chess.square_file(a), chess.square_rank(a)
        bf, br = chess.square_file(b), chess.square_rank(b)
        if max(abs(af - bf), abs(ar - br)) >= min_dist:
            return a, b


# ---------------------------------------------------------------------------
# Major-piece position generator (KQ/KR/KRR vs K) -- unchanged from before
# ---------------------------------------------------------------------------

def random_endgame_position(piece_letters, stronger_side, max_attempts=200):
    """Build a random legal position: stronger_side has K + the given
    extra pieces (e.g. ["R","R"]), weaker_side has a lone K. Retries until
    a legal, non-already-over position is produced."""
    weaker_side = not stronger_side

    for _ in range(max_attempts):
        board = chess.Board.empty()
        wk_sq, bk_sq = _random_king_squares()
        strong_king_sq = wk_sq if stronger_side == chess.WHITE else bk_sq
        weak_king_sq   = bk_sq if stronger_side == chess.WHITE else wk_sq

        board.set_piece_at(strong_king_sq, chess.Piece(chess.KING, stronger_side))
        board.set_piece_at(weak_king_sq,   chess.Piece(chess.KING, weaker_side))

        occupied = {strong_king_sq, weak_king_sq}
        ok = True
        for letter in piece_letters:
            piece_type = chess.Piece.from_symbol(letter).piece_type
            free_squares = [sq for sq in range(64) if sq not in occupied]
            if not free_squares:
                ok = False
                break
            sq = random.choice(free_squares)
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(piece_type, stronger_side))
        if not ok:
            continue

        board.turn = weaker_side if random.random() < 0.7 else stronger_side
        board.clean_castling_rights()

        if not board.is_valid():
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue
        return board

    return None


# ---------------------------------------------------------------------------
# K+P vs K position generator -- two-stage filter
# ---------------------------------------------------------------------------

def _kp_geometry_ok(pawn_sq, strong_king_sq, weak_king_sq, stronger_side):
    """Fast geometric pre-filter for K+P vs K positions.

    Accepts if:
      - Pawn is on rank >= 5 (0-based rank 4), i.e. at least 5th rank for
        White or mirrored for Black. This is where most K+P wins live; lower
        ranks are frequently drawn against the opposition rule.
      - The stronger king is within 3 Chebyshev squares of the pawn (king
        is active and supporting the push).
      - Pawn is not on the a- or h-file (rook pawns are drawn far more often
        because the defending king can reach the corner; exclude them to keep
        the win% high before the Stockfish probe).
      - No obvious stalemate: weak king is not already stuck in a corner with
        limited escape squares when it is its turn.  (Light check only --
        Stockfish probe catches anything subtle.)
    """
    pawn_file = chess.square_file(pawn_sq)
    pawn_rank = chess.square_rank(pawn_sq)  # 0 = rank 1, 7 = rank 8

    # Rook pawns excluded -- too many fortress draws
    if pawn_file in (0, 7):
        return False

    # Pawn must be advanced (rank >= 4 for White, rank <= 3 for Black mirror)
    if stronger_side == chess.WHITE:
        if pawn_rank < 4:   # must be on rank 5, 6, or 7 (not 8, that's promotion sq)
            return False
        if pawn_rank == 7:  # already on 8th rank -- shouldn't happen, skip
            return False
    else:
        # For Black the pawn ranks in reverse: rank 3 = Black's 5th rank
        if pawn_rank > 3:
            return False
        if pawn_rank == 0:
            return False

    # Strong king must be close to the pawn (supporting it)
    kf, kr = chess.square_file(strong_king_sq), chess.square_rank(strong_king_sq)
    pf, pr = chess.square_file(pawn_sq),        chess.square_rank(pawn_sq)
    if max(abs(kf - pf), abs(kr - pr)) > 3:
        return False

    return True


def _stockfish_win_pct(sf, board, stronger_side, depth=KP_PROBE_DEPTH):
    """Run a depth-limited Stockfish analysis and return the WDL win
    percentage for stronger_side (0-100).  Returns 0 on any error."""
    try:
        info = sf.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_ALL)
        wdl = info.get("wdl")
        if wdl is None:
            # Fall back to score sign if WDL not available
            score = info.get("score")
            if score is None:
                return 0
            cp = score.white().score(mate_score=10000)
            if cp is None:
                return 0
            # Very rough: positive cp for White -> likely winning for White
            if stronger_side == chess.WHITE:
                return 90 if cp > 200 else (50 if cp > 0 else 10)
            else:
                return 90 if cp < -200 else (50 if cp < 0 else 10)
        # chess.engine WDL is always from White's perspective
        w, d, l = wdl.white()
        total = w + d + l
        if total == 0:
            return 0
        win_for_stronger = w if stronger_side == chess.WHITE else l
        return int(win_for_stronger * 100 / total)
    except Exception:
        return 0


def random_kp_position(stronger_side, sf_probe, max_attempts=400):
    """Generate a random K+P vs K position that passes both the geometric
    pre-filter AND a Stockfish WDL probe confirming it is genuinely winning
    for stronger_side.

    sf_probe: a full-strength Stockfish instance (the existing sf_teacher).
    Returns a chess.Board or None if max_attempts exhausted.
    """
    weaker_side = not stronger_side

    for _ in range(max_attempts):
        # Place both kings with Chebyshev distance >= 3 (more separation than
        # the default 2 used for major-piece endings, to reduce king-proximity
        # stalemate edge cases before the probe).
        wk_sq, bk_sq = _random_king_squares(min_dist=3)
        strong_king_sq = wk_sq if stronger_side == chess.WHITE else bk_sq
        weak_king_sq   = bk_sq if stronger_side == chess.WHITE else wk_sq

        # Place pawn on a random non-rook file, advanced rank
        pawn_file = random.randint(1, 6)  # b-g files only
        if stronger_side == chess.WHITE:
            pawn_rank = random.randint(4, 6)  # ranks 5-7 (0-based 4-6)
        else:
            pawn_rank = random.randint(1, 3)  # ranks 2-4 (0-based 1-3)
        pawn_sq = chess.square(pawn_file, pawn_rank)

        # Pawn can't land on an occupied square
        if pawn_sq in (strong_king_sq, weak_king_sq):
            continue

        # Geometric pre-filter (free)
        if not _kp_geometry_ok(pawn_sq, strong_king_sq, weak_king_sq, stronger_side):
            continue

        # Build the board
        board = chess.Board.empty()
        board.set_piece_at(strong_king_sq, chess.Piece(chess.KING,  stronger_side))
        board.set_piece_at(weak_king_sq,   chess.Piece(chess.KING,  weaker_side))
        board.set_piece_at(pawn_sq,        chess.Piece(chess.PAWN,  stronger_side))
        board.clean_castling_rights()

        # Start with the stronger side to move ~60% of the time (they're
        # pushing), weaker side ~40% (defending).  Both are instructive.
        board.turn = stronger_side if random.random() < 0.6 else weaker_side

        if not board.is_valid():
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue

        # Stage 2: Stockfish WDL probe
        win_pct = _stockfish_win_pct(sf_probe, board, stronger_side)
        if win_pct < KP_WIN_THRESHOLD:
            continue

        return board

    return None


# ---------------------------------------------------------------------------
# Game runner (shared by major-piece and K+P types)
# ---------------------------------------------------------------------------

def play_one_endgame_with_teacher(sf_teacher, teacher_limit, max_moves,
                                   stronger_side, piece_letters, game_index,
                                   sf_probe=None):
    """Plays one synthetic endgame to conclusion using full-strength
    Stockfish for BOTH the policy teacher signal AND the actual moves
    played. No MCTS, no model. Returns (examples, ply, z_white).

    For K+P endings, sf_probe must be supplied (used to validate the
    position).  For major-piece endings it is unused.
    """
    is_kp = (piece_letters == ["P"])

    if is_kp:
        if sf_probe is None:
            raise ValueError("sf_probe required for K+P position generation")
        board = random_kp_position(stronger_side, sf_probe)
    else:
        board = random_endgame_position(piece_letters, stronger_side)

    if board is None:
        return [], 0, 0.0

    examples = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        side_to_move = board.turn

        info = sf_teacher.analyse(board, teacher_limit)
        teacher_move = info["pv"][0]

        mirror = side_to_move == chess.BLACK
        policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy_target[move_policy_index(teacher_move, mirror)] = 1.0
        state = board_to_tensor(board)
        examples.append([state, policy_target, side_to_move])

        assert teacher_move in board.legal_moves, \
            f"ILLEGAL teacher move '{teacher_move}' at FEN '{board.fen()}'"
        board.push(teacher_move)
        ply += 1

    terminal, result_for_mover = position_outcome(board)
    if not terminal:
        z_white = 0.0
    else:
        z_white = result_for_mover if board.turn == chess.WHITE else -result_for_mover

    finished = []
    for state, policy_target, stm in examples:
        z = z_white if stm == chess.WHITE else -z_white
        finished.append((state, policy_target, z))
    return finished, ply, z_white


# ---------------------------------------------------------------------------
# Main batch generator (called from stockfish_train.py)
# ---------------------------------------------------------------------------

def generate_endgame_batch(sf_teacher, teacher_limit, n_positions, max_moves=40,
                            desc="Synthetic endgames"):
    """Generates n_positions synthetic endgame games entirely via
    full-strength Stockfish (no model/MCTS involved), split across
    ENDGAME_TYPES per their weights, alternating stronger side for
    balance. Returns a flat list of (state, policy_target, z) tuples.

    teacher_limit: a chess.engine.Limit for sf_teacher.analyse() calls.
    Stockfish resolves these positions trivially, so a shallow/fast limit
    (e.g. depth=10 or movetime=20ms) is fine for normal move selection.
    The K+P WDL probe always uses its own fixed depth (KP_PROBE_DEPTH=8)
    regardless of teacher_limit.

    sf_teacher is reused as sf_probe for K+P position validation -- it is
    the same full-strength instance, just called at a different depth for
    the cheap pre-game WDL check.
    """
    weights   = [w for _, _, w in ENDGAME_TYPES]
    total_w   = sum(weights)
    counts    = [max(1, round(n_positions * w / total_w)) for _, _, w in ENDGAME_TYPES]

    jobs = []
    for (name, pieces, _w), count in zip(ENDGAME_TYPES, counts):
        for i in range(count):
            stronger_side = chess.WHITE if (i % 2 == 0) else chess.BLACK
            jobs.append((name, pieces, stronger_side))
    random.shuffle(jobs)

    all_examples = []
    type_counts  = {name: 0 for name, _, _ in ENDGAME_TYPES}
    mate_counts  = {name: 0 for name, _, _ in ENDGAME_TYPES}
    kp_rejected  = 0   # positions filtered out by the two-stage K+P filter

    for game_index, (name, pieces, stronger_side) in enumerate(tqdm(jobs, desc=desc, leave=False), 1):
        is_kp = (pieces == ["P"])
        examples, plies, z_white = play_one_endgame_with_teacher(
            sf_teacher, teacher_limit, max_moves, stronger_side, pieces, game_index,
            sf_probe=sf_teacher if is_kp else None)

        if not examples:
            if is_kp:
                kp_rejected += 1
            continue

        all_examples.extend(examples)
        type_counts[name] += 1
        if abs(z_white) == 1.0:
            mate_counts[name] += 1

    for name, _, _ in ENDGAME_TYPES:
        n     = type_counts[name]
        mates = mate_counts[name]
        rate  = (mates / n * 100) if n else 0.0
        log.info(f"  synthetic {name}: {n} games, mate/decisive rate {rate:.1f}%")

    if kp_rejected:
        log.info(f"  KPvK: {kp_rejected} positions rejected by two-stage win filter "
                 f"(geometry + Stockfish WDL < {KP_WIN_THRESHOLD}%).")

    log.info(f"Synthetic endgame batch: {len(all_examples)} examples from {len(jobs)} positions "
             f"(all via full-strength Stockfish teacher).")
    return all_examples
