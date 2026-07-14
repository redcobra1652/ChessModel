"""
endgame_data.py -- Generate synthetic mating-technique training data using
FULL-STRENGTH STOCKFISH as both the mover and the policy/value teacher.

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
"""

import random
import chess
import numpy as np
from tqdm import tqdm

import train
from train import ACTION_SIZE, board_to_tensor, move_policy_index, position_outcome, log

# Endgame types to sample from, weighted toward rook endgames per observed
# weakness (queen mates ~1/4 of the time, rook mates rarely at all).
ENDGAME_TYPES = [
    ("KQvK", ["Q"], 3),
    ("KRvK", ["R"], 4),
    ("KRRvK", ["R", "R"], 4),
]


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


def random_endgame_position(piece_letters, stronger_side, max_attempts=200):
    """Build a random legal position: stronger_side has K + the given
    extra pieces (e.g. ["R","R"]), weaker_side has a lone K. Retries until
    a legal, non-already-over position is produced."""
    weaker_side = not stronger_side

    for _ in range(max_attempts):
        board = chess.Board.empty()
        wk_sq, bk_sq = _random_king_squares()
        strong_king_sq = wk_sq if stronger_side == chess.WHITE else bk_sq
        weak_king_sq = bk_sq if stronger_side == chess.WHITE else wk_sq

        board.set_piece_at(strong_king_sq, chess.Piece(chess.KING, stronger_side))
        board.set_piece_at(weak_king_sq, chess.Piece(chess.KING, weaker_side))

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


def play_one_endgame_with_teacher(sf_teacher, teacher_limit, max_moves, stronger_side, piece_letters, game_index):
    """Plays one synthetic endgame to conclusion using full-strength
    Stockfish for BOTH the policy teacher signal AND the actual moves
    played. No MCTS, no model. Returns (examples, ply, z_white)."""
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


def generate_endgame_batch(sf_teacher, teacher_limit, n_positions, max_moves=40,
                            desc="Synthetic endgames"):
    """Generates n_positions synthetic endgame games entirely via
    full-strength Stockfish (no model/MCTS involved), split across
    ENDGAME_TYPES per their weights, alternating stronger side for
    balance. Returns a flat list of (state, policy_target, z) tuples.

    teacher_limit: a chess.engine.Limit for sf_teacher.analyse() calls.
    Stockfish mates trivially from these positions, so a shallow/fast
    limit (e.g. depth=10 or movetime=20ms) is plenty -- keep it fast so
    this doesn't meaningfully add to per-generation wall-clock time.
    """
    weights = [w for _, _, w in ENDGAME_TYPES]
    total_w = sum(weights)
    counts = [max(1, round(n_positions * w / total_w)) for _, _, w in ENDGAME_TYPES]

    jobs = []
    for (name, pieces, _w), count in zip(ENDGAME_TYPES, counts):
        for i in range(count):
            stronger_side = chess.WHITE if (i % 2 == 0) else chess.BLACK
            jobs.append((name, pieces, stronger_side))
    random.shuffle(jobs)

    all_examples = []
    type_counts = {name: 0 for name, _, _ in ENDGAME_TYPES}
    mate_counts = {name: 0 for name, _, _ in ENDGAME_TYPES}

    for game_index, (name, pieces, stronger_side) in enumerate(tqdm(jobs, desc=desc, leave=False), 1):
        examples, plies, z_white = play_one_endgame_with_teacher(
            sf_teacher, teacher_limit, max_moves, stronger_side, pieces, game_index)
        if not examples:
            continue
        all_examples.extend(examples)
        type_counts[name] += 1
        if abs(z_white) == 1.0:
            mate_counts[name] += 1

    for name, _, _ in ENDGAME_TYPES:
        n = type_counts[name]
        mates = mate_counts[name]
        rate = (mates / n * 100) if n else 0.0
        log.info(f"  synthetic {name}: {n} games, mate/decisive rate {rate:.1f}%")

    log.info(f"Synthetic endgame batch: {len(all_examples)} examples from {len(jobs)} positions "
             f"(all via full-strength Stockfish teacher).")
    return all_examples
