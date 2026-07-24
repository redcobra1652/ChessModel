#!/usr/bin/env python3
"""
train.py -- Minimal AlphaZero-style chess trainer.

Compiles and drives mcts.cpp (a persistent, multi-threaded MCTS engine)
over a line-based JSON stdin/stdout protocol, uses a PyTorch dual-head
(policy + value) ResNet for position evaluation, runs self-play games to
fill a replay buffer, trains the network, and periodically pits the
newly trained ("candidate") network against the previous best network
in a small tournament -- only promoting the candidate to best_model.pt
if it wins more games than it loses. If it doesn't, the in-memory model
is reverted to the previous best before continuing.

Usage:
    python3 train.py --games 100

See --help for all tunable knobs. Designed to run entirely on CPU on
macOS out of the box; pass --device mps to use Apple Silicon GPU accel.
"""

import argparse
import collections
import json
import logging
import math
import os
import random
import select
import subprocess
import sys
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

ENGINE_SRC = "mcts.cpp"
ENGINE_BIN = "./mcts_engine"
BEST_MODEL_PATH = "best_model.pt"
ACTION_SIZE = 64 * 64 + 48  # from_square * 64 + to_square for non-promotions (4096 indices),
                             # plus 48 dedicated promotion indices (see move_policy_index).
                             # Promotions: 8 files x 2 colors x 3 non-queen pieces = 48.
                             # Queen promotion uses the base from*64+to slot (index < 4096)
                             # so the model's default "just push the pawn" action is queen.

# Set by main() before every real move: which network answers this
# engine's NN-evaluation requests for the *entire* search tree of that
# move. (Each side always searches with only its own network, exactly
# as in real AlphaZero -- there is never a "mixed" search.)
CURRENT_MODEL = None
CURRENT_DEVICE = None

# Toggled True during self-play data generation (adds Dirichlet noise
# to root priors for exploration) and False during evaluation /
# tournament play (deterministic, strongest-move search).
SELF_PLAY_MODE = True
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25

# Resignation: if a side's own root value stays below RESIGN_THRESHOLD for
# RESIGN_CONSECUTIVE of its own moves in a row, it resigns instead of
# playing to checkmate/max_moves. This is a real AlphaZero/Lc0 technique --
# it lets you fit more *distinct* games into the same compute budget
# instead of grinding out already-decided endgames. RESIGN_DISABLE_FRACTION
# of self-play games ignore resignation entirely and play to a natural
# conclusion; this is also standard practice (AlphaZero used it too) --
# without it, the value head is never shown what actually happens in
# "resign-zone" positions and can silently drift out of calibration there,
# which you'd have no way of detecting since those positions would simply
# stop appearing in training data.
RESIGN_THRESHOLD = -0.90
RESIGN_CONSECUTIVE = 3
RESIGN_DISABLE_FRACTION = 0.10


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


log = logging.getLogger("train")


# ----------------------------------------------------------------------
# Engine build / lifecycle
# ----------------------------------------------------------------------

def compile_engine():
    if os.path.exists(ENGINE_BIN) and os.path.getmtime(ENGINE_BIN) > os.path.getmtime(ENGINE_SRC):
        log.info("mcts_engine binary is up to date, skipping recompilation.")
        return
    log.info(f"Compiling {ENGINE_SRC} with clang++ -O3 ...")
    cmd = ["clang++", "-O3", "-std=c++17", "-pthread", "-o", "mcts_engine", ENGINE_SRC]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Compilation failed:\n" + result.stderr)
        sys.exit(1)
    log.info("Compilation succeeded -> ./mcts_engine")


def start_engine():
    proc = subprocess.Popen(
        [ENGINE_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,   # let engine's own error/warning logs pass through to console
        text=True,
        bufsize=1,           # line-buffered
    )
    log.info(f"Started mcts_engine subprocess (pid={proc.pid}).")
    return proc


def shutdown_engine(proc):
    try:
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
        log.info("mcts_engine exited cleanly.")
    except Exception as e:
        log.warning(f"Engine did not exit cleanly ({e}); killing.")
        proc.kill()


def restart_engine(proc):
    """Forcibly kill a (possibly hung/crashed) engine subprocess and start
    a fresh one. Used for recovery so a single bad engine call can't take
    down an unattended overnight run."""
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    return start_engine()


# ----------------------------------------------------------------------
# Board <-> tensor encoding
#
# 13 channels x 8 x 8:
#   0-5   white P,N,B,R,Q,K
#   6-11  black P,N,B,R,Q,K
#   12    side-to-move plane (all 1.0 if white to move, else all 0.0)
# ----------------------------------------------------------------------

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encodes the board from the perspective of the side TO MOVE: "my"
    pieces always occupy channels 0-5 and the opponent's occupy 6-11,
    with the whole board mirrored vertically (rank flip only -- chess only
    has vertical mirror symmetry between colors) whenever Black is to
    move, so the mover always appears to be moving "up" the board.

    Previously this encoded raw White/Black regardless of whose turn it
    was, forcing the network to learn White-side and Black-side patterns
    as two mostly-separate cases instead of one shared, reusable
    representation. This is standard AlphaZero canonical-orientation
    encoding, and the move-index lookup in nn_eval()/play_one_game() must
    stay in sync with the same mirroring (see move_policy_index below) --
    only the neural-net input/output boundary is mirrored; every other
    part of the system (search, legality, replay of the actual game)
    still deals exclusively in real, un-mirrored board coordinates.
    """
    mirror = board.turn == chess.BLACK
    t = np.zeros((13, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        sq = chess.square_mirror(square) if mirror else square
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        is_movers_piece = piece.color == board.turn
        channel = PIECE_ORDER.index(piece.piece_type) + (0 if is_movers_piece else 6)
        t[channel, rank, file] = 1.0
    # Channel 12: repetition-count signal -- 0.0 the first time this exact
    # position has occurred, 0.5 the second time, 1.0 the third-or-more
    # time (i.e. an actual/imminent threefold-repetition draw). This
    # replaces a previous constant-1.0 filler that carried no information
    # at all. It lets the value head finally distinguish "this position,
    # but it's a dead draw by rule" from "first time here, still worth
    # fighting for" -- something it structurally could not do before.
    #
    # This is only meaningful if `board` carries real move history (its
    # move stack must actually have been pushed through from game start,
    # or from game-start-prefix + search path -- see handle_root /
    # handle_visit). A board reconstructed from a bare FEN has an empty
    # move stack and board.is_repetition() will always report False on
    # it, silently hiding real repetitions.
    if board.is_repetition(3):
        rep_signal = 1.0
    elif board.is_repetition(2):
        rep_signal = 0.5
    else:
        rep_signal = 0.0
    t[12, :, :] = rep_signal
    return t


def move_policy_index(move: chess.Move, mirror: bool = False) -> int:
    """Maps a move to a unique index in [0, ACTION_SIZE).

    Non-promotion moves (the vast majority): index = from_sq * 64 + to_sq,
    occupying [0, 4096) exactly as before. This is unchanged so all existing
    replay-buffer examples remain valid.

    Promotion moves: queen promotions also use the base from*64+to slot so
    that the model's natural "push pawn to back rank" action defaults to the
    correct piece. Underpromotions (rook / bishop / knight) get a dedicated
    index in [4096, 4144):

        4096 + file * 6 + color_offset * 3 + piece_offset

    where:
        file         = 0-7  (a-h file of the destination square)
        color_offset = 0 if promoting as White (to rank 8, mirror=False)
                       1 if promoting as Black (to rank 1, mirror=True)
        piece_offset = 0 for rook, 1 for bishop, 2 for knight

    This gives 8 * 2 * 3 = 48 unique underpromotion slots, for a total of
    4096 + 48 = 4144 = ACTION_SIZE.

    `mirror` must match board_to_tensor() for this position (True when Black
    is to move), ensuring the square numbering is consistent throughout.
    """
    frm = chess.square_mirror(move.from_square) if mirror else move.from_square
    to  = chess.square_mirror(move.to_square)   if mirror else move.to_square

    if move.promotion is None or move.promotion == chess.QUEEN:
        # Non-promotion and queen promotion share the base slot.
        return frm * 64 + to

    # Underpromotion: encode into the dedicated [4096, 4144) block.
    file = chess.square_file(to)
    color_offset = 1 if mirror else 0  # mirror=True means Black is promoting
    piece_offset = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[move.promotion]
    return 4096 + file * 6 + color_offset * 3 + piece_offset


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class DualHeadResNet(nn.Module):
    """Dual-head (policy + value) residual network.

    Input:  (N, 13, 8, 8) stacked board-state planes.
    Output: policy logits (N, ACTION_SIZE); value (N, 1) in [-1, 1] via tanh.
    """

    def __init__(self, in_channels: int = 13, channels: int = 128, n_blocks: int = 10,
                 action_size: int = ACTION_SIZE):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])

        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = torch.flatten(p, 1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = torch.flatten(v, 1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


@torch.no_grad()
def nn_eval(board: chess.Board, is_root: bool = False):
    """Runs CURRENT_MODEL on `board`, returns (legal_move_ucis, priors, value)
    with priors renormalized (softmax) over legal moves only.

    `is_root` must be True only for the actual MCTS root of the current
    search call. Dirichlet exploration noise is a *root-only* AlphaZero
    technique: it diversifies which lines get explored from the top of
    the tree for a single real move, while every other (non-root) node
    is still evaluated with the network's raw priors so PUCT selection
    deeper in the tree stays trustworthy. Applying noise at every node
    (the previous bug here) corrupts priors throughout the whole tree on
    every simulation, degrading search quality broadly -- a very
    plausible cause of the erratic/passive self-play you were seeing."""
    model = CURRENT_MODEL
    device = CURRENT_DEVICE
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    logits, value_t = model(x)
    logits = logits.squeeze(0).cpu().numpy()
    value = float(value_t.squeeze().cpu().numpy())

    mirror = board.turn == chess.BLACK
    legal = list(board.legal_moves)
    idxs = np.array([move_policy_index(m, mirror) for m in legal], dtype=np.int64)
    sel = logits[idxs]
    sel = sel - sel.max()
    exps = np.exp(sel)
    priors = exps / (exps.sum() + 1e-8)

    if SELF_PLAY_MODE and is_root and len(legal) > 0:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal))
        priors = (1 - DIRICHLET_EPS) * priors + DIRICHLET_EPS * noise

    return [m.uci() for m in legal], priors.tolist(), value


def find_immediate_mate(board: chess.Board):
    """Scans legal moves for an immediate mate-in-1.

    This is a cheap, deterministic safety net that sits in front of the
    search/network entirely: regardless of what MCTS or the value head
    currently believe about the position, if a mate is sitting right
    there, take it -- full stop. Cost is one legality scan plus a
    push/is_checkmate/pop per legal move (a handful of microseconds),
    negligible next to a real search call, and it eliminates the most
    embarrassing failure mode -- missing a mate in 1 -- outright instead
    of hoping search/sims happen to find and trust it.

    Returns the first mating chess.Move found, or None if there isn't
    one.
    """
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
    return None


def position_outcome(board: chess.Board):
    """Returns (terminal, result) where result is from the perspective of
    the side to move at `board` (+1 win / -1 loss / 0 draw), following
    the standard AlphaZero value-head sign convention.

    Uses claim_draw=False for the base outcome check, then explicitly
    checks is_repetition(3) and can_claim_fifty_moves() to detect draws.
    This avoids the python-chess can_claim_threefold_repetition() lookahead
    bug where the game is declared a draw one move early because *some*
    legal move would cause a third repetition -- even when the current
    position has only repeated twice. We only declare a draw when the
    position has actually repeated three times already.
    """
    # Check non-draw terminals first (checkmate, stalemate, insufficient
    # material, 75-move rule, fivefold repetition) without claim_draw so
    # we don't trigger the lookahead behaviour.
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        if outcome.winner is None:
            return True, 0.0
        return True, (1.0 if outcome.winner == board.turn else -1.0)
    # Now check claimable draws strictly: only if the position has ACTUALLY
    # repeated three times already, or the fifty-move rule is already met.
    if board.is_repetition(3) or board.can_claim_fifty_moves():
        return True, 0.0
    return False, 0.0


# ----------------------------------------------------------------------
# Engine request handlers (Python side of the protocol)
# ----------------------------------------------------------------------

def _replay(history: list, path: list = (), start_fen: str = None) -> chess.Board:
    """Reconstructs a board by replaying real moves from the game start.

    This is the crux of correct repetition handling: a board built from a
    bare FEN (the old approach) has an empty move stack, so
    board.is_repetition()/board.outcome(claim_draw=True) can NEVER detect
    threefold repetition on it -- there is nothing to compare against.
    Replaying the actual move sequence gives the board real history, so
    both the terminal check below and the repetition-count feature in
    board_to_tensor work correctly, including for hypothetical lines deep
    inside the search tree (via `path`), not just the real game.
    """
    board = chess.Board(start_fen) if start_fen else chess.Board()
    for uci in history:
        board.push_uci(uci)
    for uci in path:
        board.push_uci(uci)
    return board


_SAFE_TERMINAL = {"terminal": True, "result": 0.0, "moves": [], "priors": [], "value": 0.0}


def handle_root(history: list, start_fen: str = None) -> dict:
    try:
        board = _replay(history, start_fen=start_fen)
        terminal, result = position_outcome(board)
        if terminal:
            return {"terminal": True, "result": result, "moves": [], "priors": [], "value": result}
        moves, priors, value = nn_eval(board, is_root=True)
        return {"terminal": False, "result": 0.0, "moves": moves, "priors": priors, "value": value}
    except Exception as e:
        log.warning(f"handle_root error (returning safe terminal): {e}")
        return _SAFE_TERMINAL


def handle_batch(requests: list, start_fen: str = None) -> list:
    """Handle a batch of root/visit requests in one call.

    Each request is a dict with the same shape as the individual
    root/visit protocol messages.  Responses are returned as a list of
    JSON strings (so the C++ side can deserialise them with its existing
    per-object parser without needing a new vector-of-object JSON type).

    The key win: all non-terminal positions in the batch are evaluated
    in a single batched GPU forward pass instead of N separate ones.
    Terminal positions are handled cheaply in pure Python with no GPU
    involvement, so they don't dilute the batch.
    """
    model = CURRENT_MODEL
    device = CURRENT_DEVICE

    # --- Phase 1: replay boards and detect terminals ---
    boards = []
    results = [None] * len(requests)  # pre-fill; terminals filled here

    for i, req in enumerate(requests):
        try:
            if req.get("type") == "root":
                board = _replay(req.get("history", []), start_fen=start_fen)
                terminal, result = position_outcome(board)
                if terminal:
                    results[i] = {"terminal": True, "result": result,
                                  "moves": [], "priors": [], "value": result}
                else:
                    boards.append((i, board, True))  # (index, board, is_root)
            else:  # visit
                board = _replay(req.get("history", []), req.get("path", []), start_fen=start_fen)
                move = chess.Move.from_uci(req["move"])
                if move not in board.legal_moves:
                    log.warning(f"handle_batch: illegal move {req['move']} — returning safe terminal")
                    results[i] = {"fen": board.fen(), **_SAFE_TERMINAL}
                    continue
                board.push(move)
                child_fen = board.fen()
                terminal, result = position_outcome(board)
                if terminal:
                    results[i] = {"fen": child_fen, "terminal": True, "result": result,
                                  "moves": [], "priors": [], "value": result}
                else:
                    boards.append((i, board, False, child_fen))
        except Exception as e:
            log.warning(f"handle_batch: error on request {i} ({e}); returning safe terminal")
            results[i] = _SAFE_TERMINAL

    # --- Phase 2: batch GPU eval for all non-terminal boards ---
    if boards and model is not None:
        try:
            # Stack all board tensors into one batch.
            tensors = []
            for entry in boards:
                board = entry[1]
                tensors.append(torch.from_numpy(board_to_tensor(board)))
            batch_x = torch.stack(tensors).to(device)  # (N, 13, 8, 8)

            with torch.no_grad():
                batch_logits, batch_values = model(batch_x)
            batch_logits = batch_logits.cpu().numpy()
            batch_values = batch_values.squeeze(-1).cpu().numpy()

            for j, entry in enumerate(boards):
                idx = entry[0]
                board = entry[1]
                is_root = entry[2]
                logits = batch_logits[j]
                value = float(batch_values[j])

                mirror = board.turn == chess.BLACK
                legal = list(board.legal_moves)
                idxs = np.array([move_policy_index(m, mirror) for m in legal], dtype=np.int64)
                sel = logits[idxs]
                sel = sel - sel.max()
                exps = np.exp(sel)
                priors = (exps / (exps.sum() + 1e-8)).tolist()

                if SELF_PLAY_MODE and is_root and len(legal) > 0:
                    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal))
                    priors = ((1 - DIRICHLET_EPS) * np.array(priors)
                              + DIRICHLET_EPS * noise).tolist()

                move_ucis = [m.uci() for m in legal]

                if len(entry) == 4:  # visit (has child_fen)
                    child_fen = entry[3]
                    results[idx] = {"fen": child_fen, "terminal": False, "result": 0.0,
                                    "moves": move_ucis, "priors": priors, "value": value}
                else:  # root
                    results[idx] = {"terminal": False, "result": 0.0,
                                    "moves": move_ucis, "priors": priors, "value": value}
        except Exception as e:
            log.warning(f"handle_batch: GPU eval failed ({e}); falling back to safe terminals")
            for entry in boards:
                idx = entry[0]
                if results[idx] is None:
                    results[idx] = _SAFE_TERMINAL

    # Fill any remaining None slots (shouldn't happen but defensive).
    for i in range(len(results)):
        if results[i] is None:
            results[i] = _SAFE_TERMINAL

    return [json.dumps(r) for r in results]


def handle_visit(history: list, path: list, move_uci: str, start_fen: str = None) -> dict:
    try:
        board = _replay(history, path, start_fen=start_fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            log.warning(f"Illegal move from engine: {move_uci} at {board.fen()} — returning safe terminal")
            return {"fen": board.fen(), **_SAFE_TERMINAL}
        board.push(move)
        child_fen = board.fen()
        terminal, result = position_outcome(board)
        if terminal:
            return {"fen": child_fen, "terminal": True, "result": result,
                    "moves": [], "priors": [], "value": result}
        moves, priors, value = nn_eval(board, is_root=False)
        return {"fen": child_fen, "terminal": False, "result": 0.0,
                "moves": moves, "priors": priors, "value": value}
    except Exception as e:
        log.warning(f"handle_visit error (returning safe terminal): {e}")
        return {"fen": "", **_SAFE_TERMINAL}


ENGINE_READ_TIMEOUT_SECS = 120  # generous margin above any real per-request cost


def _readline_with_timeout(proc, timeout=ENGINE_READ_TIMEOUT_SECS):
    """Like proc.stdout.readline(), but raises instead of blocking forever
    if the engine subprocess stops responding -- protects long, unattended
    runs from silently hanging overnight if the engine ever deadlocks,
    crashes without closing its pipe, or gets stuck in a runaway loop."""
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise RuntimeError(
            f"mcts_engine did not respond within {timeout}s -- it is likely hung "
            f"or crashed. Aborting rather than blocking forever. "
            f"(pid={proc.pid}, check for a stale/zombie process to kill.)"
        )
    return proc.stdout.readline()


def search(proc, board: chess.Board, sims: int, threads: int):
    """Drives one full MCTS search call to the engine, servicing every
    root/visit request it makes along the way, until it returns the
    final visit-count result for the given root position.

    Takes the real, live game `board` (not just its FEN) so the engine
    can be given the actual move history played so far. That history is
    threaded through every root/visit request the engine makes during
    this search, so terminal detection and the repetition-count encoding
    can see real, path-dependent repetition -- something a bare FEN can
    never represent.

    Strictly synchronous request/response alternation -- this is what
    makes the pipe deadlock-free even with the engine's internal worker
    threads: exactly one side is ever blocked waiting on the other.
    """
    history = [m.uci() for m in board.move_stack]

    # Detect a non-standard starting position so _replay() inside
    # handle_root/handle_visit can reconstruct the board correctly.
    # Walk back to the root of the move stack to find the true base FEN.
    if board.move_stack:
        temp = board.copy(stack=True)
        while temp.move_stack:
            temp.pop()
        start_fen = None if temp.fen() == chess.STARTING_FEN else temp.fen()
    else:
        start_fen = None if board.fen() == chess.STARTING_FEN else board.fen()

    proc.stdin.write(json.dumps({
        "cmd": "search", "fen": board.fen(), "sims": sims, "threads": threads,
        "history": history,
    }) + "\n")
    proc.stdin.flush()

    while True:
        line = _readline_with_timeout(proc)
        if not line:
            stderr_tail = ""
            raise RuntimeError(f"mcts_engine pipe closed unexpectedly during search. {stderr_tail}")
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning(f"Malformed JSON from engine (skipping): {e} -- line={line[:120]!r}")
            continue

        if obj.get("cmd") == "result":
            visits = dict(zip(obj["moves"], obj["visits"]))
            return visits, obj["value"]

        if obj.get("cmd") == "batch":
            # Batched eval: one GPU forward pass for all requests.
            responses = handle_batch(obj.get("requests", []), start_fen=start_fen)
            # Encode responses as array-of-strings so C++ can parse each
            # element with its existing per-object JSON parser.
            encoded = json.dumps({"cmd": "batch_result", "responses": responses})
            proc.stdin.write(encoded + "\n")
            proc.stdin.flush()
            continue

        if obj.get("type") == "root":
            resp = handle_root(obj.get("history", []), start_fen=start_fen)
        elif obj.get("type") == "visit":
            resp = handle_visit(obj.get("history", []), obj.get("path", []), obj["move"], start_fen=start_fen)
        else:
            # Engine echoed something unexpected (e.g. a bounced response).
            # Send a safe terminal so the engine is never left waiting.
            log.warning(f"Unexpected engine message (sending safe terminal): {line[:120]!r}")
            resp = _SAFE_TERMINAL

        proc.stdin.write(json.dumps(resp) + "\n")
        proc.stdin.flush()


def pick_move_from_visits(visits: dict, temperature: float) -> str:
    moves = list(visits.keys())
    counts = np.array([visits[m] for m in moves], dtype=np.float64)
    if counts.sum() <= 0:
        return random.choice(moves)
    if temperature <= 1e-3:
        return moves[int(np.argmax(counts))]
    weighted = counts ** (1.0 / temperature)
    probs = weighted / weighted.sum()
    return np.random.choice(moves, p=probs)


def find_safe_moves(board: chess.Board):
    """Returns the subset of board.legal_moves that do NOT hand the
    opponent an immediate mate-in-1 reply.

    This is the defensive mirror of find_immediate_mate(), which only
    ever checks whether the side to move CAN mate right now -- it never
    checks whether a candidate move lets the opponent mate back next
    move. That gap is real: with a shallow search (few sims) or a value
    head that hasn't fully learned to weight a one-ply mate threat, MCTS
    can end up preferring a move that walks straight into a mate the
    opponent already had lined up, while some other, boring, perfectly
    legal move would have defused it. (Observed in practice: 22.b4 Qh2#,
    where 22.b4 ignored an already-available ...Qh2# and plenty of other
    legal 22nd moves for White would have prevented it.)

    Cheap: one push/find_immediate_mate/pop per legal move, same cost
    class as find_immediate_mate itself.

    Returns [] if every legal move allows an immediate mate reply (i.e.
    mate is unavoidable next move regardless of what's played now) --
    callers should fall back to normal move selection in that case, since
    there is nothing safer to prefer.
    """
    safe = []
    for move in board.legal_moves:
        board.push(move)
        opponent_can_mate = find_immediate_mate(board) is not None
        board.pop()
        if not opponent_can_mate:
            safe.append(move)
    return safe


def pick_safe_move_from_visits(board: chess.Board, visits: dict, temperature: float) -> str:
    """Like pick_move_from_visits, but for deterministic (temperature ~0)
    play only: screens out any candidate move that would hand the
    opponent an immediate mate-in-1, using find_safe_moves() as a
    last-line safety net on top of whatever MCTS itself concluded.

    Only engages for temperature <= 1e-3 (real, deterministic play) --
    self-play's temperature>0 exploratory sampling is left untouched, so
    this doesn't change what the value/policy heads get trained on.

    Falls back to the plain visit-count choice whenever there's nothing
    better to do: if the search's own top pick is already safe, if none
    of the moves MCTS actually visited are safe (falls back to any safe
    legal move it didn't happen to visit, if one exists), or if literally
    every legal move allows a mate reply (a real unavoidable mate -- no
    override can fix that).
    """
    best_uci = pick_move_from_visits(visits, temperature)
    if temperature > 1e-3:
        return best_uci

    best_move = chess.Move.from_uci(best_uci)
    board.push(best_move)
    in_danger = find_immediate_mate(board) is not None
    board.pop()
    if not in_danger:
        return best_uci

    safe_moves = find_safe_moves(board)
    if not safe_moves:
        return best_uci  # mate is unavoidable regardless -- nothing to do

    safe_ucis = {m.uci() for m in safe_moves}
    safe_visited = {uci: n for uci, n in visits.items() if uci in safe_ucis}
    if safe_visited:
        # Prefer the safe move search itself gave the most weight to.
        return max(safe_visited, key=safe_visited.get)
    # None of the moves MCTS actually visited happen to be safe (can
    # happen with very few sims/legal moves) -- better an unsearched
    # safe move than a searched one that loses outright.
    return safe_moves[0].uci()


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------

class ReplayBuffer:
    """Simple FIFO replay buffer, capped at `capacity` total examples.

    Note: an earlier version of this buffer split examples into decisive
    (z != 0) and drawn (z == 0) pools and capped how much of each training
    batch could be draws, as a guard against a self-play repetition-spiral
    failure mode. That cap was removed because this buffer is populated by
    Stockfish-vs-Stockfish self-play at a fixed Elo, not the model's own
    self-play -- two real engines occasionally drawing via genuine
    equal-position repetition or the 50-move rule is expected, rational
    behavior, not the runaway degenerate pattern the cap was designed to
    guard against. Sampling is now uniform across all stored examples.
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.buffer))
        if n == 0:
            raise ValueError("ReplayBuffer.sample called on an empty buffer")

        batch = random.sample(self.buffer, n)
        states, policies, values = zip(*batch)
        states_t = torch.from_numpy(np.stack(states)).float()
        policies_t = torch.from_numpy(np.stack(policies)).float()
        values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return states_t, policies_t, values_t


# ----------------------------------------------------------------------
# Self-play
# ----------------------------------------------------------------------

def play_one_game(proc, sims: int, threads: int, max_moves: int, temp_moves: int, game_index: int):
    """Plays one self-play game with the CURRENT_MODEL for both sides.
    Returns a list of (state, policy_target[4096], side_to_move_color)
    training examples with the value target filled in afterwards."""
    global SELF_PLAY_MODE
    SELF_PLAY_MODE = True

    # See RESIGN_* constants: a fraction of games always play to a natural
    # conclusion (no resignation) specifically so the value head's
    # calibration in clearly-losing positions keeps getting checked
    # against real outcomes instead of drifting unseen.
    resign_disabled = random.random() < RESIGN_DISABLE_FRACTION
    bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
    resigned_loser = None

    board = chess.Board()
    examples = []
    ply = 0

    while not board.is_game_over(claim_draw=False) and not board.is_repetition(3) and not board.can_claim_fifty_moves() and ply < max_moves:
        # --- Mate-in-1 safety net: checked before trusting MCTS at all ---
        # Regardless of what the network/search currently think, if a
        # legal move mates right now, play it -- full stop. This also
        # gives the value/policy heads a clean, unambiguous training
        # signal for the position instead of skipping it.
        mate_move = find_immediate_mate(board)
        if mate_move is not None:
            mirror = board.turn == chess.BLACK
            policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy_target[move_policy_index(mate_move, mirror)] = 1.0
            state = board_to_tensor(board)
            examples.append([state, policy_target, board.turn])
            bad_streak[board.turn] = 0  # forced mate is the opposite of "bad"
            board.push(mate_move)
            ply += 1
            continue

        visits, root_value = search(proc, board, sims=sims, threads=threads)

        mirror = board.turn == chess.BLACK
        policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
        total_visits = sum(visits.values()) or 1
        for uci, n in visits.items():
            idx = move_policy_index(chess.Move.from_uci(uci), mirror)
            policy_target[idx] += n / total_visits

        state = board_to_tensor(board)
        examples.append([state, policy_target, board.turn])

        # root_value is from the current mover's own perspective (see
        # mcts.cpp: root->W/root->N under the same alternating-sign
        # convention used everywhere else). Track consecutive bad
        # evaluations per side so a single noisy/blundered eval doesn't
        # trigger a premature resignation.
        mover = board.turn
        if root_value < RESIGN_THRESHOLD:
            bad_streak[mover] += 1
        else:
            bad_streak[mover] = 0
        if not resign_disabled and bad_streak[mover] >= RESIGN_CONSECUTIVE:
            resigned_loser = mover
            break

        temperature = 1.0 if ply < temp_moves else 0.1
        chosen_uci = pick_move_from_visits(visits, temperature)
        move = chess.Move.from_uci(chosen_uci)

        # --- Safety check: never push an illegal move ---
        assert move in board.legal_moves, (
            f"ILLEGAL MOVE selected during self-play game {game_index}: "
            f"{chosen_uci} at FEN '{board.fen()}'"
        )
        board.push(move)
        ply += 1

    if resigned_loser is not None:
        z_white = -1.0 if resigned_loser == chess.WHITE else 1.0
        outcome_desc = f"resignation ({'White' if resigned_loser == chess.WHITE else 'Black'} resigned)"
    else:
        terminal, result_for_mover = position_outcome(board)
        if not terminal:
            # Hit max_moves without a natural game end -> treat as a draw.
            z_white = 0.0
        else:
            # result_for_mover is from the perspective of the side to move
            # in the *final* (terminal) position; convert to an absolute
            # white-perspective scalar so we can sign it per training sample.
            z_white = result_for_mover if board.turn == chess.WHITE else -result_for_mover
        outcome_desc = str(board.outcome(claim_draw=True))

    finished = []
    for state, policy_target, side_to_move in examples:
        z = z_white if side_to_move == chess.WHITE else -z_white
        finished.append((state, policy_target, z))

    log.info(f"Game {game_index}: {ply} plies, result(white)={z_white:+.1f}, "
             f"outcome={outcome_desc}{' [resign disabled]' if resign_disabled else ''}")
    return finished


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_step(model, optimizer, buffer: ReplayBuffer, batch_size: int, device):
    states, policies, values = buffer.sample(batch_size)
    states, policies, values = states.to(device), policies.to(device), values.to(device)

    logits, pred_values = model(states)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(policies * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_values, values)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


# ----------------------------------------------------------------------
# Tournament evaluation: candidate vs. previous best
# ----------------------------------------------------------------------

def tournament(proc, model_a, model_b, n_games: int, sims: int, threads: int,
                max_moves: int, device):
    """Plays n_games between model_a (candidate) and model_b (previous best),
    alternating colors. Each side searches only with its own network for
    the entire move (exactly like two independent chess engines).
    Returns (wins_a, losses_a, draws)."""
    global SELF_PLAY_MODE, CURRENT_MODEL, CURRENT_DEVICE
    SELF_PLAY_MODE = False
    CURRENT_DEVICE = device

    wins = losses = draws = 0
    OPENING_RANDOM_PLIES = 2  # de-correlates games so two near-identical
                              # deterministic policies don't just walk into
                              # the same position and repeat it forever.

    for g in tqdm(range(n_games), desc="Tournament games", leave=False):
        board = chess.Board()
        a_is_white = (g % 2 == 0)
        white_model = model_a if a_is_white else model_b
        black_model = model_b if a_is_white else model_a
        ply = 0

        # A few random legal opening plies so repeated tournament games
        # (and candidate vs. best, which start out nearly identical) don't
        # all collapse into the exact same deterministic line.
        for _ in range(OPENING_RANDOM_PLIES):
            if board.is_game_over(claim_draw=False) or board.is_repetition(3) or board.can_claim_fifty_moves():
                break
            board.push(random.choice(list(board.legal_moves)))
            ply += 1

        while not board.is_game_over(claim_draw=False) and not board.is_repetition(3) and not board.can_claim_fifty_moves() and ply < max_moves:
            # Same mate-in-1 safety net as self-play: never let search
            # second-guess a move that just ends the game outright.
            mate_move = find_immediate_mate(board)
            if mate_move is not None:
                board.push(mate_move)
                ply += 1
                continue

            CURRENT_MODEL = white_model if board.turn == chess.WHITE else black_model
            visits, _ = search(proc, board, sims=sims, threads=threads)
            best_uci = pick_safe_move_from_visits(board, visits, temperature=0.0)
            move = chess.Move.from_uci(best_uci)
            assert move in board.legal_moves, (
                f"ILLEGAL MOVE during tournament: {best_uci} at FEN '{board.fen()}'"
            )
            board.push(move)
            ply += 1

        terminal, result_for_mover = position_outcome(board)
        if not terminal:
            draws += 1
            continue
        if result_for_mover == 0.0:
            draws += 1
            continue

        winner_is_white = (result_for_mover == 1.0) == (board.turn == chess.WHITE)
        # winner_is_white True means white won this game
        a_won = (winner_is_white and a_is_white) or ((not winner_is_white) and not a_is_white)
        if a_won:
            wins += 1
        else:
            losses += 1

    return wins, losses, draws


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal AlphaZero-style chess trainer.")
    parser.add_argument("--games", type=int, required=True, help="Total number of self-play games to run, then exit.")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move during self-play.")
    parser.add_argument("--eval-sims", type=int, default=150, help="MCTS simulations per move during tournament eval.")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads inside mcts_engine per search call.")
    parser.add_argument("--games-per-gen", type=int, default=10, help="Self-play games between each train+eval cycle.")
    parser.add_argument("--eval-games", type=int, default=10, help="Games per candidate-vs-best tournament.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=None,
                         help="Optimizer steps per generation. If omitted (default), this is "
                              "auto-computed from --epochs-per-gen so the number of gradient "
                              "steps scales with how much NEW data that generation actually "
                              "produced, instead of a fixed count that -- with a small/correlated "
                              "self-play batch -- can revisit the same few hundred positions "
                              "20-40x per generation and overfit to them. Set this explicitly to "
                              "restore the old fixed-step behavior.")
    parser.add_argument("--epochs-per-gen", type=float, default=4.0,
                         help="Used only when --train-steps is not set: train for roughly this "
                              "many effective passes over the examples generated since the last "
                              "training round.")
    parser.add_argument("--min-train-steps", type=int, default=20,
                         help="Floor on auto-computed steps per generation, so a very small "
                              "generation still gets a minimally useful training round.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-moves", type=int, default=150, help="Ply cap per game before declaring a draw.")
    parser.add_argument("--temp-moves", type=int, default=15, help="Plies of temperature=1.0 exploration before playing greedily.")
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--min-decisive-for-training", type=int, default=50,
                         help="Minimum number of decisive (non-draw) games' worth of examples "
                              "that must have been added before a generation is allowed to "
                              "train at all. Note: with the simplified ReplayBuffer this is now "
                              "an approximate self-play-games-based heuristic, not an exact "
                              "buffer count -- see the check below.")
    parser.add_argument("--stall-warn-generations", type=int, default=5,
                         help="If this many consecutive generations get skipped for lack of "
                              "decisive examples, log a warning -- this means self-play itself "
                              "is not producing decisive games (not just a slow start), which "
                              "--min-decisive-for-training alone can't fix.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    setup_logging()
    log.info(f"Starting AlphaZero-chess run: {args.games} self-play games, "
             f"{args.sims} sims/move, generation size {args.games_per_gen}.")

    compile_engine()
    proc = start_engine()

    device = torch.device(args.device)
    model = DualHeadResNet().to(device)
    best_model = DualHeadResNet().to(device)

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        log.info(f"Loaded existing weights from {BEST_MODEL_PATH}.")
    else:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        log.info(f"No existing {BEST_MODEL_PATH} found; initialized and saved a fresh model.")
    best_model.load_state_dict(model.state_dict())
    best_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = ReplayBuffer(args.buffer_size)

    global CURRENT_MODEL, CURRENT_DEVICE
    CURRENT_DEVICE = device

    games_played = 0
    generation = 0
    new_examples_since_train = 0
    consecutive_training_skips = 0
    decisive_examples_seen = 0  # running counts, not buffer-derived, since
    drawn_examples_seen = 0     # ReplayBuffer no longer splits decisive/draws

    try:
        pbar = tqdm(total=args.games, desc="Self-play games")
        while games_played < args.games:
            CURRENT_MODEL = model
            model.eval()  # BatchNorm must use its learned running stats, not
                          # batch-of-1 statistics, during self-play inference.
            t0 = time.time()
            examples = play_one_game(proc, args.sims, args.threads, args.max_moves,
                                      args.temp_moves, game_index=games_played)
            for state, policy_target, z in examples:
                buffer.push(state, policy_target, z)
                if z == 0.0:
                    drawn_examples_seen += 1
                else:
                    decisive_examples_seen += 1
            new_examples_since_train += len(examples)

            games_played += 1
            pbar.update(1)
            pbar.set_postfix(buffer=len(buffer), secs=f"{time.time() - t0:.1f}")

            is_last_game = games_played == args.games
            if games_played % args.games_per_gen == 0 or is_last_game:
                generation += 1
                log.info(f"=== Generation {generation}: training after {games_played} total games "
                          f"(buffer size {len(buffer)}) ===")

                if len(buffer) < args.batch_size:
                    log.info("Replay buffer smaller than one batch; skipping training this generation.")
                    consecutive_training_skips += 1
                    continue

                # Guard against training on an all-draw buffer before any
                # decisive self-play result has come in. NOTE: this now uses
                # cumulative running counts (decisive_examples_seen), not a
                # live buffer.decisive count -- the simplified ReplayBuffer
                # doesn't track that split, and old examples can be evicted
                # from the buffer as new ones arrive. This is a reasonable
                # proxy for "has self-play produced real signal yet" but is
                # no longer an exact count of what's currently in the
                # buffer. (This whole main() is currently unused -- only
                # stockfish_train.py is run -- kept coherent for future use.)
                if decisive_examples_seen < args.min_decisive_for_training:
                    log.info(f"Only {decisive_examples_seen} decisive example(s) seen so far "
                              f"(need {args.min_decisive_for_training}); skipping training "
                              f"this generation so the value head isn't trained on an "
                              f"all-draw batch. Self-play continues.")
                    consecutive_training_skips += 1
                    if consecutive_training_skips == args.stall_warn_generations:
                        log.warning(
                            f"{consecutive_training_skips} consecutive generations skipped for "
                            f"lack of decisive examples ({decisive_examples_seen} "
                            f"decisive / {drawn_examples_seen} drawn seen so far). This isn't "
                            f"just a slow start anymore -- self-play itself is producing almost "
                            f"nothing but draws, which --min-decisive-for-training can't fix on "
                            f"its own. Consider: increasing --sims (deeper search finds decisive "
                            f"lines more often), checking --max-moves isn't cutting games off too "
                            f"early, or checking resign/repetition behavior isn't itself buggy."
                        )
                    continue
                consecutive_training_skips = 0

                if args.train_steps is not None:
                    steps_this_gen = args.train_steps
                else:
                    steps_this_gen = max(
                        args.min_train_steps,
                        math.ceil(args.epochs_per_gen * new_examples_since_train / args.batch_size),
                    )
                    log.info(f"Auto-sized training: {new_examples_since_train} new examples this "
                              f"generation, {args.epochs_per_gen} epochs -> {steps_this_gen} steps.")
                new_examples_since_train = 0

                total_pl, total_vl = 0.0, 0.0
                model.train()
                for step in tqdm(range(steps_this_gen), desc=f"Gen {generation} training", leave=False):
                    pl, vl = train_step(model, optimizer, buffer, args.batch_size, device)
                    total_pl += pl
                    total_vl += vl
                    if (step + 1) % 50 == 0:
                        log.info(f"  step {step + 1}/{steps_this_gen}  "
                                  f"policy_loss={pl:.4f}  value_loss={vl:.4f}")
                log.info(f"Generation {generation} training done. "
                          f"avg policy_loss={total_pl / steps_this_gen:.4f} "
                          f"avg value_loss={total_vl / steps_this_gen:.4f}")

                log.info(f"=== Generation {generation}: evaluation tournament "
                          f"(candidate vs best, {args.eval_games} games) ===")
                model.eval()

                # A tie (wins == losses, e.g. an all-draw tournament) is not
                # evidence of improvement -- it's inconclusive. The previous
                # version of this gate treated any tie as an automatic
                # promotion ("wins >= losses"), which is exactly backwards in
                # the one situation where it matters most: a collapsing value
                # head makes every tournament game trend toward a draw, which
                # this gate would then wave through as "improved" every single
                # generation, with nothing left to catch the regression.
                #
                # Instead: on a tie, re-run the eval tournament (fresh games)
                # up to MAX_EVAL_ATTEMPTS times, hoping a decisive result
                # emerges. If it's still tied after all attempts, treat that
                # as genuinely inconclusive and do NOT promote -- keep the
                # previous best model rather than assume progress.
                MAX_EVAL_ATTEMPTS = 3
                wins = losses = draws = 0
                for attempt in range(1, MAX_EVAL_ATTEMPTS + 1):
                    wins, losses, draws = tournament(proc, model, best_model, args.eval_games,
                                                      args.eval_sims, args.threads, args.max_moves, device)
                    log.info(f"Tournament attempt {attempt}/{MAX_EVAL_ATTEMPTS} -> "
                              f"candidate wins={wins}  losses={losses}  draws={draws}")
                    if wins != losses:
                        break
                    if attempt < MAX_EVAL_ATTEMPTS:
                        log.info(f"Tied result ({wins}-{losses}-{draws}) is inconclusive -- "
                                  f"re-running the eval tournament instead of treating it as promotion.")

                if wins > losses:
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    best_model.load_state_dict(model.state_dict())
                    best_model.eval()
                    log.info(f"RESULT: candidate IMPROVED ({wins}-{losses}-{draws}). "
                              f"Saved new {BEST_MODEL_PATH}.")
                elif losses > wins:
                    model.load_state_dict(best_model.state_dict())
                    log.info(f"RESULT: candidate did NOT improve ({wins}-{losses}-{draws}). "
                              f"Reverted in-memory model to previous {BEST_MODEL_PATH}.")
                else:
                    model.load_state_dict(best_model.state_dict())
                    log.info(f"RESULT: still tied after {MAX_EVAL_ATTEMPTS} eval attempts "
                              f"({wins}-{losses}-{draws}). Treating as inconclusive -- NOT "
                              f"promoting. Reverted in-memory model to previous {BEST_MODEL_PATH}.")

        pbar.close()
        log.info(f"Completed all {args.games} self-play games across {generation} generation(s). Exiting.")

    finally:
        shutdown_engine(proc)


if __name__ == "__main__":
    main()
