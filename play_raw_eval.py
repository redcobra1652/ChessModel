#!/usr/bin/env python3
"""
play_raw_eval.py -- Experimental variant of play.py where the model picks
moves by raw one-ply value-head eval instead of MCTS visit counts.

For each legal move: push it, run the position through the network's
value head once, take that as the move's score, then pick the move whose
resulting position looks worst for the opponent (i.e. best for us).
No search, no policy head involvement, no priors -- purely a single
forward pass per legal move.

This is NOT what train.py or play.py do during real play -- it exists
purely to A/B against real MCTS play, per our discussion of why visit
count (aggregated, low-variance) differs from raw single-node eval
(unaggregated, high-variance, and only as good as the value head's
one-shot accuracy on that exact position).

Usage:
    python3 play_raw_eval.py
    python3 play_raw_eval.py --color black
    python3 play_raw_eval.py --model best_model.pt
"""

import argparse
import sys

import chess
import torch

import train


def render_board(board: chess.Board):
    print()
    print(board)
    print()


def parse_human_move(board: chess.Board, text: str):
    text = text.strip()
    if not text:
        return None
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def pick_best_raw_eval_move(model, board: chess.Board, device):
    """Evaluates every legal move by pushing it and running the value
    head once on the resulting position. Picks whichever resulting
    position is worst for the side about to move there (i.e. best for
    us), matching the mover-relative sign convention used everywhere
    else in this pipeline.
    """
    model.eval()
    best_move = None
    best_score = None
    with torch.no_grad():
        for move in board.legal_moves:
            board.push(move)
            state = train.board_to_tensor(board)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            _, value = model(state_t)
            # value is from the perspective of whoever is to move in the
            # resulting position (our opponent) -- flip sign to score it
            # from our own perspective before pushing.
            score = -value.item()
            board.pop()
            if best_score is None or score > best_score:
                best_score = score
                best_move = move
    return best_move, best_score


def print_result(board: chess.Board, human_is_white: bool):
    outcome = board.outcome(claim_draw=True)
    print("\n" + "=" * 60)
    print(f"Game over: {outcome}")
    if outcome is None or outcome.winner is None:
        print("It's a draw.")
    else:
        human_won = (outcome.winner == chess.WHITE) == human_is_white
        print("You won!" if human_won else "The model won.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Play against a model using raw eval move selection (no MCTS).")
    parser.add_argument("--model", type=str, default=train.BEST_MODEL_PATH)
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    train.setup_logging()
    device = torch.device(args.device)

    model = train.DualHeadResNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except FileNotFoundError:
        print(f"Could not find model weights at '{args.model}'.")
        sys.exit(1)
    model.eval()

    human_is_white = args.color == "white"
    board = chess.Board()

    print("=" * 60)
    print(f"Playing against {args.model} (RAW EVAL mode -- no MCTS)")
    print(f"You are {'White' if human_is_white else 'Black'}.")
    print("Enter moves in SAN (e.g. 'Nf3', 'e4', 'O-O') or UCI (e.g. 'g1f3').")
    print("Type 'quit' to exit, 'board' to redraw, 'moves' to list legal moves.")
    print("=" * 60)
    render_board(board)

    try:
        while not board.is_game_over(claim_draw=True):
            human_turn = (board.turn == chess.WHITE) == human_is_white

            if human_turn:
                move = None
                while move is None:
                    text = input("Your move: ").strip()
                    if text.lower() in ("quit", "exit"):
                        print("Goodbye.")
                        return
                    if text.lower() == "board":
                        render_board(board)
                        continue
                    if text.lower() == "moves":
                        print(", ".join(sorted(board.san(m) for m in board.legal_moves)))
                        continue
                    move = parse_human_move(board, text)
                    if move is None:
                        print("Not a legal move -- try again ('moves' to see options).")
                board.push(move)
            else:
                print("Model is thinking (raw eval, no search)...")
                move, score = pick_best_raw_eval_move(model, board, device)
                print(f"Model plays: {board.san(move)}  (raw eval: {score:+.3f})")
                board.push(move)

            render_board(board)

        print_result(board, human_is_white)

    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")


if __name__ == "__main__":
    main()
