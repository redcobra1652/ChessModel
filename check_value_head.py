#!/usr/bin/env python3
"""
check_value_head.py -- Quick standalone check of whether the CURRENT
best_model.pt's value head can tell winning/losing/drawn positions apart.

This calls the network directly (no MCTS, no engine subprocess) so it
isolates the value head from any search or play.py issues. If this shows
a healthy spread of values across the test positions, the "shuffling
pieces, ignoring blunders" behavior you're seeing in play.py is coming
from somewhere in search/play, not from the value head itself. If this
shows a tight, uninformative band again (like the original collapse),
the value head genuinely hasn't learned material evaluation yet and
that's the root cause of the weak play.

Usage (run in the same directory as train.py and best_model.pt):

    python3 check_value_head.py
    python3 check_value_head.py --model gen5_model.pt   # check a specific checkpoint
"""

import argparse

import chess
import numpy as np
import torch

import train

TEST_POSITIONS = [
    ("Start position (balanced)",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "~0"),
    ("White to move, White up a whole queen",
     "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "strongly positive"),
    ("White to move, White down a whole queen",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNB1KBNR w KQkq - 0 1",
     "strongly negative"),
    ("Black to move, Black up a whole queen",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNB1KBNR b KQkq - 0 1",
     "strongly positive"),
    ("Black to move, Black down a whole queen",
     "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
     "strongly negative"),
    ("White to move, White has only a king vs Black's full army",
     "rnbqkbnr/pppppppp/8/8/8/8/8/4K3 w kq - 0 1",
     "strongly negative (mover is nearly lost)"),
    ("Black to move, Black has only a king vs White's full army",
     "4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1",
     "strongly negative (mover is nearly lost)"),
    ("Dead draw: king vs king, insufficient material",
     "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
     "~0 (correctly drawn, not confused)"),
    ("White to move, White up a single hanging rook (no other imbalance)",
     "4k3/8/8/8/8/8/4K3/R7 w - - 0 1",
     "positive"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=train.BEST_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model = train.DualHeadResNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("=" * 100)
    print(f"Checking value head of: {args.model}")
    print("=" * 100)
    print(f"{'Position':<62}{'value':>8}   expected")
    print("-" * 100)

    values = []
    for name, fen, expected in TEST_POSITIONS:
        board = chess.Board(fen)
        tensor = train.board_to_tensor(board)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = model(x)
        v = value.item()
        values.append(v)
        print(f"{name:<62}{v:>8.4f}   {expected}")

    values = np.array(values)
    print("-" * 100)
    print(f"std across all positions: {values.std():.4f}   max |value|: {np.abs(values).max():.4f}")
    if values.std() < 0.05 and np.abs(values).max() < 0.15:
        print(">>> LOOKS COLLAPSED: outputs barely move across wildly different material")
        print("    balances. The value head is not distinguishing winning from losing")
        print("    from drawn positions -- this alone would explain ignoring hanging")
        print("    material and drifting into repetition in play.py.")
    else:
        print(">>> Value head is producing a real spread across these positions.")
        print("    If play.py is still ignoring hanging material, the issue is more")
        print("    likely in search/engine behavior or a stale/mismatched checkpoint")
        print("    file, not the value head itself.")


if __name__ == "__main__":
    main()
