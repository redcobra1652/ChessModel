#!/usr/bin/env python3
"""
checkpoint_vs_checkpoint.py -- Play two saved model checkpoints against
each other using train.py's existing tournament() function. Read-only:
no training, no buffer writes, no checkpoint writes. Safe to run anytime,
including while another training run is in progress (uses its own MCTS
subprocess).

Usage:
    python3 checkpoint_vs_checkpoint.py --model-a best_model.pt \
        --model-b best_model_gen10.pt --games 30 --sims 400 --device mps

If you only have one saved checkpoint (most setups overwrite best_model.pt
every generation and don't keep history), this can still be useful to:
  - Sanity-check a model against itself (should score ~50%, confirms the
    tournament harness / search isn't systematically biased by color).
  - Compare best_model.pt against an older copy you saved manually.
"""

import argparse
import torch

import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, required=True, help="Path to first checkpoint (candidate).")
    parser.add_argument("--model-b", type=str, required=True, help="Path to second checkpoint (baseline).")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)

    model_a = train.DualHeadResNet().to(device)
    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_a.eval()
    print(f"Loaded model A from {args.model_a}")

    model_b = train.DualHeadResNet().to(device)
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))
    model_b.eval()
    print(f"Loaded model B from {args.model_b}")

    train.compile_engine()
    proc = train.start_engine()

    try:
        wins_a, losses_a, draws = train.tournament(
            proc, model_a, model_b,
            n_games=args.games, sims=args.sims, threads=args.threads,
            max_moves=args.max_moves, device=device,
        )
        total = wins_a + losses_a + draws
        score_a = (wins_a + 0.5 * draws) / total if total else 0.0
        print(f"\n=== {args.model_a} (A) vs {args.model_b} (B) over {total} games ===")
        print(f"A wins: {wins_a}   A losses: {losses_a}   Draws: {draws}")
        print(f"A score: {score_a:.1%}  (50% = evenly matched, >50% = A stronger, <50% = B stronger)")
    finally:
        train.shutdown_engine(proc)


if __name__ == "__main__":
    main()
