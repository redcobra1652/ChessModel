#!/usr/bin/env python3
"""
checkpoint_vs_checkpoint.py -- Play two saved model checkpoints against
each other using train.py's existing tournament() function. Read-only:
no training, no buffer writes, no checkpoint writes. Safe to run anytime,
including while another training run is in progress (uses its own MCTS
subprocess).

Handles both checkpoint formats:
  - Full training checkpoint: {"model": state_dict, "optimizer": ..., "lr": ..., ...}
  - Bare state dict: {"conv_in.weight": ..., ...}

Architecture (channels, blocks) is auto-detected from the checkpoint's
weight shapes, so pretrain checkpoints and stockfish_train checkpoints
with different architectures can be compared directly.

Usage:
    python3 checkpoint_vs_checkpoint.py --model-a best_model.pt \
        --model-b pretrain_checkpoint.pt --games 30 --sims 400 --device mps
"""

import argparse
import torch

import train


def load_model(path: str, device: torch.device) -> train.DualHeadResNet:
    """Load a model from a checkpoint file, handling both full training
    checkpoints (with optimizer state etc.) and bare state dicts.
    Auto-detects architecture from weight shapes."""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Unwrap full training checkpoint if needed.
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model" and k != "optimizer"}
    else:
        state_dict = ckpt
        meta = {}

    # Auto-detect architecture from weight shapes.
    # conv_in.weight shape: (channels, 13, 3, 3)
    channels = state_dict["conv_in.weight"].shape[0]
    # Count residual blocks by finding all blocks.N.conv1.weight keys.
    n_blocks = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".conv1.weight"))

    model = train.DualHeadResNet(channels=channels, n_blocks=n_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    arch_str = f"{n_blocks}×{channels}"
    meta_str = ""
    if meta:
        parts = []
        if "global_gen" in meta:
            parts.append(f"gen={meta['global_gen']}")
        if "num_promotions" in meta:
            parts.append(f"promotions={meta['num_promotions']}")
        if "lr" in meta:
            parts.append(f"lr={meta['lr']:.2e}")
        if parts:
            meta_str = "  (" + ", ".join(parts) + ")"
    print(f"Loaded {path}  [{arch_str}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params]{meta_str}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, required=True,
                        help="Path to first checkpoint (treated as 'candidate').")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Path to second checkpoint (treated as 'baseline').")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)

    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)

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
        print(f"\n=== Results: {args.model_a} (A) vs {args.model_b} (B) over {total} games ===")
        print(f"A wins: {wins_a}   A losses: {losses_a}   Draws: {draws}")
        print(f"A score: {score_a:.1%}  (50% = evenly matched, >50% = A stronger, <50% = B stronger)")
    finally:
        train.shutdown_engine(proc)


if __name__ == "__main__":
    main()
