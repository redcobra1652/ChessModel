#!/usr/bin/env python3
"""
eval_only.py -- Pure evaluation: play the current model against a fixed-Elo
Stockfish for N games and report the score. No training, no buffer, no
checkpoint writes. Safe to run as many times as you want.

Usage:
    python3 eval_only.py --model best_model.pt --elo 1320 --games 30 \
        --sims 400 --stockfish-movetime-ms 100 --device mps
"""

import argparse
import torch
import chess.engine

import train
from stockfish_train import find_stockfish_binary, start_stockfish, make_limit
from eval_game_logger import run_eval_batch_with_pgn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=train.BEST_MODEL_PATH)
    parser.add_argument("--stockfish-dir", type=str, default="stockfish")
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--stockfish-hash-mb", type=int, default=64)
    parser.add_argument("--elo", type=int, default=1320)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--pgn-out", type=str, default="eval_sample_games.pgn",
                         help="Path to save a stratified sample of games (win/loss/draw mix)")
    parser.add_argument("--pgn-sample-size", type=int, default=5)
    args = parser.parse_args()

    train.setup_logging()
    device = torch.device(args.device)

    model = train.DualHeadResNet().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {args.model}")

    train.compile_engine()
    proc = train.start_engine()

    sf_path = find_stockfish_binary(args.stockfish_dir, args.stockfish_path)
    sf_engine = start_stockfish(sf_path, args.stockfish_threads, args.stockfish_hash_mb)
    sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
    sf_limit = make_limit(args.stockfish_movetime_ms, None)

    try:
        result = run_eval_batch_with_pgn(
            proc, sf_engine, model, device, args.sims, args.threads,
            args.max_moves, args.games, sf_limit, args.elo,
            desc=f"Eval vs Stockfish@{args.elo}",
            pgn_sample_path=args.pgn_out, pgn_sample_size=args.pgn_sample_size,
        )
        print(f"\n=== RESULT vs Stockfish@{args.elo} over {result['total']} games ===")
        print(f"Wins: {result['wins']}  Losses: {result['losses']}  Draws: {result['draws']}")
        print(f"Score: {result['score']:.1%}")
        print(f"Saved {min(args.pgn_sample_size, result['total'])} sample games to {args.pgn_out}")
    finally:
        sf_engine.quit()
        train.shutdown_engine(proc)


if __name__ == "__main__":
    main()
