#!/usr/bin/env python3
"""
bench_multipv.py -- Measures actual wall time for MultiPV=1/2/5 at various
movetimes, and shows what the soft target distribution looks like vs one-hot.

Usage:
    python3 bench_multipv.py --stockfish-dir stockfish
"""

import argparse
import time
import math
import chess
import chess.engine


def softmax(vals, temperature):
    v = [x / temperature for x in vals]
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    return [x / s for x in exps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish-dir", default="stockfish")
    parser.add_argument("--movetime-ms", type=int, default=50)
    parser.add_argument("--reps", type=int, default=20,
                        help="Repetitions per config for timing")
    args = parser.parse_args()

    # Find binary
    import os
    candidates = [
        f"{args.stockfish_dir}/stockfish-macos-m1-apple-silicon",
        f"{args.stockfish_dir}/stockfish",
        "stockfish",
    ]
    sf_path = next((p for p in candidates if os.path.exists(p)), None)
    if sf_path is None:
        print("Stockfish binary not found. Pass --stockfish-dir.")
        return
    print(f"Using: {sf_path}")

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    limit = chess.engine.Limit(time=args.movetime_ms / 1000)

    # Use a few different positions: start, after e4, a middlegame
    positions = [
        ("Start position", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Middlegame", chess.Board("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7")),
    ]

    for pos_name, board in positions:
        print(f"\n{'='*60}")
        print(f"Position: {pos_name}")
        print(f"{'='*60}")

        for multipv in [1, 2, 5]:
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                info = engine.analyse(board, limit, multipv=multipv)
                times.append((time.perf_counter() - t0) * 1000)
            avg = sum(times) / len(times)
            mn = min(times)
            mx = max(times)
            print(f"  MultiPV={multipv}: avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")

        # Show actual distribution for MultiPV=5
        info = engine.analyse(board, limit, multipv=5)
        moves = []
        scores = []
        for pv in info:
            move = pv["pv"][0]
            cp = pv["score"].white().score(mate_score=1000)
            moves.append(move.uci())
            scores.append(cp if cp is not None else 0)

        print(f"\n  MultiPV=5 moves and scores (White perspective):")
        for m, s in zip(moves, scores):
            print(f"    {m:8s}  {s:+5d}cp")

        print(f"\n  Resulting soft policy targets at different temperatures:")
        print(f"  {'move':8s}  {'one-hot':>8s}  {'T=25cp':>8s}  {'T=50cp':>8s}  {'T=100cp':>8s}")
        print(f"  {'-'*50}")
        for temp in [None, 25, 50, 100]:
            if temp is None:
                probs = [1.0] + [0.0] * (len(scores) - 1)
                label = "one-hot"
            else:
                probs = softmax(scores, temp)
                label = f"T={temp}cp"
            row = f"  {label:>8s}  "
            row += "  ".join(f"{p:8.4f}" for p in probs)
            print(row)

        # Show margin between rank-1 and rank-2
        if len(scores) >= 2:
            margin = scores[0] - scores[1]
            print(f"\n  Margin (rank1 - rank2): {margin:+d}cp")
            print(f"  At T=50cp, rank-2 gets {softmax(scores, 50)[1]:.4f} vs rank-1 {softmax(scores, 50)[0]:.4f}")

    print(f"\n{'='*60}")
    print("TIMING SUMMARY: overhead of MultiPV vs single PV")
    print(f"{'='*60}")

    # Final timing summary across all positions combined
    board = chess.Board()
    for multipv in [1, 2, 5]:
        times = []
        for _ in range(args.reps * 3):
            t0 = time.perf_counter()
            engine.analyse(board, limit, multipv=multipv)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        overhead = avg - (sum(times[:args.reps]) / args.reps if multipv == 1 else 0)
        print(f"  MultiPV={multipv}: {avg:.1f}ms avg")

    print(f"\nData gen impact estimate:")
    base_ms = args.movetime_ms
    print(f"  150 games x 60 plies x {base_ms}ms (MultiPV=1) = {150*60*base_ms/1000/60:.1f} min")
    print(f"  (MultiPV=2 and =5 overhead shown above)")
    print(f"  Note: these calls run CONCURRENTLY with mover.play() so")
    print(f"  overhead only matters if MultiPV takes longer than mover.play()")

    engine.quit()


if __name__ == "__main__":
    main()
