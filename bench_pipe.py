"""
bench_pipe.py — measure actual Python<->mcts_engine pipe round-trip latency
============================================================================
Starts the mcts_engine process exactly as train.py does, sends real
visit/batch requests for a fixed position, and times the round-trips.

Usage:
    python3 bench_pipe.py
    python3 bench_pipe.py --engine ./mcts_engine --rounds 200
    python3 bench_pipe.py --batch-sizes 1 4 8 16 32

What it measures:
    single  — one visit request -> one response (what backup.cpp did)
    batch=N — one batch of N visit requests -> N responses (current engine)

The position used is the starting position. We send dummy visit requests
for e2e4 (a real legal first move) so the engine has a valid request to
process. The model is NOT involved — we mock Python's side, so this measures
pure pipe + C++ JSON parsing overhead only.

To measure with the real model included, use --with-model and point it at
your train.py search() function instead.
"""

import argparse
import json
import subprocess
import sys
import time
import statistics

# Starting position FEN and a trivial one-move history
START_FEN     = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN  = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Fake NN response for a visit request — 30 random moves with flat priors
FAKE_MOVES  = [f"e{r}{e}{r+1}" for r, e in zip(range(2,8), "abcdef")][:20]
FAKE_PRIORS = [1.0 / 20] * 20
FAKE_VISIT_RESPONSE = json.dumps({
    "fen":      AFTER_E4_FEN,
    "terminal": False,
    "result":   0.0,
    "moves":    FAKE_MOVES,
    "priors":   FAKE_PRIORS,
    "value":    0.05,
})
FAKE_ROOT_RESPONSE = json.dumps({
    "terminal": False,
    "result":   0.0,
    "moves":    FAKE_MOVES,
    "priors":   FAKE_PRIORS,
    "value":    0.05,
})


def start_engine(engine_path: str) -> subprocess.Popen:
    return subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )


def send_line(proc: subprocess.Popen, line: str):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def recv_line(proc: subprocess.Popen) -> str:
    return proc.stdout.readline().rstrip("\n")


def do_search_handshake(proc: subprocess.Popen, sims: int = 1):
    """
    Send a search command and handle the root request so the engine
    is ready to accept visit requests.
    """
    send_line(proc, json.dumps({
        "cmd": "search",
        "fen": START_FEN,
        "history": [],
        "sims": sims,
        "threads": 1,
    }))
    # Engine sends root request
    root_req = json.loads(recv_line(proc))
    assert root_req.get("type") == "root", f"Expected root request, got: {root_req}"
    send_line(proc, FAKE_ROOT_RESPONSE)


def bench_single_roundtrip(engine_path: str, n_rounds: int) -> list:
    """
    Measure batch=1 round-trip as the baseline (engine always batches now).
    """
    return bench_batch_roundtrip(engine_path, batch_size=1, n_rounds=n_rounds)


def bench_batch_roundtrip(engine_path: str, batch_size: int, n_rounds: int) -> list:
    """
    Measure one batch round-trip: engine sends batch request, we respond.
    Timing wraps: batch request arrives -> we send batch_result -> engine sends result.

    IMPORTANT: sims=batch_size+1 does NOT guarantee the whole search finishes
    in a single batch round-trip. Whether it does depends on the engine's
    *own compiled-in* BATCH_SIZE, which this script does not know and should
    not assume matches the `batch_size` parameter here. If the engine's
    internal batch size is smaller, it will keep sending further "batch"
    requests after the first one, and a harness that only answers once will
    leave the engine hanging on stdin (which is exactly what produced the
    "FATAL: stdin closed awaiting batch_result" crash for batch_size >= 8).

    Fix: loop, answering every "batch" request as it arrives, until the
    engine sends the final "result" message. We time only the *first* batch
    round-trip (request -> response) per game, which is the quantity we
    actually want (single round-trip latency at this batch size) -- but we
    still drain and answer any subsequent batch requests so the engine
    process terminates cleanly and doesn't corrupt the next round's timing.
    """
    latencies = []

    fake_batch_response = json.dumps({
        "cmd":       "batch_result",
        "responses": [FAKE_VISIT_RESPONSE] * batch_size,
    })

    for i in range(n_rounds):
        proc = start_engine(engine_path)
        try:
            do_search_handshake(proc, sims=batch_size + 1)

            first_latency = None
            awaiting_first_reply = False
            t0 = None

            while True:
                line = recv_line(proc)
                if line == "":
                    # Engine exited unexpectedly (EOF on stdout) before
                    # sending "result". Don't silently drop this round --
                    # surface it so a protocol mismatch is visible instead
                    # of quietly shrinking the sample size.
                    raise RuntimeError(
                        f"round {i}: engine closed stdout before sending "
                        f"'result' (batch_size={batch_size})"
                    )

                obj = json.loads(line)
                cmd = obj.get("cmd")

                if awaiting_first_reply:
                    # This is the engine's next message after our first
                    # batch_result -- whatever it is (another "batch" or
                    # "result"), its arrival completes the first round-trip.
                    first_latency = (time.perf_counter() - t0) * 1000
                    awaiting_first_reply = False
                    # fall through: still need to handle `obj` below

                if cmd == "batch":
                    if first_latency is None and t0 is None:
                        t0 = time.perf_counter()
                        awaiting_first_reply = True
                    send_line(proc, fake_batch_response)
                    continue

                if cmd == "result":
                    break

                raise AssertionError(
                    f"round {i}: unexpected message while awaiting batch/result: "
                    f"{list(obj.keys())}"
                )

            if first_latency is not None:
                latencies.append(first_latency)
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)

    return latencies


def stats(latencies: list) -> dict:
    return {
        "mean":   statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95":    sorted(latencies)[int(len(latencies) * 0.95)],
        "stdev":  statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min":    min(latencies),
        "max":    max(latencies),
    }


def print_stats(label: str, s: dict):
    print(f"  {label:<18}  mean={s['mean']:6.2f}ms  "
          f"median={s['median']:6.2f}ms  "
          f"p95={s['p95']:6.2f}ms  "
          f"stdev={s['stdev']:5.2f}ms  "
          f"[{s['min']:.2f}..{s['max']:.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",      default="./mcts_engine",
                        help="Path to compiled mcts_engine binary")
    parser.add_argument("--rounds",      type=int, default=100,
                        help="Measurement rounds per configuration")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[4, 8, 16, 32],
                        help="Batch sizes to benchmark")
    parser.add_argument("--warmup",      type=int, default=10,
                        help="Warmup rounds to discard (process startup noise)")
    args = parser.parse_args()

    print(f"Engine:  {args.engine}")
    print(f"Rounds:  {args.warmup} warmup + {args.rounds} timed per config")
    print()

    # ------------------------------------------------------------------
    # Single round-trip (what backup.cpp did — one visit per round-trip)
    # ------------------------------------------------------------------
    print("Measuring batch=1 round-trip (baseline — one visit per round-trip)...")
    _ = bench_single_roundtrip(args.engine, args.warmup)  # discard warmup
    single_lats = bench_single_roundtrip(args.engine, args.rounds)
    single_stats = stats(single_lats)
    print_stats("batch=1 (baseline)", single_stats)
    print()

    # ------------------------------------------------------------------
    # Batch round-trips
    # ------------------------------------------------------------------
    print(f"Measuring batch round-trips (current batched engine)...")
    print(f"  {'Config':<18}  {'mean':>8}  {'median':>8}  {'p95':>8}  "
          f"{'stdev':>7}  {'range':>14}  {'ms/visit':>10}  {'vs single':>10}")
    print("  " + "-" * 90)

    for bs in args.batch_sizes:
        _ = bench_batch_roundtrip(args.engine, bs, args.warmup)  # warmup
        lats = bench_batch_roundtrip(args.engine, bs, args.rounds)
        s = stats(lats)
        ms_per_visit = s["mean"] / bs
        vs_single    = s["mean"] / single_stats["mean"]
        print(f"  batch={bs:<12}  "
              f"{s['mean']:>7.2f}ms  "
              f"{s['median']:>7.2f}ms  "
              f"{s['p95']:>7.2f}ms  "
              f"{s['stdev']:>6.2f}ms  "
              f"[{s['min']:.1f}..{s['max']:.1f}]ms  "
              f"{ms_per_visit:>9.2f}ms  "
              f"{vs_single:>9.2f}x")

    print()
    print("Interpretation:")
    print("  ms/visit   — effective per-sim pipe cost at this batch size")
    print("               (total round-trip time / batch_size)")
    print("  vs single  — how much longer the batch round-trip takes vs single")
    print("               Ideal: 1.0x (no overhead for batching)")
    print("               Reality: slightly >1.0x due to larger JSON payload")
    print()
    print("Plug 'mean' into bench_mcts.py --pipe-overhead-ms for accurate estimates.")
    print(f"  e.g. python3 bench_mcts.py --pipe-overhead-ms <mean from batch=8 above>")


if __name__ == "__main__":
    main()
