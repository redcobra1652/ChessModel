"""
bench_mcts.py — MCTS batch staleness & network speed benchmarking
=================================================================
Two independent benchmarks:

  1. STALENESS  — simulates MCTS search in pure Python (no chess, synthetic
                  tree) and measures how much stale Q values distort PUCT
                  decisions at different BATCH_SIZE values.

  2. SPEED      — times actual PyTorch forward passes on MPS (or CPU) at
                  different (blocks, channels) configs and batch sizes,
                  so you can predict real eval-game wall time before committing
                  to a larger architecture.

Usage:
    python3 bench_mcts.py                  # runs both benchmarks
    python3 bench_mcts.py --staleness-only
    python3 bench_mcts.py --speed-only
    python3 bench_mcts.py --sims 800 --batch-sizes 1 4 8 16 32 64
    python3 bench_mcts.py --archs "6,64" "8,96" "10,128" "12,192"

Dependencies: torch  (pip install torch)
              numpy  (pip install numpy)
              scipy  (pip install scipy)   <- optional, for Spearman r
"""

import argparse
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy for Spearman rank correlation
# ---------------------------------------------------------------------------
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not found — Spearman rank correlation will be skipped. "
          "pip install scipy to enable it.\n")

# ---------------------------------------------------------------------------
# Optional torch for speed benchmark
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[warn] torch not found — speed benchmark will be skipped. "
          "pip install torch to enable it.\n")


# ===========================================================================
# BENCHMARK 1 — STALENESS
# ===========================================================================
# Core question: within a batch window of B sims, each descent makes its
# PUCT selection using Q values that don't yet reflect earlier sims in that
# same window (since backups only land after the whole batch is fired).
#
# Correct measurement approach:
#   - Run the FULL search serially (batch=1) as ground truth.
#     At every sim, record the "oracle" child that PUCT would pick with
#     fully up-to-date Q (this is the serial baseline).
#   - Re-run the same search with batch=B on an identical fresh tree.
#     At each sim within a batch window, record which child PUCT ACTUALLY
#     picks with stale Q.
#   - Compare: selection_error = fraction of sims where batched PUCT
#     picked a different child than serial oracle PUCT.
#   - Also record Q drift for the actually-selected child between the
#     moment of selection and when the batch backup lands.
#
# batch=1 MUST give exactly 0% error — no staleness, every sim sees
# fully updated Q. If it doesn't, the measurement logic is wrong.

BRANCHING    = 30   # chess average legal moves
C_PUCT       = 1.5
VIRTUAL_LOSS = 3


@dataclass
class SNode:
    """Synthetic MCTS node (depth-1 tree: root + children only)."""
    P:         float
    N:         int   = 0
    W:         float = 0.0
    in_flight: bool  = False
    children:  List["SNode"] = field(default_factory=list)
    value:     float = 0.0   # fixed NN value for this node

    @property
    def Q(self) -> float:
        return (self.W / self.N) if self.N > 0 else 0.0


def make_root(branching: int, rng: random.Random) -> SNode:
    """Create a root with `branching` children, random priors and values."""
    root = SNode(P=1.0, value=0.0)
    raw  = [rng.random() for _ in range(branching)]
    s    = sum(raw)
    for r in raw:
        root.children.append(SNode(P=r / s, value=rng.uniform(-1.0, 1.0)))
    return root


def clone_root(src: SNode) -> SNode:
    """Deep-clone a depth-1 root so we can replay on a fresh tree."""
    dst = SNode(P=src.P, value=src.value)
    for c in src.children:
        dst.children.append(SNode(P=c.P, value=c.value))
    return dst


def get_puct_scores(root: SNode, skip_inflight: bool) -> List[float]:
    sqN = math.sqrt(max(1, root.N))
    scores = []
    for c in root.children:
        if skip_inflight and c.in_flight:
            scores.append(-1e18)
        else:
            scores.append(-c.Q + C_PUCT * c.P * sqN / (1.0 + c.N))
    return scores


def best_child_idx(root: SNode, skip_inflight: bool) -> int:
    scores = get_puct_scores(root, skip_inflight)
    idx = int(np.argmax(scores))
    if skip_inflight and scores[idx] == -1e18:
        # all in_flight fallback
        idx = int(np.argmax(get_puct_scores(root, skip_inflight=False)))
    return idx


def apply_vl(child: SNode):
    child.N       += VIRTUAL_LOSS
    child.W       += VIRTUAL_LOSS
    child.in_flight = True


def do_backup(root: SNode, child_idx: int):
    """Back up one sim: undo VL, apply real value."""
    c = root.children[child_idx]
    c.N -= VIRTUAL_LOSS
    c.W -= VIRTUAL_LOSS
    c.in_flight = False
    leaf_val = c.value
    # child receives +1 visit, W += leaf_val
    c.N += 1
    c.W += leaf_val
    # root receives +1 visit, W -= leaf_val (negamax flip)
    root.N += 1
    root.W -= leaf_val


def run_serial(proto: SNode, sims: int) -> List[int]:
    """
    Ground-truth serial search (batch=1).
    Returns list of selected child indices in sim order.
    Every sim sees fully up-to-date Q — zero staleness by definition.
    """
    root = clone_root(proto)
    selections = []
    for _ in range(sims):
        idx = best_child_idx(root, skip_inflight=False)
        apply_vl(root.children[idx])
        do_backup(root, idx)
        selections.append(idx)
    return selections


def run_batched(proto: SNode, sims: int, batch_size: int):
    """
    Batched search: collect batch_size descents before any backup.
    For each sim records:
      - which child was actually selected (with stale Q)
      - PUCT score at selection vs after backup (Q drift)

    Returns (actual_selections, q_drifts).
    """
    root             = clone_root(proto)
    actual_selections = []
    q_drifts          = []

    completed = 0
    while completed < sims:
        # --- collection phase: pick children, apply VL, don't back up yet ---
        batch = []
        while len(batch) < batch_size and completed + len(batch) < sims:
            idx        = best_child_idx(root, skip_inflight=True)
            score_now  = get_puct_scores(root, skip_inflight=False)[idx]
            apply_vl(root.children[idx])
            batch.append((idx, score_now))

        # --- backup phase ---
        for (idx, score_at_selection) in batch:
            do_backup(root, idx)
            completed += 1
            score_after = get_puct_scores(root, skip_inflight=False)[idx]
            q_drifts.append(abs(score_at_selection - score_after))
            actual_selections.append(idx)

    return actual_selections, q_drifts


def run_staleness_benchmark(
    batch_sizes: List[int],
    sims:        int = 800,
    n_positions: int = 300,
    branching:   int = BRANCHING,
    seed:        int = 42,
) -> dict:
    rng     = random.Random(seed)
    results = {}

    for batch_size in batch_sizes:
        sel_errors  = []
        drift_all   = []
        spearman_rs = []

        for _ in range(n_positions):
            proto  = make_root(branching, rng)
            oracle = run_serial(proto, sims)
            actual, drifts = run_batched(proto, sims, batch_size)

            for o, a in zip(oracle, actual):
                sel_errors.append(int(o != a))
            drift_all.extend(drifts)

            # Spearman on final visit distributions
            if HAS_SCIPY:
                ov = [0] * branching
                av = [0] * branching
                for idx in oracle: ov[idx] += 1
                for idx in actual: av[idx] += 1
                r, _ = spearmanr(ov, av)
                if not math.isnan(r):
                    spearman_rs.append(r)

        results[batch_size] = {
            "sel_error_pct": float(np.mean(sel_errors) * 100),
            "mean_q_drift":  float(np.mean(drift_all)),
            "p95_q_drift":   float(np.percentile(drift_all, 95)),
            "spearman_r":    float(np.mean(spearman_rs)) if spearman_rs else None,
        }

    return results


def print_staleness_results(results: dict):
    batch_sizes = sorted(results.keys())
    print("\n" + "=" * 72)
    print("STALENESS BENCHMARK  (batch=1 is ground truth — must show 0.00% error)")
    print("=" * 72)
    print(f"  {'Batch':>6}  {'Select err%':>13}  {'Mean Q drift':>13}  "
          f"{'p95 Q drift':>12}  {'Spearman r':>11}")
    print("-" * 72)
    for bs in batch_sizes:
        r  = results[bs]
        sp = f"{r['spearman_r']:.4f}" if r["spearman_r"] is not None else "   n/a"
        print(f"  {bs:>6}  {r['sel_error_pct']:>12.2f}%  "
              f"{r['mean_q_drift']:>13.5f}  "
              f"{r['p95_q_drift']:>12.5f}  {sp:>11}")
    print()
    print("  Select err% — % of sims where batched PUCT picked a different child")
    print("                than fully-serial (zero-staleness) PUCT. Direct measure")
    print("                of search quality loss. batch=1 must be exactly 0.00%.")
    print("  Q drift     — how much the selected child's PUCT score shifts between")
    print("                when it was selected and when its backup lands.")
    print("  Spearman r  — rank correlation of final visit counts (serial vs batched).")
    print("                1.0 = identical exploration pattern.")
    print()


# ===========================================================================
# BENCHMARK 2 — NETWORK SPEED
# ===========================================================================
# Builds a DualHeadResNet at each (blocks, channels) config, runs warmup +
# timed forward passes at each batch_size on MPS (or CPU), and reports:
#   - ms per forward pass
#   - estimated sims/sec  (sims = passes * batch_size)
#   - estimated wall time for a 16-game eval at 800 sims
#   - relative slowdown vs 6×64 baseline

ACTION_SIZE = 4144
INPUT_CHANNELS = 13
BOARD_SIZE = 8

if HAS_TORCH:
    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.net(x))

    class DualHeadResNet(nn.Module):
        def __init__(self, blocks: int, channels: int):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
            # Policy head
            self.policy_conv = nn.Sequential(
                nn.Conv2d(channels, 2, 1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
            )
            self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)
            # Value head
            self.value_conv = nn.Sequential(
                nn.Conv2d(channels, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(BOARD_SIZE * BOARD_SIZE, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.tower(x)
            p = self.policy_conv(x).flatten(1)
            p = self.policy_fc(p)
            v = self.value_conv(x).flatten(1)
            v = self.value_fc(v)
            return p, v

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())


def pick_device() -> "torch.device":
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bench_forward(
    model: "nn.Module",
    device: "torch.device",
    batch_size: int,
    n_warmup: int = 20,
    n_timed: int = 100,
) -> float:
    """Returns mean ms per forward pass."""
    model.eval()
    dummy = torch.randn(batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device).startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_timed):
            model(dummy)
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) / n_timed * 1000  # ms


def run_speed_benchmark(
    archs: List[Tuple[int, int]],
    batch_sizes: List[int],
    sims: int = 800,
    eval_games: int = 16,
    pipe_overhead_ms: float = 18.0,  # estimated round-trip pipe cost on macOS
) -> dict:
    """
    For each (blocks, channels) architecture and each batch_size:
      - Build the model on device
      - Time forward passes
      - Estimate wall time for eval_games at `sims` sims each
    """
    device = pick_device()
    print(f"\nDevice: {device}")
    results = {}

    for (blocks, channels) in archs:
        arch_key = f"{blocks}x{channels}"
        results[arch_key] = {}

        model = DualHeadResNet(blocks, channels).to(device)
        params = model.param_count()

        print(f"  Building {arch_key} ({params/1e6:.2f}M params)...", flush=True)

        for bs in batch_sizes:
            ms_per_pass = bench_forward(model, device, bs)

            # Passes per game = ceil(sims / bs) + 1 (root call is separate,
            # single pass of bs=1). We model pipe overhead per round-trip.
            passes_per_game = math.ceil(sims / bs) + 1
            forward_ms_per_game = ms_per_pass * passes_per_game
            pipe_ms_per_game    = pipe_overhead_ms * passes_per_game
            total_ms_per_game   = forward_ms_per_game + pipe_ms_per_game
            total_min_16games   = (total_ms_per_game * eval_games) / 60_000

            results[arch_key][bs] = {
                "params_M":           params / 1e6,
                "ms_per_pass":        ms_per_pass,
                "passes_per_game":    passes_per_game,
                "fwd_ms_per_game":    forward_ms_per_game,
                "pipe_ms_per_game":   pipe_ms_per_game,
                "total_ms_per_game":  total_ms_per_game,
                "est_min_16games":    total_min_16games,
            }

        del model
        if HAS_TORCH and str(device) == "mps":
            torch.mps.empty_cache()

    return results, device


def print_speed_results(results: dict, batch_sizes: List[int], device):
    archs = list(results.keys())
    baseline_arch = archs[0]

    print("\n" + "=" * 90)
    print("SPEED BENCHMARK  (forward pass timing + eval wall-time estimate)")
    print(f"Device: {device}   Pipe overhead assumption: ~18ms/round-trip")
    print("=" * 90)

    for bs in batch_sizes:
        print(f"\n--- Batch size = {bs} ---")
        print(f"{'Architecture':>14}  {'Params':>8}  {'ms/pass':>9}  "
              f"{'passes/game':>12}  {'fwd min':>8}  {'pipe min':>9}  "
              f"{'total min':>10}  {'vs 6x64':>8}")
        print("-" * 90)

        baseline_total = results[baseline_arch][bs]["total_ms_per_game"] * 16 / 60_000

        for arch in archs:
            r = results[arch][bs]
            ratio = r["est_min_16games"] / baseline_total if baseline_total > 0 else 1.0
            fwd_min = r["fwd_ms_per_game"] * 16 / 60_000
            pipe_min = r["pipe_ms_per_game"] * 16 / 60_000
            print(f"{arch:>14}  {r['params_M']:>7.2f}M  {r['ms_per_pass']:>8.2f}ms"
                  f"  {r['passes_per_game']:>12}  {fwd_min:>7.1f}m  "
                  f"{pipe_min:>8.1f}m  {r['est_min_16games']:>9.1f}m  "
                  f"{ratio:>7.2f}x")

    print()
    print("Notes:")
    print("  fwd min  = time spent in GPU forward passes across 16 games")
    print("  pipe min = time spent in Python<->C++ pipe round-trips across 16 games")
    print("  total    = fwd + pipe  (does NOT include board-replay or Python overhead)")
    print("  vs 6x64  = slowdown multiplier relative to 6x64 at this batch size")
    print("  Actual wall time will be ~1.3-1.8x 'total' due to Python overhead.")
    print()

    # Best batch size recommendation per arch
    print("--- Recommended batch size per architecture (minimises est. total time) ---")
    print(f"{'Architecture':>14}  {'Best batch':>11}  {'Est. 16-game min':>18}")
    print("-" * 50)
    for arch in archs:
        best_bs = min(batch_sizes, key=lambda bs: results[arch][bs]["est_min_16games"])
        best_min = results[arch][best_bs]["est_min_16games"]
        print(f"{arch:>14}  {best_bs:>11}  {best_min:>17.1f}m")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def parse_arch(s: str) -> Tuple[int, int]:
    parts = s.replace("x", ",").split(",")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description="MCTS batch staleness & network speed benchmark")
    parser.add_argument("--staleness-only", action="store_true")
    parser.add_argument("--speed-only",     action="store_true")
    parser.add_argument("--sims",      type=int, default=800)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--positions", type=int, default=300,
                        help="Synthetic positions for staleness benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--archs", type=str, nargs="+",
                        default=["6,64", "8,96", "10,128", "12,192", "16,256"],
                        help="Architecture specs as 'blocks,channels' e.g. '10,128'")
    parser.add_argument("--pipe-overhead-ms", type=float, default=18.0,
                        help="Estimated pipe round-trip latency in ms (default 18ms for macOS)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archs = [parse_arch(a) for a in args.archs]
    run_staleness = not args.speed_only
    run_speed     = not args.staleness_only

    # -----------------------------------------------------------------------
    # Staleness benchmark
    # -----------------------------------------------------------------------
    if run_staleness:
        print("\nRunning staleness benchmark...")
        print(f"  {args.positions} synthetic positions × {args.sims} sims each")
        print(f"  Branching factor: {BRANCHING}  |  C_PUCT: {C_PUCT}  |  VL: {VIRTUAL_LOSS}")
        staleness = run_staleness_benchmark(
            batch_sizes=args.batch_sizes,
            sims=args.sims,
            n_positions=args.positions,
            seed=args.seed,
        )
        print_staleness_results(staleness)

        # Highlight key thresholds
        print("Key thresholds for your setup (800 sims, VL=3, branching=30):")
        for bs, r in sorted(staleness.items()):
            sel_err = r["sel_error_pct"]
            if sel_err < 2.0:
                flag = "✓ safe"
            elif sel_err < 8.0:
                flag = "~ acceptable"
            else:
                flag = "✗ significant quality loss"
            print(f"  batch={bs:>3}: {sel_err:.2f}% selection error  {flag}")
        print()

    # -----------------------------------------------------------------------
    # Speed benchmark
    # -----------------------------------------------------------------------
    if run_speed:
        if not HAS_TORCH:
            print("Skipping speed benchmark — torch not available.")
        else:
            print("Running speed benchmark...")
            print(f"  Architectures: {[f'{b}x{c}' for b,c in archs]}")
            print(f"  Batch sizes:   {args.batch_sizes}")
            speed, device = run_speed_benchmark(
                archs=archs,
                batch_sizes=args.batch_sizes,
                sims=args.sims,
                eval_games=args.eval_games,
                pipe_overhead_ms=args.pipe_overhead_ms,
            )
            print_speed_results(speed, args.batch_sizes, device)

            # Combined recommendation
            print("=" * 70)
            print("COMBINED RECOMMENDATION (staleness + speed)")
            print("=" * 70)
            if run_staleness:
                # Find the largest batch_size with <5% selection error
                safe_batches = [bs for bs, r in staleness.items()
                                if r["sel_error_pct"] < 8.0]
                best_safe_bs = max(safe_batches) if safe_batches else args.batch_sizes[0]
                print(f"  Largest 'safe' batch size (<5% selection error): {best_safe_bs}")
                print()
                print(f"  At batch_size={best_safe_bs}:")
                for arch_key in speed:
                    if best_safe_bs in speed[arch_key]:
                        r = speed[arch_key][best_safe_bs]
                        print(f"    {arch_key:>10}: ~{r['est_min_16games']:.1f} min "
                              f"for 16 eval games  "
                              f"({'within' if r['est_min_16games'] < 25 else 'OVER'} 25-min budget)")
            print()


if __name__ == "__main__":
    main()
