#!/usr/bin/env python3
"""
migrate_checkpoint.py

Migrates a best_model.pt checkpoint from the old ACTION_SIZE=4096 policy head
to the new ACTION_SIZE=4144 policy head (adds 48 underpromotion slots).

All weights are preserved exactly. The 48 new rows in policy_fc.weight and
policy_fc.bias are initialized with the same default that nn.Linear uses
(Kaiming uniform for weight, uniform for bias), so the model starts with
sensible small values for underpromotion logits rather than zeros.

Usage:
    python3 migrate_checkpoint.py --input best_model.pt --output best_model.pt

The script detects whether the checkpoint is old-format (raw state_dict) or
new-format (dict with "model" key) and preserves the format on output.
"""

import argparse
import math
import torch
import torch.nn as nn


OLD_ACTION_SIZE = 4096
NEW_ACTION_SIZE = 4144  # 4096 + 48 underpromotion slots
POLICY_FC_WEIGHT_KEY = "policy_fc.weight"
POLICY_FC_BIAS_KEY   = "policy_fc.bias"


def kaiming_uniform_rows(n_rows: int, fan_in: int) -> torch.Tensor:
    """Initialize `n_rows` new weight rows with Kaiming uniform (same as
    nn.Linear default), matching PyTorch's own init so the new slots are
    neither dead nor oversized relative to the existing weights."""
    bound = math.sqrt(1.0 / fan_in) if fan_in > 0 else 0.0
    return torch.empty(n_rows, fan_in).uniform_(-bound, bound)


def uniform_bias_rows(n_rows: int, fan_in: int) -> torch.Tensor:
    """Initialize `n_rows` new bias values with the same uniform range
    PyTorch uses for nn.Linear bias (bound = 1/sqrt(fan_in))."""
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    return torch.empty(n_rows).uniform_(-bound, bound)


def migrate_state_dict(sd: dict) -> dict:
    w = sd[POLICY_FC_WEIGHT_KEY]  # shape: [old_action_size, hidden]
    b = sd[POLICY_FC_BIAS_KEY]    # shape: [old_action_size]

    current_out, fan_in = w.shape
    if current_out == NEW_ACTION_SIZE:
        print(f"Checkpoint policy head already has {NEW_ACTION_SIZE} outputs -- nothing to migrate.")
        return sd
    if current_out != OLD_ACTION_SIZE:
        raise ValueError(
            f"Unexpected policy_fc output size {current_out}. "
            f"Expected {OLD_ACTION_SIZE} (old) or {NEW_ACTION_SIZE} (already migrated)."
        )

    n_new = NEW_ACTION_SIZE - OLD_ACTION_SIZE  # 48
    new_w_rows = kaiming_uniform_rows(n_new, fan_in)
    new_b_rows = uniform_bias_rows(n_new, fan_in)

    sd = dict(sd)  # shallow copy so we don't mutate the original
    sd[POLICY_FC_WEIGHT_KEY] = torch.cat([w, new_w_rows], dim=0)
    sd[POLICY_FC_BIAS_KEY]   = torch.cat([b, new_b_rows], dim=0)

    print(f"Migrated policy_fc: [{OLD_ACTION_SIZE}, {fan_in}] -> [{NEW_ACTION_SIZE}, {fan_in}]")
    print(f"  weight: appended {n_new} rows (Kaiming uniform, bound={math.sqrt(1.0/fan_in):.4f})")
    print(f"  bias:   appended {n_new} values (uniform, bound={1.0/math.sqrt(fan_in):.4f})")
    return sd


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoint policy head from 4096 to 4144 outputs.")
    parser.add_argument("--input",  type=str, default="best_model.pt", help="Input checkpoint path")
    parser.add_argument("--output", type=str, default="best_model.pt", help="Output checkpoint path")
    args = parser.parse_args()

    print(f"Loading checkpoint from '{args.input}' ...")
    raw = torch.load(args.input, map_location="cpu")

    if isinstance(raw, dict) and "model" in raw:
        # New-format checkpoint (from updated stockfish_train.py)
        print("Detected new-format checkpoint (has 'model' key).")
        raw["model"] = migrate_state_dict(raw["model"])
        out = raw
    else:
        # Old-format checkpoint: raw state_dict
        print("Detected legacy checkpoint (raw state_dict).")
        out = migrate_state_dict(raw)

    torch.save(out, args.output)
    print(f"Saved migrated checkpoint to '{args.output}'.")
    print()
    print("Next steps:")
    print("  1. Replace train.py with the updated version (ACTION_SIZE=4144, new move_policy_index).")
    print("  2. Run stockfish_train.py as normal -- it will load the migrated checkpoint.")
    print("  3. The replay buffer (replay_buffer.pkl) contains examples with 4096-wide policy")
    print("     targets. These are INCOMPATIBLE with the new model and should be deleted:")
    print("       rm replay_buffer.pkl")
    print("     The buffer will rebuild from scratch over the first few generations.")


if __name__ == "__main__":
    main()
