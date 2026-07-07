#!/usr/bin/env python3
"""
reset_value_head.py -- Re-initializes ONLY the value head of best_model.pt,
leaving the conv trunk and policy head exactly as trained.

Why: diagnose_value_head.py confirmed the value head has collapsed --
it outputs roughly the same value regardless of whether a position is
completely winning, completely losing, or genuinely drawn. That's a bad
local optimum, not just "undertrained." Continuing to train it as-is
risks the same collapse dynamic repeating, since gradient descent has no
particular reason to climb back out on its own once the buffer is
draw-heavy.

The conv trunk (piece-pattern features) and policy head (which move
looks good) are a separate matter -- policy loss has been decreasing
normally and self-play checkmate rate is holding around 80%, i.e. the
trunk/policy side of the network is doing real work. There's no
diagnostic reason to throw that away, so this only resets:

    value_conv, value_bn, value_fc1, value_fc2

Everything else in the checkpoint is copied through unchanged.

Usage (run in the same directory as train.py and best_model.pt):

    python3 reset_value_head.py

This makes a backup (best_model.pt.pre_value_reset.bak) before writing,
and won't overwrite an existing backup from a previous run.
"""

import os
import shutil
import sys

import torch

import train

SRC = "best_model.pt"
BACKUP = SRC + ".pre_value_reset.bak"

VALUE_HEAD_PREFIXES = ("value_conv", "value_bn", "value_fc1", "value_fc2")


def main():
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found in the current directory.")
        sys.exit(1)

    if os.path.exists(BACKUP):
        print(f"ERROR: {BACKUP} already exists -- refusing to overwrite an "
              f"existing backup. Move or delete it first if you really want "
              f"to run this again.")
        sys.exit(1)

    device = torch.device("cpu")
    old_state = torch.load(SRC, map_location=device)

    fresh_model = train.DualHeadResNet().to(device)
    fresh_state = fresh_model.state_dict()

    new_state = dict(old_state)
    reset_keys = [k for k in fresh_state if k.startswith(VALUE_HEAD_PREFIXES)]
    for k in reset_keys:
        new_state[k] = fresh_state[k]

    kept = len(new_state) - len(reset_keys)
    print(f"Resetting {len(reset_keys)} value-head tensors to fresh init:")
    for k in reset_keys:
        print(f"    {k}")
    print(f"Keeping {kept} trunk/policy-head tensors exactly as trained.")

    shutil.copy(SRC, BACKUP)
    print(f"Backed up original weights to {BACKUP}")

    torch.save(new_state, SRC)
    print(f"Wrote value-head-reset weights to {SRC}")
    print()
    print("You can now resume training normally (python3 train.py --games ...). "
          "Expect the value loss to look 'bad' again for the first generation or "
          "two while the fresh value head relearns from real data -- that's "
          "expected, not a regression.")


if __name__ == "__main__":
    main()
