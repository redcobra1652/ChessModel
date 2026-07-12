# ChessModel

An AlphaZero-style chess engine written in Python and C++. The project trains a dual-head (policy + value) ResNet through multiple complementary stages: supervised warm-start on human games, self-play reinforcement learning, Stockfish-guided curriculum training, and targeted endgame fine-tuning. A full graphical interface lets you play against any trained checkpoint.

---

## Architecture

```
ChessModel/
├── mcts.cpp                        # Multi-threaded MCTS engine (C++); communicates with Python over JSON stdin/stdout
├── mcts_engine                     # Compiled binary (built automatically at runtime; git-ignored)
├── train.py                        # Self-play training loop (AlphaZero-style)
├── pretrain.py                     # Supervised warm-start on real human PGN games
├── stockfish_train.py              # Stockfish-guided curriculum training (adaptive Elo ratchet)
├── train_endgame.py                # Targeted endgame fine-tuning pass
├── play.py                         # Graphical interface (pygame) for playing against the model
├── play_raw_eval.py                # Experimental: move selection via raw one-ply value-head eval (no MCTS)
├── check_value_head.py             # Diagnostic: inspect value-head outputs across canonical positions
├── reset_value_head.py             # Utility: re-initialise the value head in a checkpoint
├── best_model.pt                   # Current best model weights (git-tracked)
├── best_model_stockfish_train.pt   # Best weights from Stockfish curriculum stage (git-tracked)
├── stockfish_curriculum_log.jsonl  # Per-generation Stockfish training log (git-ignored)
├── replay_buffer.pkl               # Self-play replay buffer (git-ignored; ~1 GB)
├── assets/                         # Chess piece PNGs and board images (used by play.py)
├── sound/                          # Sound effects (used by play.py)
├── stockfish/                      # Stockfish binary (git-ignored)
└── warm_train/                     # Lichess broadcast PGN files used by pretrain.py (git-ignored)
```

### Model

A dual-head ResNet that takes a 13-channel board tensor as input and outputs:
- **Policy head** — probability distribution over all 4096 (from, to) moves.
- **Value head** — scalar estimate of the position value from the current player's perspective (−1 to +1).

### MCTS Engine (`mcts.cpp`)

A persistent, multi-threaded Monte Carlo Tree Search engine compiled as a native binary. Python spawns it as a subprocess and drives it over a line-based JSON protocol. Neural network evaluation requests are handled on the Python side (via PyTorch) and returned to the engine over the same pipe.

---

## Training Pipeline

The model is trained through four complementary stages, each building on the last.

### Stage 1 — Supervised pre-training (`pretrain.py`)

Warm-starts the network on real human games (Lichess broadcast PGNs in `warm_train/`) before handing off to self-play. Prevents the value-head collapse that plagues a cold-start AlphaZero loop:
- Policy target = the move actually played (one-hot).
- Value target = the game's real final result, expressed from each mover's own perspective.
- Supports optional blending with PGN `%eval` comment scores for denser signal.

### Stage 2 — Self-play RL (`train.py`)

The core AlphaZero loop:
1. Generates self-play games using the current best network (with Dirichlet noise for exploration).
2. Trains a candidate network on the replay buffer.
3. Pits the candidate against the previous best in a short tournament.
4. Promotes the candidate to `best_model.pt` only if it wins more than it loses; otherwise reverts.

Key details:
- Resignation (`RESIGN_THRESHOLD = -0.90`, 3 consecutive moves) with 10% forced-play games to keep the value head calibrated in resign-zone positions.
- Replay buffer persisted to `replay_buffer.pkl` for warm restart across sessions.

### Stage 3 — Stockfish curriculum (`stockfish_train.py`)

Learns directly from Stockfish to break out of the self-play cold-start loop:

- **Competitive game generation**: Two *evenly matched* Stockfish instances at the current curriculum Elo play each game against each other. Using identically-rated opponents ensures outcomes span wins, losses, and draws, keeping the value target `z` over the full [−1, +1] range rather than collapsing to a constant.
- **Full-strength annotation**: A separate, always-full-strength Stockfish instance (`sf_teacher`) acts purely as an annotator. At every ply of every game, it analyses the position and its top move / evaluation become the policy target and eval-based value-target component for that training example. Policy signal stays at 3200+ Elo quality even while the game is played at a weaker adaptive Elo.
- **Adaptive Elo ratchet**: Calibration → initial Elo → promote when the model's own MCTS (not Stockfish) beats the current-Elo opponent above `--promotion-threshold`. Both competitive engines are always kept in lockstep with the current curriculum Elo.
- Saves to `best_model_stockfish_train.pt` (and optionally `best_model.pt` via `--output`). Progress tracked in `stockfish_curriculum_log.jsonl`.

### Stage 4 — Endgame fine-tuning (`train_endgame.py`)

Supplements stages 1–3 because the curriculum games rarely reach clean endgames, leaving the value head poorly calibrated for simple theoretical wins/draws:

- Generates tabular endgame positions (K+Q vs K, K+R vs K, drawn K vs K, etc.) across all four color/turn combinations to avoid color bias.
- Plays out each position with Stockfish and weights the loss by outcome quality:

  | Theoretical | Actual | Weight |
  |------------|--------|--------|
  | Win | Win | +0.80 |
  | Win | Draw | −0.50 |
  | Win | Loss | −1.00 |
  | Draw | Win | +0.15 |
  | Draw | Draw | 0.00 |
  | Draw | Loss | −0.30 |

- Re-run whenever `check_value_head.py` shows a regression in value-head calibration.

---

## Diagnostics

### `check_value_head.py`

Runs the network directly (no MCTS, no subprocess) on a set of canonical positions to verify the value head produces a meaningful spread across winning, losing, and drawn material imbalances. Run this any time play quality regresses unexpectedly.

```bash
python3 check_value_head.py
python3 check_value_head.py --model best_model_stockfish_train.pt
```

### `play_raw_eval.py`

Experimental terminal-mode interface where the model picks moves via raw one-ply value-head evaluation (no search). Useful for A/B comparisons against full MCTS play to isolate whether issues are in the value head or in search.

### `reset_value_head.py`

Re-initialises the value head weights in an existing checkpoint without touching the policy head. Useful when the value head has collapsed but the policy head is still healthy.

---

## Requirements

- Python 3.10+
- PyTorch (CPU works; pass `--device mps` for Apple Silicon GPU acceleration)
- `python-chess`
- `numpy`
- `tqdm`
- `pygame` (required only for `play.py`)
- `pyperclip` (optional, for PGN clipboard copy in `play.py`)
- A C++17-compatible compiler (e.g. `clang++` on macOS — the engine is compiled automatically)
- Stockfish binary placed in `stockfish/` (required for `stockfish_train.py` and `train_endgame.py`)

Install Python dependencies:
```bash
pip install torch python-chess numpy tqdm pygame pyperclip
```

---

## Quick Start

### 1. (Optional) Supervised pre-training on human games

Download Lichess broadcast PGNs (or any PGN collection) into `warm_train/` and run:

```bash
python3 pretrain.py --pgn warm_train/*.pgn --output best_model.pt
# or point at the whole directory:
python3 pretrain.py --pgn warm_train/ --output best_model.pt
```

### 2. Self-play training

```bash
python3 train.py --games 100
```

Run `python3 train.py --help` to see all tunable knobs (number of MCTS simulations, replay buffer size, tournament games, learning rate, etc.).

### 3. Stockfish curriculum training

Place a Stockfish binary in `stockfish/` then:

```bash
python3 stockfish_train.py --stockfish-dir stockfish
```

Run `python3 stockfish_train.py --help` for all options (`--batch-games`, `--elo-step`, `--promotion-threshold`, `--output`, etc.).

### 4. Endgame fine-tuning

```bash
python3 train_endgame.py --model best_model.pt --stockfish-dir stockfish \
    --device mps --positions 1000 --train-steps 50
```

Re-run whenever `check_value_head.py` shows value-head regression.

### 5. Play against the model

```bash
python3 play.py
```

A pygame window opens with a chess.com-style board. Click or drag pieces to move. The model responds using MCTS search.

**Key controls:**
| Action | How |
|--------|-----|
| Select a piece | Click it |
| Move | Click a highlighted square, or drag & drop |
| Premove | Click/drag one of your pieces while it's the model's turn |
| Cancel premove | Click the premoved piece again |
| Analysis overlay | Press `a` or click the Analysis button |
| Review moves | Scroll the mouse wheel over the sidebar |
| Toggle eval bar | "Eval Sidebar" button in right panel |
| Fullscreen | Press `F11` |
| New game | Press `r` after game over |
| Quit | Press `q` or close the window |

---

## Model Checkpoints

| File | Description | Git-tracked |
|------|-------------|-------------|
| `best_model.pt` | Current best model (self-play or combined pipeline) | ✅ |
| `best_model_stockfish_train.pt` | Best weights from Stockfish curriculum stage | ✅ |
| `best_model_1_year.pt` | ~1-year training milestone | ❌ (ignored) |
| `best_model_5months_checkpoint.pt` | ~5-month checkpoint | ❌ (ignored) |
| `best_model_7monthschkpoint.pt` | ~7-month checkpoint | ❌ (ignored) |
| `backup.pt` | Manual backup snapshot | ❌ (ignored) |

> All `.pt` checkpoints **except** `best_model.pt` and `best_model_stockfish_train.pt` are excluded from version control by `.gitignore`. To force-track a specific checkpoint, use `git add -f <file>.pt`.

---

## Notes

- Training is designed to run on CPU on macOS out of the box. Pass `--device mps` to any training script to use Apple Silicon GPU acceleration.
- The `mcts_engine` binary is compiled automatically from `mcts.cpp` the first time it is needed.
- Large PGN training files (`warm_train/*.pgn`) and the replay buffer (`replay_buffer.pkl`) are excluded from version control — they can be hundreds of MB to several GB.
- The `stockfish/` directory is also git-ignored; download a Stockfish release binary and place it there manually.
- The Stockfish curriculum log (`stockfish_curriculum_log.jsonl`) is git-ignored to avoid committing large, append-only log files.
