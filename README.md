# ChessModel

An AlphaZero-style chess engine written in Python and C++. The project trains a dual-head (policy + value) ResNet entirely from self-play, optionally warm-started on real human games, and ships a graphical interface for playing against the trained model.

---

## Architecture

```
ChessModel/
├── mcts.cpp              # Multi-threaded MCTS engine (C++); communicates with Python over JSON stdin/stdout
├── mcts_engine           # Compiled binary (built automatically at runtime)
├── train.py              # Self-play training loop (AlphaZero-style)
├── pretrain.py           # Supervised warm-start on real human PGN games
├── play.py               # Graphical interface (pygame) for playing against the model
├── check_value_head.py   # Diagnostic: inspect value-head outputs
├── reset_value_head.py   # Utility: re-initialise the value head in a checkpoint
├── best_model.pt         # Current best model weights
├── assets/               # Chess piece PNGs and board images (used by play.py)
├── sound/                # Sound effects (used by play.py)
└── warm_train/           # Lichess broadcast PGN files used by pretrain.py
```

### Model

A dual-head ResNet that takes a 13-channel board tensor as input and outputs:
- **Policy head** — probability distribution over all 4096 (from, to) moves.
- **Value head** — scalar estimate of the position value from the current player's perspective (−1 to +1).

### MCTS Engine (`mcts.cpp`)

A persistent, multi-threaded Monte Carlo Tree Search engine compiled as a native binary. Python spawns it as a subprocess and drives it over a line-based JSON protocol. Neural network evaluation requests are handled on the Python side (via PyTorch) and returned to the engine over the same pipe.

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

Install Python dependencies:
```bash
pip install torch python-chess numpy tqdm pygame pyperclip
```

---

## Quick Start

### 1. (Optional) Supervised pre-training on human games

Download Lichess broadcast PGNs (or any PGN collection) into `warm_train/` and run:

```bash
python3 pretrain.py --pgn warm_train/*.pgn --epochs 5
```

This produces `best_model.pt`, which `train.py` will continue training from.

### 2. Self-play training

```bash
python3 train.py --games 100
```

Run `python3 train.py --help` to see all tunable knobs (number of MCTS simulations, replay buffer size, tournament games, learning rate, etc.).

The training loop:
1. Generates self-play games using the current best network.
2. Trains a candidate network on the replay buffer.
3. Pits the candidate against the previous best in a short tournament.
4. Promotes the candidate to `best_model.pt` only if it wins more than it loses; otherwise reverts.

### 3. Play against the model

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
| Review moves | Scroll the mouse wheel over the sidebar |
| New game | Press `r` after game over |
| Quit | Press `q` or close the window |

---

## Model Checkpoints

| File | Description |
|------|-------------|
| `best_model.pt` | Current best model |
| `best_model_2months_checkpoint.pt` | Checkpoint after ~2 months of training |
| `best_model_4months_checkpoint.pt` | Checkpoint after ~4 months of training |
| `best_model_5months_checkpoint.pt` | Checkpoint after ~5 months of training |
| `best_model_july_checkpoint.pt` | July milestone checkpoint |

---

## Notes

- Training is designed to run on CPU on macOS out of the box. Pass `--device mps` to `train.py` or `play.py` to use Apple Silicon GPU acceleration.
- The `mcts_engine` binary is compiled automatically from `mcts.cpp` the first time it is needed.
- Large PGN training files (`warm_train/*.pgn`) are excluded from version control via `.gitignore`.
- Model `.pt` checkpoint files are tracked by git by default. Comment out the relevant lines in `.gitignore` if you prefer not to track them.
