"""
eval_game_logger.py -- Play eval games while recording full PGN move lists,
and save a small, outcome-stratified sample (not just whatever came first /
all losses) to a .pgn file for manual review.

Import and use `run_eval_batch_with_pgn` in place of `run_eval_batch` when
you want representative sample games saved to disk. Falls back to picking
whatever is available if a given outcome type didn't occur enough times.
"""

import random
import chess
import chess.pgn
from tqdm import tqdm

import train


def play_one_eval_game_pgn(mcts_proc, sf_engine, model, device, sims, threads,
                            max_moves, model_is_white, sf_limit, game_index):
    """Same as play_one_eval_game, but also returns a chess.pgn.Game with
    every move (both sides) recorded, plus per-move MCTS root value for the
    model's own moves as a comment (handy for spotting where it misjudged
    a position)."""
    train.SELF_PLAY_MODE = False
    train.CURRENT_DEVICE = device
    train.CURRENT_MODEL = model
    model_color = chess.WHITE if model_is_white else chess.BLACK
    board = chess.Board()
    ply = 0
    model_records = []

    game = chess.pgn.Game()
    game.headers["White"] = "Model" if model_is_white else f"Stockfish"
    game.headers["Black"] = "Stockfish" if model_is_white else "Model"
    node = game

    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        if board.turn == model_color:
            mate_move = train.find_immediate_mate(board)
            comment = None
            if mate_move is not None:
                move = mate_move
                comment = "immediate mate"
            else:
                visits, root_value = train.search(mcts_proc, board, sims=sims, threads=threads)
                best_uci = train.pick_safe_move_from_visits(board, visits, temperature=0.0)
                move = chess.Move.from_uci(best_uci)
                comment = f"model root_value={root_value:.3f}"
            assert move in board.legal_moves
            model_records.append((board.fen(), move.uci()))
            node = node.add_variation(move)
            if comment:
                node.comment = comment
        else:
            sf_move = sf_engine.play(board, sf_limit).move
            assert sf_move in board.legal_moves
            node = node.add_variation(sf_move)
        board.push(node.move)
        ply += 1

    terminal, result_for_mover = train.position_outcome(board)
    if not terminal or result_for_mover == 0.0:
        outcome = "draw"
    else:
        winner_is_white = (result_for_mover == 1.0) == (board.turn == chess.WHITE)
        model_won = winner_is_white == model_is_white
        outcome = "win" if model_won else "loss"

    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["ModelColor"] = "White" if model_is_white else "Black"
    game.headers["Outcome"] = outcome  # from the model's perspective
    game.headers["Plies"] = str(ply)
    return outcome, ply, model_records, game


def run_eval_batch_with_pgn(mcts_proc, sf_engine, model, device, sims, threads,
                             max_moves, n_games, sf_limit, elo, desc,
                             sf_teacher=None, corrective_limit=None,
                             corrective_cp_threshold=30.0,
                             corrective_eval_blend=0.5,
                             pgn_sample_path=None, pgn_sample_size=5,
                             seed=None):
    """Same behavior/return value as stockfish_train.run_eval_batch, but also
    saves a small, outcome-stratified sample of games to `pgn_sample_path`.

    Stratification: with pgn_sample_size=5 and outcomes present across win/
    loss/draw, this tries to save a spread (e.g. ~2 losses, ~2 wins, ~1 draw
    for a typical below-50% score) rather than whatever games happen to be
    first or all of one outcome. If one outcome type doesn't have enough
    games, the remaining slots are filled from other outcomes.
    """
    rng = random.Random(seed)
    wins = losses = draws = 0
    all_corrective = []
    do_corrective = sf_teacher is not None and corrective_limit is not None
    if do_corrective:
        from stockfish_train import analyse_eval_game  # lazy import: avoids circular import with stockfish_train.py
    games_by_outcome = {"win": [], "loss": [], "draw": []}

    for g in tqdm(range(n_games), desc=desc, leave=False):
        model_is_white = (g % 2 == 0)
        outcome, plies, model_records, pgn_game = play_one_eval_game_pgn(
            mcts_proc, sf_engine, model, device, sims, threads,
            max_moves, model_is_white, sf_limit, g)

        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
            if do_corrective:
                model_color = chess.WHITE if model_is_white else chess.BLACK
                corrective = analyse_eval_game(
                    sf_teacher, model_records, model_color,
                    z_outcome=-1.0,
                    corrective_limit=corrective_limit,
                    cp_threshold=corrective_cp_threshold,
                    eval_blend=corrective_eval_blend,
                )
                all_corrective.extend(corrective)
        else:
            draws += 1

        games_by_outcome[outcome].append(pgn_game)

    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total else 0.0

    if pgn_sample_path and total > 0:
        sample = _stratified_sample(games_by_outcome, pgn_sample_size, rng)
        with open(pgn_sample_path, "w") as f:
            for g in sample:
                print(g, file=f, end="\n\n")

    return {"wins": wins, "losses": losses, "draws": draws, "total": total, "score": score,
            "corrective_examples": all_corrective}


def _stratified_sample(games_by_outcome, sample_size, rng):
    """Pick up to sample_size games spread across win/loss/draw in proportion
    to how many of each occurred, guaranteeing at least one of each outcome
    that actually happened (so a 30%-score batch doesn't get you 5 losses)."""
    outcomes = [o for o in ("win", "loss", "draw") if games_by_outcome[o]]
    if not outcomes:
        return []

    # Start by guaranteeing 1 of each outcome that occurred.
    picked = []
    remaining_pool = {o: list(games_by_outcome[o]) for o in outcomes}
    for o in outcomes:
        if len(picked) >= sample_size:
            break
        g = rng.choice(remaining_pool[o])
        picked.append(g)
        remaining_pool[o].remove(g)

    # Fill remaining slots proportionally to how common each outcome was,
    # picking randomly within each bucket.
    pool_flat = [(o, g) for o in outcomes for g in remaining_pool[o]]
    rng.shuffle(pool_flat)
    for o, g in pool_flat:
        if len(picked) >= sample_size:
            break
        picked.append(g)

    rng.shuffle(picked)
    return picked
