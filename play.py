#!/usr/bin/env python3
"""
play.py -- Play chess against a trained AlphaZero-style model, on a
graphical, chess.com-style board (pygame).

Reuses train.py's model definition, MCTS engine lifecycle, and search
glue code directly, so the model plays exactly the way it does during
self-play/tournament (same engine binary, same search protocol) --
just with a human clicking/dragging pieces for the other side instead
of a second network.

Controls:
    - Click a piece to select it. Legal destination squares light up
      with a dot (a ring around the square for captures).
    - Click a highlighted square to move there, OR just drag the piece
      and drop it on the destination square.
    - Clicking the already-selected piece again unselects it.
    - Dropping/clicking on a non-legal square snaps the piece back and
      plays an "illegal move" (buzzer) sound.
    - While it's the model's turn, click one of your own pieces and then
      any other square (or drag-and-drop it) to queue a premove -- both
      squares light up solid blue, chess.com-style, with no destination
      dots (a premove can target literally any square, including a
      square occupied by one of your own pieces -- handy for queuing a
      recapture before the opponent's move that creates it even lands;
      only the real legality once each of the model's moves actually
      lands decides whether it fires). You can queue up several premoves
      in a row: they fire one at a time, in order, one per model move,
      and the moment any of them turns out illegal the rest of the queue
      is discarded (same as chess.com). Click the start square of any
      queued premove to cancel just that one.
    - Promoting a pawn pops up a small picker for Q/R/B/N (premove
      promotions default to queen).
    - Scroll with the mouse wheel over the sidebar to review earlier
      moves; it auto-follows the latest move again once a new move lands.
    - The "Eval Sidebar" toggle near the bottom of the right panel shows
      or hides the eval bar. The "Analysis" button next to it opens a
      full-window review of every move the model has made so far this
      game -- its search's evaluation of the position and the top
      candidate moves it considered (by search-visit share), alongside
      the move it actually played. Press 'a' or Esc to toggle/close it,
      and scroll with the mouse wheel while it's open.
    - When the game ends, a popup offers "Copy PGN" (copies the full game
      as PGN to your clipboard -- uses pyperclip if installed, else
      pygame's own clipboard support) and "Rematch as White" / "Rematch
      as Black" (start a new game playing the chosen color; pressing 'r'
      repeats a same-color rematch).
    - Press 'r' after a game ends to start a new game (same color).
    - The window is resizable, and can be dragged to any size -- the
      board/sidebar scale to fit. Press F11 to toggle fullscreen.
    - Press 'q' or close the window to quit.

Sound effects (chess.com's default set) are loaded from --sounds
(default "sound/") -- game-start, game-end, move-self, move-opponent,
move-check, capture, castle, promote, premove, illegal. The
notification sound (notify.mp3) is intentionally not used anywhere.

The board background is drawn from --assets/checkboard_white.png (used
when you're playing White) or --assets/checkboard_black.png (used when
you're playing Black), whatever their native resolution -- each is
center-cropped to a square and scaled to fit, so mismatched or
non-square source art (e.g. 3352x3340 next to 1286x1280) just works.

Usage:
    python3 play.py                       # you play White vs best_model.pt
    python3 play.py --color black         # you play Black
    python3 play.py --sims 800            # give the model more thinking time
    python3 play.py --model gen5_model.pt # play a specific checkpoint
    python3 play.py --assets assets       # folder containing piece/board PNGs
    python3 play.py --sounds sound        # folder containing sound .mp3 files

Piece images are expected at:
    <assets>/white_pawn.png   <assets>/black_pawn.png
    <assets>/white_knight.png <assets>/black_knight.png
    <assets>/white_bishop.png <assets>/black_bishop.png
    <assets>/white_rook.png   <assets>/black_rook.png
    <assets>/white_queen.png  <assets>/black_queen.png
    <assets>/white_king.png   <assets>/black_king.png

Board background images are expected at:
    <assets>/checkboard_white.png
    <assets>/checkboard_black.png

Sound files are expected at:
    <sounds>/game-start.mp3   <sounds>/game-end.mp3
    <sounds>/move-self.mp3    <sounds>/move-opponent.mp3
    <sounds>/move-check.mp3   <sounds>/capture.mp3
    <sounds>/castle.mp3       <sounds>/promote.mp3
    <sounds>/premove.mp3      <sounds>/illegal.mp3
"""

import argparse
import os
import queue
import sys
import threading
import time

import chess
import chess.pgn
import pygame
import torch

import train

try:
    import pyperclip
    _HAVE_PYPERCLIP = True
except ImportError:
    _HAVE_PYPERCLIP = False


def copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard copy: pyperclip first, then pygame's own
    (SDL2) clipboard support, falling back to just printing the text so
    nothing is lost if neither is available in this environment."""
    if _HAVE_PYPERCLIP:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            pass
    try:
        pygame.scrap.init()
        pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        return True
    except Exception:
        pass
    print(f"[clipboard unavailable] {text}")
    return False


def build_pgn(board: chess.Board, human_is_white: bool, model_name: str) -> str:
    """Builds a PGN string for the whole game played so far (every move,
    not just the final position), tagging the human/model players and
    the result."""
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Casual Game"
    game.headers["White"] = "Human" if human_is_white else os.path.basename(model_name)
    game.headers["Black"] = os.path.basename(model_name) if human_is_white else "Human"
    return str(game)

# ----------------------------------------------------------------------
# Layout / appearance constants
# ----------------------------------------------------------------------

SQUARE = 93
BOARD_PX = SQUARE * 8

# Eval bar sits to the left of the board; a toggle in the right-hand
# sidebar can hide it (board/sidebar positions stay fixed either way, so
# toggling doesn't reflow the layout).
EVAL_BAR_W = 50
BOARD_X = EVAL_BAR_W

# Piece images are 256x256 source assets. We scale them down within their
# square (rather than stretching them edge-to-edge) to leave a
# chess.com-style buffer between the piece and the square border.
PIECE_SCALE = 1.01
PIECE_SIZE = int(SQUARE * PIECE_SCALE)
PIECE_OFFSET = (SQUARE - PIECE_SIZE) // 2
SIDEBAR_W = 300
SIDEBAR_X = BOARD_X + BOARD_PX
WINDOW_W = BOARD_X + BOARD_PX + SIDEBAR_W
WINDOW_H = BOARD_PX

EVAL_BAR_BG = (40, 40, 42)
EVAL_WHITE_COLOR = (250, 250, 250)

SIDEBAR_BG = (34, 36, 40)
SIDEBAR_TEXT = (235, 235, 235)
SIDEBAR_SUBTEXT = (150, 155, 160)

# Board background art (see load_board_backgrounds). Two separate images
# are used depending on board orientation (which color sits at the
# bottom), since they're typically pre-rendered with matching coordinate
# labels for that orientation. Source assets may not be perfectly square
# (e.g. 3352x3340) -- they get center-cropped to square, then scaled.
BOARD_BG_FILES = {
    "white": "checkboard_white.png",
    "black": "checkboard_black.png",
}

# Semi-transparent tint overlays drawn on top of the board background art
# for highlighting (selection, last move, premoves, check).
SELECTED_TINT = (246, 234, 120, 150)
LAST_MOVE_TINT = (210, 210, 130, 140)
CHECK_TINT = (220, 90, 80, 150)
DOT_COLOR = (30, 30, 30, 110)
RING_COLOR = (30, 30, 30, 140)

# Premove highlighting: chess.com-style solid-ish blue tint on both the
# start and end square, no destination dots.
PREMOVE_TINT = (70, 130, 230, 150)

PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PROMOTION_CHOICES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Fixed screen-space rect for the "Show Eval Bar" toggle switch, drawn in
# the top-right of the sidebar. Kept as a constant (rather than computed
# during drawing) so click hit-testing and rendering can't drift apart.
EVAL_TOGGLE_W, EVAL_TOGGLE_H = 46, 24
EVAL_TOGGLE_RECT = pygame.Rect(SIDEBAR_X + SIDEBAR_W - 18 - EVAL_TOGGLE_W, WINDOW_H - 44,
                                EVAL_TOGGLE_W, EVAL_TOGGLE_H)

# End-of-game popup: centered over the whole window, with "Copy PGN" and
# "Analysis" side by side on the top row, and "Rematch as White" /
# "Rematch as Black" side by side below them.
POPUP_W, POPUP_H = 420, 270
POPUP_RECT = pygame.Rect((WINDOW_W - POPUP_W) // 2, (WINDOW_H - POPUP_H) // 2, POPUP_W, POPUP_H)
POPUP_BTN_W, POPUP_BTN_H = 170, 46
POPUP_BTN_GAP = 20
POPUP_ROW_GAP = 14

_popup_row2_y = POPUP_RECT.bottom - 30 - POPUP_BTN_H
_popup_row1_y = _popup_row2_y - POPUP_ROW_GAP - POPUP_BTN_H
_popup_row_total_w = POPUP_BTN_W * 2 + POPUP_BTN_GAP
_popup_row_x0 = POPUP_RECT.left + (POPUP_W - _popup_row_total_w) // 2

COPY_PGN_BTN_RECT = pygame.Rect(_popup_row_x0, _popup_row1_y, POPUP_BTN_W, POPUP_BTN_H)
POPUP_ANALYSIS_BTN_RECT = pygame.Rect(_popup_row_x0 + POPUP_BTN_W + POPUP_BTN_GAP, _popup_row1_y,
                                       POPUP_BTN_W, POPUP_BTN_H)
REMATCH_WHITE_BTN_RECT = pygame.Rect(_popup_row_x0, _popup_row2_y, POPUP_BTN_W, POPUP_BTN_H)
REMATCH_BLACK_BTN_RECT = pygame.Rect(_popup_row_x0 + POPUP_BTN_W + POPUP_BTN_GAP, _popup_row2_y,
                                      POPUP_BTN_W, POPUP_BTN_H)

# Sidebar "Analysis" button (opens the full-window move-analysis panel),
# drawn next to the "Eval Sidebar" toggle at the bottom of the right
# panel. Fixed screen-space rect, same rationale as EVAL_TOGGLE_RECT.
ANALYSIS_BTN_W, ANALYSIS_BTN_H = 100, 26
ANALYSIS_BTN_RECT = pygame.Rect(SIDEBAR_X + 18, WINDOW_H - 44 - 1, ANALYSIS_BTN_W, ANALYSIS_BTN_H)

# Close ("X") button for the analysis overlay, top-right of the window.
ANALYSIS_CLOSE_BTN_RECT = pygame.Rect(WINDOW_W - 50, 16, 34, 34)



# ----------------------------------------------------------------------
# Sound effects (chess.com-style). "notify.mp3" is intentionally not
# wired up anywhere -- the person asked to ignore the notification sound.
# ----------------------------------------------------------------------

SOUND_FILES = {
    "game-start": "game-start.mp3",
    "game-end": "game-end.mp3",
    "capture": "capture.mp3",
    "castle": "castle.mp3",
    "premove": "premove.mp3",
    "move-self": "move-self.mp3",
    "move-opponent": "move-opponent.mp3",
    "move-check": "move-check.mp3",
    "promote": "promote.mp3",
    "illegal": "illegal.mp3",
}


def load_sounds(sound_dir: str):
    sounds = {}
    for key, filename in SOUND_FILES.items():
        path = os.path.join(sound_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: missing sound file '{path}' -- '{key}' will be silent.")
            continue
        try:
            sounds[key] = pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Warning: could not load sound '{path}': {e}")
    return sounds


# ----------------------------------------------------------------------
# Asset loading
# ----------------------------------------------------------------------

def load_piece_images(assets_dir: str):
    images = {}
    for color, color_name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for pt, piece_name in PIECE_NAME.items():
            path = os.path.join(assets_dir, f"{color_name}_{piece_name}.png")
            if not os.path.exists(path):
                print(f"Missing piece asset: {path}")
                sys.exit(1)
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, (PIECE_SIZE, PIECE_SIZE))
            images[(color, pt)] = img
    return images


def load_board_backgrounds(assets_dir: str):
    """Loads the two board background images and returns them scaled to
    exactly BOARD_PX x BOARD_PX. Source art doesn't have to be square or
    any particular resolution (e.g. 3352x3340, or 1286x1280 for a
    differently-sized second asset) -- it's center-cropped to a square
    using its smaller dimension first, then scaled down/up to fit."""
    backgrounds = {}
    for key, filename in BOARD_BG_FILES.items():
        path = os.path.join(assets_dir, filename)
        if not os.path.exists(path):
            print(f"Missing board background asset: {path}")
            sys.exit(1)
        img = pygame.image.load(path).convert()
        w, h = img.get_size()
        side = min(w, h)
        x_off, y_off = (w - side) // 2, (h - side) // 2
        if x_off or y_off:
            img = img.subsurface(pygame.Rect(x_off, y_off, side, side)).copy()
        backgrounds[key] = pygame.transform.smoothscale(img, (BOARD_PX, BOARD_PX))
    return backgrounds


# ----------------------------------------------------------------------
# Board <-> screen coordinate helpers
# ----------------------------------------------------------------------

def square_to_pixel(square: int, flipped: bool):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if not flipped:
        col, row = file, 7 - rank
    else:
        col, row = 7 - file, rank
    return BOARD_X + col * SQUARE, row * SQUARE


def pixel_to_square(pos, flipped: bool):
    x, y = pos
    x -= BOARD_X
    if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
        return None
    col, row = x // SQUARE, y // SQUARE
    if not flipped:
        file, rank = col, 7 - row
    else:
        file, rank = 7 - col, row
    return chess.square(file, rank)


def compute_premove_piece_map(board, premoves):
    """The board's piece placement after optimistically applying every
    queued premove in order, purely for display/selection purposes -- the
    real board itself is untouched. Used both to render premoved pieces on
    their (not-yet-real) destination squares, and to let the player pick
    up such a piece again to queue the *next* link of a premove chain
    (e.g. premove a pawn e2-e4, then, without waiting for it to land,
    premove the same pawn onward from e4-e5)."""
    piece_map = {sq: board.piece_at(sq) for sq in chess.SQUARES if board.piece_at(sq) is not None}
    for from_sq, to_sq in premoves:
        moved_piece = piece_map.pop(from_sq, None)
        if moved_piece is not None:
            piece_map[to_sq] = moved_piece
    return piece_map


def premove_shape_ok(board, premoves, from_sq, to_sq):
    """Loose legality filter applied when queuing a premove: restricts the
    destination to a square that's at least geometrically reachable for
    the piece being (optimistically) moved -- i.e. the correct movement
    pattern for its type (pawn forward/diagonal, knight L-shape, bishop
    diagonal, rook straight, queen either, king one square or a castling
    slide). It deliberately ignores board occupancy, check, and whose
    turn it actually is otherwise -- a premove is still allowed to target
    a currently-occupied-by-own-piece square, or a square that's only
    reachable because of how prior queued premoves will have moved
    things -- exactly mirroring chess.com's own input restriction on
    premoves. The real legality check happens later, once the premove is
    next in line to fire (see try_execute_premove)."""
    if from_sq == to_sq:
        return False
    piece = compute_premove_piece_map(board, premoves).get(from_sq)
    if piece is None:
        return False

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr
    adf, adr = abs(df), abs(dr)

    pt = piece.piece_type
    if pt == chess.KNIGHT:
        return (adf, adr) in ((1, 2), (2, 1))
    if pt == chess.BISHOP:
        return adf == adr and adf != 0
    if pt == chess.ROOK:
        return (df == 0) != (dr == 0)
    if pt == chess.QUEEN:
        return (adf == adr and adf != 0) or ((df == 0) != (dr == 0))
    if pt == chess.KING:
        if max(adf, adr) == 1:
            return True
        # castling: king slides two squares along its own home rank
        home_rank = 0 if piece.color == chess.WHITE else 7
        return dr == 0 and fr == home_rank and adf == 2
    if pt == chess.PAWN:
        forward = 1 if piece.color == chess.WHITE else -1
        start_rank = 1 if piece.color == chess.WHITE else 6
        if df == 0 and dr == forward:
            return True
        if df == 0 and dr == 2 * forward and fr == start_rank:
            return True
        if adf == 1 and dr == forward:
            return True
        return False
    return False


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def draw_board(screen, board_bg, board, images, flipped, selected_square,
               legal_targets, last_move, dragging_square,
               premoves, premove_selected_square):
    screen.blit(board_bg, (BOARD_X, 0))

    def tint(square, color):
        x, y = square_to_pixel(square, flipped)
        overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        overlay.fill(color)
        screen.blit(overlay, (x, y))

    if last_move is not None:
        tint(last_move.from_square, LAST_MOVE_TINT)
        tint(last_move.to_square, LAST_MOVE_TINT)

    # Premove highlighting: chess.com-style -- both squares filled blue,
    # no destination dots, whether still being chosen or already queued.
    # Several premoves can be queued at once; all of them light up.
    for from_sq, to_sq in premoves:
        tint(from_sq, PREMOVE_TINT)
        tint(to_sq, PREMOVE_TINT)
    if premove_selected_square is not None:
        tint(premove_selected_square, PREMOVE_TINT)

    if selected_square is not None:
        tint(selected_square, SELECTED_TINT)

    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            x, y = square_to_pixel(king_sq, flipped)
            glow = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
            pygame.draw.circle(glow, CHECK_TINT, (SQUARE // 2, SQUARE // 2), SQUARE // 2)
            screen.blit(glow, (x, y))

    # pieces (skip the one currently being dragged; drawn on top separately).
    # If premoves are queued, render them optimistically in order: each
    # piece hops to its premove's target square immediately, chess.com-
    # style, purely for display -- the real board underneath is untouched.
    # Once a premove resolves (fires because it turned out legal, so the
    # real board catches up, or gets silently discarded -- along with the
    # rest of the queue -- because it didn't), this same map is simply
    # rebuilt from the unchanged real board next frame, so any premoved
    # pieces visually snap back to their original squares on their own.
    piece_map = compute_premove_piece_map(board, premoves)

    for square in chess.SQUARES:
        if square == dragging_square:
            continue
        piece = piece_map.get(square)
        if piece is None:
            continue
        x, y = square_to_pixel(square, flipped)
        screen.blit(images[(piece.color, piece.piece_type)], (x + PIECE_OFFSET, y + PIECE_OFFSET))

    # legal-move indicators (real moves only -- premoves intentionally show no dots)
    for target in legal_targets:
        x, y = square_to_pixel(target, flipped)
        overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        if board.piece_at(target) is not None or target == board.ep_square and \
                board.piece_type_at(selected_square) == chess.PAWN:
            pygame.draw.circle(overlay, RING_COLOR, (SQUARE // 2, SQUARE // 2), SQUARE // 2 - 4, width=6)
        else:
            pygame.draw.circle(overlay, DOT_COLOR, (SQUARE // 2, SQUARE // 2), SQUARE // 7)
        screen.blit(overlay, (x, y))


def draw_dragging_piece(screen, images, board, premoves, dragging_square, mouse_pos):
    if dragging_square is None:
        return
    piece = compute_premove_piece_map(board, premoves).get(dragging_square)
    if piece is None:
        return
    img = images[(piece.color, piece.piece_type)]
    rect = img.get_rect(center=mouse_pos)
    screen.blit(img, rect)


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines, current = [], ""
    for w in words:
        trial = (current + " " + w).strip()
        if font.size(trial)[0] > max_width and current:
            lines.append(current)
            current = w
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def draw_toggle(screen, rect, is_on):
    track_color = (90, 190, 110) if is_on else (95, 98, 104)
    pygame.draw.rect(screen, track_color, rect, border_radius=rect.height // 2)
    knob_r = rect.height // 2 - 2
    knob_x = rect.right - knob_r - 2 if is_on else rect.left + knob_r + 2
    pygame.draw.circle(screen, (250, 250, 250), (knob_x, rect.centery), knob_r)


def draw_button(screen, font, rect, label, hovered=False):
    color = (90, 96, 108) if hovered else (72, 78, 90)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, (140, 146, 158), rect, width=1, border_radius=8)
    text = font.render(label, True, (240, 240, 240))
    screen.blit(text, text.get_rect(center=rect.center))


def draw_popup(screen, font, big_font, result_text, pgn_copied):
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    screen.blit(overlay, (0, 0))

    pygame.draw.rect(screen, (40, 42, 48), POPUP_RECT, border_radius=14)
    pygame.draw.rect(screen, (100, 105, 115), POPUP_RECT, width=2, border_radius=14)

    title = big_font.render("Game Over", True, (245, 245, 245))
    screen.blit(title, title.get_rect(center=(POPUP_RECT.centerx, POPUP_RECT.top + 44)))

    result_label = font.render(result_text or "", True, (255, 210, 90))
    screen.blit(result_label, result_label.get_rect(center=(POPUP_RECT.centerx, POPUP_RECT.top + 84)))

    mouse_pos = pygame.mouse.get_pos()
    draw_button(screen, font, COPY_PGN_BTN_RECT, "Copy PGN", COPY_PGN_BTN_RECT.collidepoint(mouse_pos))
    draw_button(screen, font, POPUP_ANALYSIS_BTN_RECT, "Analysis",
                POPUP_ANALYSIS_BTN_RECT.collidepoint(mouse_pos))
    draw_button(screen, font, REMATCH_WHITE_BTN_RECT, "Rematch as White",
                REMATCH_WHITE_BTN_RECT.collidepoint(mouse_pos))
    draw_button(screen, font, REMATCH_BLACK_BTN_RECT, "Rematch as Black",
                REMATCH_BLACK_BTN_RECT.collidepoint(mouse_pos))

    if pgn_copied:
        note = font.render("PGN copied to clipboard!", True, (140, 220, 150))
        screen.blit(note, note.get_rect(center=(POPUP_RECT.centerx, COPY_PGN_BTN_RECT.top - 18)))


def draw_sidebar(screen, font, big_font, board, human_is_white, model_name,
                  thinking, san_history, result_text, premoves, show_eval_bar,
                  moves_scroll):
    pygame.draw.rect(screen, SIDEBAR_BG, (SIDEBAR_X, 0, SIDEBAR_W, WINDOW_H))
    pad = 18
    y = pad

    title = big_font.render("Chess vs Model", True, SIDEBAR_TEXT)
    screen.blit(title, (SIDEBAR_X + pad, y))
    y += 40

    you_are = "White" if human_is_white else "Black"
    info_lines = [
        f"You: {you_are}",
        f"Model: {os.path.basename(model_name)}",
    ]
    for line in info_lines:
        screen.blit(font.render(line, True, SIDEBAR_SUBTEXT), (SIDEBAR_X + pad, y))
        y += 24

    y += 10
    if result_text:
        for line in wrap_text(result_text, font, SIDEBAR_W - 2 * pad):
            screen.blit(font.render(line, True, (255, 210, 90)), (SIDEBAR_X + pad, y))
            y += 24
        y += 6
        screen.blit(font.render("Press 'r' for a new game", True, SIDEBAR_SUBTEXT),
                    (SIDEBAR_X + pad, y))
        y += 30
    else:
        turn_str = "White to move" if board.turn == chess.WHITE else "Black to move"
        screen.blit(font.render(turn_str, True, SIDEBAR_TEXT), (SIDEBAR_X + pad, y))
        y += 26
        if thinking:
            screen.blit(font.render("Model is thinking...", True, (120, 200, 255)),
                        (SIDEBAR_X + pad, y))
            y += 26
        if premoves:
            chain = " -> ".join(f"{chess.square_name(f)}-{chess.square_name(t)}" for f, t in premoves)
            for line in wrap_text(f"Premoves queued: {chain}", font, SIDEBAR_W - 2 * pad):
                screen.blit(font.render(line, True, (120, 170, 250)), (SIDEBAR_X + pad, y))
                y += 22
            y += 4

    y += 10
    screen.blit(font.render("Moves:", True, SIDEBAR_TEXT), (SIDEBAR_X + pad, y))
    y += 24
    list_top = y

    # render move history two-per-line ("1. e4 e5")
    move_lines = []
    for i in range(0, len(san_history), 2):
        num = i // 2 + 1
        white_move = san_history[i]
        black_move = san_history[i + 1] if i + 1 < len(san_history) else ""
        move_lines.append(f"{num}. {white_move}  {black_move}")

    line_h = 22
    bottom_reserved = WINDOW_H - EVAL_TOGGLE_RECT.top + 14  # room for the toggle below the list
    max_lines = max(1, (WINDOW_H - y - pad - bottom_reserved) // line_h)
    total = len(move_lines)
    max_scroll = max(0, total - max_lines)
    scroll = max(0, min(moves_scroll, max_scroll))

    # scroll=0 means "pinned to the bottom" (latest move visible); higher
    # scroll reveals older moves further up the list.
    start = max(0, total - max_lines - scroll)
    end = min(total, start + max_lines)

    for line in move_lines[start:end]:
        screen.blit(font.render(line, True, SIDEBAR_SUBTEXT), (SIDEBAR_X + pad, y))
        y += line_h

    if start > 0:
        hint = font.render("^ scroll up for earlier moves", True, SIDEBAR_SUBTEXT)
        screen.blit(hint, (SIDEBAR_X + pad, list_top - 20))
    if end < total:
        hint = font.render("v more moves below", True, SIDEBAR_SUBTEXT)
        screen.blit(hint, (SIDEBAR_X + pad, min(y, EVAL_TOGGLE_RECT.top - 22)))

    # Analysis button, pinned near the bottom-left of the right panel.
    mouse_pos = pygame.mouse.get_pos()
    draw_button(screen, font, ANALYSIS_BTN_RECT, "Analysis", ANALYSIS_BTN_RECT.collidepoint(mouse_pos))

    # Eval sidebar toggle, pinned near the bottom of the right panel.
    eval_label = font.render("Eval Sidebar", True, SIDEBAR_SUBTEXT)
    screen.blit(eval_label, (EVAL_TOGGLE_RECT.left - 10 - eval_label.get_width(),
                              EVAL_TOGGLE_RECT.centery - eval_label.get_height() // 2))
    draw_toggle(screen, EVAL_TOGGLE_RECT, show_eval_bar)

    return max_scroll


# ----------------------------------------------------------------------
# Game analysis overlay -- full-window panel listing, for every move the
# model has made so far, its search's evaluation of the position it was
# choosing from and the top candidate moves it considered (by
# search-visit share), next to the move it actually played.
# ----------------------------------------------------------------------

ANALYSIS_BG = (15, 16, 20, 235)
ANALYSIS_ROW_H = 66


def draw_analysis(screen, font, big_font, move_analysis, scroll):
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill(ANALYSIS_BG)
    screen.blit(overlay, (0, 0))

    title = big_font.render("Game Analysis", True, (245, 245, 245))
    screen.blit(title, (30, 20))

    mouse_pos = pygame.mouse.get_pos()
    close_hovered = ANALYSIS_CLOSE_BTN_RECT.collidepoint(mouse_pos)
    pygame.draw.rect(screen, (90, 95, 105) if close_hovered else (70, 74, 84),
                      ANALYSIS_CLOSE_BTN_RECT, border_radius=6)
    x_label = font.render("X", True, (240, 240, 240))
    screen.blit(x_label, x_label.get_rect(center=ANALYSIS_CLOSE_BTN_RECT.center))

    pad = 30
    y0 = 72

    if not move_analysis:
        empty = font.render(
            "No model moves recorded yet -- they'll appear here as the model plays.",
            True, (190, 190, 195))
        screen.blit(empty, (pad, y0))
        return 0

    subtitle = font.render(
        "Evaluation is from White's perspective (+ favors White). "
        "\"Considered\" lists the model's top candidates by share of search visits.",
        True, SIDEBAR_SUBTEXT)
    screen.blit(subtitle, (pad, y0))
    list_top = y0 + 30

    visible_h = WINDOW_H - list_top - 20
    max_lines = max(1, visible_h // ANALYSIS_ROW_H)
    total = len(move_analysis)
    max_scroll = max(0, total - max_lines)
    scroll = max(0, min(scroll, max_scroll))

    # scroll=0 pins to the latest (most recent) model move, mirroring the
    # move-list sidebar's scroll convention.
    start = max(0, total - max_lines - scroll)
    end = min(total, start + max_lines)

    y = list_top
    for entry in move_analysis[start:end]:
        header = (f"{entry['move_number']}. {entry['color']}: {entry['san']}"
                   f"    eval: {entry['value_white']:+.2f}")
        screen.blit(font.render(header, True, (235, 235, 235)), (pad, y))
        y += 24
        cand_strs = [f"{san} {pct * 100:.0f}% ({cnt})" for san, cnt, pct in entry["candidates"]]
        cand_line = "Considered: " + ", ".join(cand_strs) if cand_strs else "Considered: (no data)"
        for line in wrap_text(cand_line, font, WINDOW_W - 2 * pad)[:1]:
            screen.blit(font.render(line, True, (150, 200, 255)), (pad, y))
            y += 20
        y += ANALYSIS_ROW_H - 44

    if start > 0:
        hint = font.render("^ scroll up for earlier moves", True, (170, 170, 175))
        screen.blit(hint, (pad, list_top - 22))
    if end < total:
        hint = font.render("v scroll down for more moves", True, (170, 170, 175))
        screen.blit(hint, (pad, WINDOW_H - 22))

    return max_scroll


# ----------------------------------------------------------------------
# Eval bar (chess.com-style vertical bar to the left of the board)
# ----------------------------------------------------------------------

def draw_eval_bar(screen, eval_font, eval_white, show, human_is_white):
    """`eval_white` is the model's static value estimate in [-1, 1] from
    White's perspective (+1 = White winning). The bar always fills from
    whichever edge the human's own side occupies, matching the board's
    orientation, so it reads naturally no matter which color you're
    playing."""
    rect = pygame.Rect(0, 0, EVAL_BAR_W, BOARD_PX)
    pygame.draw.rect(screen, EVAL_BAR_BG, rect)

    if not show:
        return

    ev = max(-1.0, min(1.0, eval_white))
    white_fraction = (ev + 1.0) / 2.0  # 0..1

    if human_is_white:
        # human (White) sits at the bottom -> White's share fills upward from the bottom
        white_h = int(BOARD_PX * white_fraction)
        pygame.draw.rect(screen, EVAL_WHITE_COLOR, (0, BOARD_PX - white_h, EVAL_BAR_W, white_h))
    else:
        # human (Black) sits at the bottom -> Black's share fills upward from the bottom
        black_fraction = 1.0 - white_fraction
        black_h = int(BOARD_PX * black_fraction)
        pygame.draw.rect(screen, EVAL_WHITE_COLOR, (0, 0, EVAL_BAR_W, BOARD_PX - black_h))

    text = f"{ev:+.2f}"
    label = eval_font.render(text, True, (225, 225, 225))
    label_bg_y = 6
    bg_rect = pygame.Rect(2, label_bg_y - 2, EVAL_BAR_W - 4, label.get_height() + 4)
    pygame.draw.rect(screen, (0, 0, 0, 120), bg_rect, border_radius=4)
    screen.blit(label, label.get_rect(center=bg_rect.center))


# ----------------------------------------------------------------------
# Promotion picker (small blocking modal loop)
# ----------------------------------------------------------------------

def prompt_promotion(screen, clock, images, color, background_draw_fn):
    box_w, box_h = SQUARE * len(PROMOTION_CHOICES), SQUARE
    box_x = BOARD_X + (BOARD_PX - box_w) // 2
    box_y = (WINDOW_H - box_h) // 2

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if box_y <= my <= box_y + box_h and box_x <= mx <= box_x + box_w:
                    idx = (mx - box_x) // SQUARE
                    if 0 <= idx < len(PROMOTION_CHOICES):
                        return PROMOTION_CHOICES[idx]

        background_draw_fn()

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        pygame.draw.rect(screen, (250, 250, 250), (box_x - 6, box_y - 6, box_w + 12, box_h + 12),
                          border_radius=10)
        for i, pt in enumerate(PROMOTION_CHOICES):
            x = box_x + i * SQUARE
            pygame.draw.rect(screen, (225, 225, 225), (x, box_y, SQUARE, SQUARE))
            screen.blit(images[(color, pt)], (x + PIECE_OFFSET, box_y + PIECE_OFFSET))

        pygame.display.flip()
        clock.tick(60)


# ----------------------------------------------------------------------
# Model-thinking background worker
# ----------------------------------------------------------------------

def model_move_worker(proc, board_snapshot, sims, threads, result_queue):
    """Runs entirely off the main/GUI thread. Only reads `board_snapshot`
    (a chess.Board the GUI thread will not mutate again until the move
    lands), so there is no concurrent-write race with the render loop,
    which only reads the *live* board object (a different one) each frame.

    Also hands back the raw search-visit counts and root value alongside
    the chosen move -- these are exactly what the game-analysis panel
    needs to show how the model evaluated the position and which
    candidates its search preferred, and they're already produced by
    train.search() for free.

    Mate-in-1 safety net: checked before trusting MCTS at all, same as
    self-play/tournament in train.py. Regardless of what search/value
    currently believe, if a legal move mates right now, play it --
    full stop, no dependence on sims/visit convergence. The synthetic
    visits/value below just make the analysis panel show this clearly
    (100% on the mating move, value pinned to a win) rather than lying
    about there having been a real search.
    """
    try:
        mate_move = train.find_immediate_mate(board_snapshot)
        if mate_move is not None:
            best_uci = mate_move.uci()
            result_queue.put(("ok", {"uci": best_uci, "visits": {best_uci: 1}, "value": 1.0}))
            return
        visits, value = train.search(proc, board_snapshot, sims=sims, threads=threads)
        best_uci = train.pick_move_from_visits(visits, temperature=0.0)
        result_queue.put(("ok", {"uci": best_uci, "visits": visits, "value": value}))
    except Exception as e:  # surface engine crashes to the GUI instead of hanging it
        result_queue.put(("error", str(e)))


# ----------------------------------------------------------------------
# Main game loop
# ----------------------------------------------------------------------

class Game:
    def __init__(self, args, screen, font, big_font, eval_font, images, sounds,
                 board_backgrounds, proc):
        self.args = args
        self.screen = screen
        self.font = font
        self.big_font = big_font
        self.eval_font = eval_font
        self.images = images
        self.sounds = sounds
        self.proc = proc

        self.human_is_white = args.color == "white"
        self.flipped = not self.human_is_white
        # Whichever color sits at the bottom of the board picks the
        # matching background art (pre-rendered coordinate labels, etc.).
        self._board_backgrounds = board_backgrounds
        self.board_bg = board_backgrounds["white" if self.human_is_white else "black"]

        self.board = None
        self.selected_square = None
        self.legal_targets = []
        self.dragging_square = None
        self.last_move = None
        self.san_history = []
        self.result_text = None
        self.moves_scroll = 0  # 0 = pinned to the latest move
        self.moves_max_scroll = 0

        # Premoves: an ordered queue of (from_square, to_square) pairs the
        # human staged while it wasn't their turn. One fires per model
        # move, in order -- the instant a model move lands, the first
        # queued premove is tried; if it's still legal it's played
        # automatically, otherwise it (and the rest of the queue, since
        # it was planned assuming the discarded one would land first) is
        # silently dropped, same as chess.com. Deliberately unrestricted
        # while choosing a destination -- any square, including one
        # occupied by another of the player's own pieces (e.g. to queue a
        # recapture before the capture that creates it has even happened),
        # ignoring normal movement/check rules entirely; only the final
        # legality check (against the real position once the model
        # actually moves) matters.
        self.premoves = []
        self.premove_selected_square = None

        # Game analysis: one entry per model move so far this game,
        # recording the search's root evaluation and its top candidate
        # moves by visit share, captured the instant each model move is
        # decided (see record_model_analysis / poll_model_move).
        self.move_analysis = []
        self.show_analysis = False
        self.analysis_scroll = 0
        self.analysis_max_scroll = 0

        # Eval bar: a static model value-head estimate for the current
        # position, from White's perspective, in [-1, 1]. Toggleable via
        # a switch in the sidebar.
        self.show_eval_bar = True
        self.eval_white = 0.0

        self.thinking = False
        self.result_queue = queue.Queue()
        self.worker_thread = None

        self.pgn_copied_at = None  # timestamp (ms) of the last "Copy PGN" click

        self.reset_game()

    def reset_game(self, human_is_white=None):
        """Starts a fresh game. `human_is_white`, if given, switches which
        color the human plays (used by the "Rematch as White/Black"
        popup buttons); omitted (e.g. pressing 'r'), it keeps whatever
        color was just being played."""
        if human_is_white is not None and human_is_white != self.human_is_white:
            self.human_is_white = human_is_white
            self.flipped = not self.human_is_white
            self.board_bg = self._board_backgrounds["white" if self.human_is_white else "black"]

        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.dragging_square = None
        self.last_move = None
        self.san_history = []
        self.result_text = None
        self.moves_scroll = 0
        self.premoves = []
        self.premove_selected_square = None
        self.pgn_copied_at = None
        self.thinking = False
        self.worker_thread = None
        self.move_analysis = []
        self.show_analysis = False
        self.analysis_scroll = 0
        self.analysis_max_scroll = 0
        while not self.result_queue.empty():
            self.result_queue.get_nowait()
        self.compute_eval()
        self.play_sound("game-start")
        if not self.human_is_white:
            self.kick_off_model_move()

    def human_turn(self):
        return (self.board.turn == chess.WHITE) == self.human_is_white

    def legal_moves_from(self, square):
        return [m for m in self.board.legal_moves if m.from_square == square]

    def compute_eval(self):
        """Static (non-search) model value-head estimate for the current
        position, converted to White's perspective for the eval bar. Cheap
        single forward pass -- separate from, and safe to call alongside,
        the model's own MCTS search running on the worker thread."""
        if self.board.is_checkmate():
            loser_is_white = self.board.turn == chess.WHITE
            self.eval_white = -1.0 if loser_is_white else 1.0
            return
        if self.board.is_game_over(claim_draw=True):
            self.eval_white = 0.0
            return
        try:
            _, _, value = train.nn_eval(self.board, is_root=False)
        except Exception:
            return  # leave the previous eval on-screen rather than crash
        self.eval_white = value if self.board.turn == chess.WHITE else -value

    def play_sound(self, key):
        snd = self.sounds.get(key)
        if snd is not None:
            snd.play()

    # ---------------- move execution ----------------

    def try_move(self, from_sq, to_sq):
        """Attempts an immediate move for the human. Plays the illegal-move
        sound and refuses if it isn't actually legal."""
        candidates = [m for m in self.legal_moves_from(from_sq) if m.to_square == to_sq]
        if not candidates:
            self.play_sound("illegal")
            return False
        if len(candidates) == 1:
            move = candidates[0]
        else:
            # multiple candidates only happens for underpromotion choices
            piece = self.board.piece_at(from_sq)
            chosen_pt = prompt_promotion(
                self.screen, pygame.time.Clock(), self.images, piece.color,
                background_draw_fn=self.draw_frame,
            )
            move = next(m for m in candidates if m.promotion == chosen_pt)

        self.push_move(move, mover_is_human=True)
        return True

    def push_move(self, move, mover_is_human):
        """Applies `move` to the live board, updates history/highlights, and
        plays the appropriate chess.com-style sound for what just happened."""
        is_capture = self.board.is_capture(move)
        is_castle = self.board.is_castling(move)
        is_promo = move.promotion is not None

        san = self.board.san(move)
        self.board.push(move)
        self.san_history.append(san)
        self.last_move = move
        self.moves_scroll = 0  # auto-follow the latest move
        self.compute_eval()
        is_checkmate = self.board.is_checkmate()
        is_check = self.board.is_check()
        game_over_now = self.board.is_game_over(claim_draw=True)

        if is_checkmate:
            self.play_sound("move-check")
            self.play_sound("game-end")
        elif is_check:
            self.play_sound("move-check")
        elif is_promo:
            self.play_sound("promote")
        elif is_capture:
            self.play_sound("capture")
        elif is_castle:
            self.play_sound("castle")
        else:
            self.play_sound("move-self" if mover_is_human else "move-opponent")

        if game_over_now and not is_checkmate:
            self.play_sound("game-end")

        self.check_game_over()

        if mover_is_human:
            if not self.result_text:
                self.kick_off_model_move()
        else:
            # The model just moved -- it's now genuinely the human's turn,
            # so any in-progress (not-yet-finalized) premove selection is
            # stale and shouldn't keep showing. If a premove is queued,
            # try to fire the next one now.
            self.premove_selected_square = None
            if not self.result_text and self.premoves:
                self.try_execute_premove()

    def scroll_moves(self, direction):
        self.moves_scroll = max(0, min(self.moves_scroll + direction, self.moves_max_scroll))

    def kick_off_model_move(self):
        self.thinking = True
        board_snapshot = self.board.copy(stack=True)
        self.worker_thread = threading.Thread(
            target=model_move_worker,
            args=(self.proc, board_snapshot, self.args.sims, self.args.threads, self.result_queue),
            daemon=True,
        )
        self.worker_thread.start()

    def poll_model_move(self):
        if not self.thinking:
            return
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            return
        self.thinking = False
        if status == "error":
            self.result_text = f"Engine error: {payload}"
            return
        move = chess.Move.from_uci(payload["uci"])
        # Must run before push_move: it needs self.board (and self.board.san)
        # in the pre-move position to label the chosen move and its
        # candidates, and push_move mutates self.board in place.
        self.record_model_analysis(move, payload["visits"], payload["value"])
        self.push_move(move, mover_is_human=False)

    def record_model_analysis(self, move, visits, value):
        """Captures how the model's search evaluated the position it just
        moved from: the root value estimate (converted to White's
        perspective, matching the eval bar) and its top few candidate
        moves by share of search visits. Called right before the move is
        applied, so `self.board` is still the position being analyzed."""
        mover_is_white = self.board.turn == chess.WHITE
        total_visits = sum(visits.values()) or 1
        top_candidates = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)[:3]
        candidates = []
        for uci, count in top_candidates:
            try:
                san = self.board.san(chess.Move.from_uci(uci))
            except Exception:
                san = uci
            candidates.append((san, count, count / total_visits))

        self.move_analysis.append({
            "move_number": len(self.san_history) // 2 + 1,
            "color": "White" if mover_is_white else "Black",
            "san": self.board.san(move),
            "value_white": value if mover_is_white else -value,
            "candidates": candidates,
        })
        self.analysis_scroll = 0  # auto-follow the latest analyzed move

    def check_game_over(self):
        if self.board.is_game_over(claim_draw=True):
            outcome = self.board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                self.result_text = "Draw."
            else:
                human_won = (outcome.winner == chess.WHITE) == self.human_is_white
                self.result_text = "You won!" if human_won else "The model won."

    # ---------------- premove ----------------

    def handle_premove_mousedown(self, sq):
        # Use the optimistic (post-premove-chain) piece placement, not the
        # real board, so a piece that's already been virtually moved by an
        # earlier queued premove -- and so only "lives" on its destination
        # square in the optimistic view -- can still be picked up here to
        # queue the next link of the chain (e.g. premove e2-e4, then,
        # without waiting for it to land, premove that same pawn e4-e5).
        piece_at_sq = compute_premove_piece_map(self.board, self.premoves).get(sq)
        is_own_piece = piece_at_sq is not None and \
            (piece_at_sq.color == chess.WHITE) == self.human_is_white

        if self.premove_selected_square is not None:
            if sq == self.premove_selected_square:
                # re-clicking the already-chosen piece cancels the selection
                self.premove_selected_square = None
                self.dragging_square = None
                return
            # click-click finalize (mirrors the human-turn move flow below).
            # Deliberately allowed to target ANY square -- including one
            # with one of the player's own pieces (e.g. queuing a
            # recapture) or the start square of another already-queued
            # premove -- only the real legality check once it's about to
            # fire (see try_execute_premove) decides whether it plays.
            # This check MUST come before the "cancel a queued premove"
            # check below: otherwise finalizing a second/third premove
            # whose destination happens to land on an earlier queued
            # premove's start square would silently cancel that one
            # instead of queuing the new one.
            self.finalize_premove(self.premove_selected_square, sq)
            self.dragging_square = None
            return

        # No selection in progress -- clicking the start square of an
        # already-queued premove cancels just that one (leaving the rest
        # of the queue intact).
        for i, (from_sq, _) in enumerate(self.premoves):
            if sq == from_sq:
                del self.premoves[i]
                return

        if is_own_piece:
            # start a new selection, and also arm dragging so the piece can
            # be dragged straight to its destination instead of click-click
            self.premove_selected_square = sq
            self.dragging_square = sq
        else:
            self.premove_selected_square = None
            self.dragging_square = None

    def finalize_premove(self, from_sq, to_sq):
        """Appends (from_sq, to_sq) to the premove queue, provided the
        destination at least matches the piece's own movement pattern
        (see premove_shape_ok) -- so a pawn still can't be premoved three
        squares, or a knight moved diagonally like a bishop, etc. Beyond
        that it's deliberately unrestricted: any square, including one
        occupied by another of the player's own pieces, ignoring
        occupancy/check/whose-turn-it-is otherwise -- only the real
        legality check once it's next in line to fire (see
        try_execute_premove) decides whether it actually plays."""
        if not premove_shape_ok(self.board, self.premoves, from_sq, to_sq):
            self.play_sound("illegal")
            self.premove_selected_square = None
            return
        self.premoves.append((from_sq, to_sq))
        self.premove_selected_square = None
        self.play_sound("premove")

    def try_execute_premove(self):
        if not self.premoves:
            return
        from_sq, to_sq = self.premoves.pop(0)
        candidates = [m for m in self.legal_moves_from(from_sq) if m.to_square == to_sq]
        if not candidates:
            # No longer legal -- silently discarded, like chess.com. The
            # rest of the queue was planned assuming this move would land
            # first, so it's stale too; drop it rather than firing moves
            # in an order/position the player never actually queued for.
            self.premoves = []
            return
        if len(candidates) == 1:
            move = candidates[0]
        else:
            # underpromotion choices: premoves default to queen
            move = next((m for m in candidates if m.promotion == chess.QUEEN), candidates[0])
        self.push_move(move, mover_is_human=True)

    # ---------------- end-of-game popup ----------------

    def handle_popup_click(self, pos):
        if COPY_PGN_BTN_RECT.collidepoint(pos):
            pgn = build_pgn(self.board, self.human_is_white, self.args.model)
            copy_to_clipboard(pgn)
            self.pgn_copied_at = pygame.time.get_ticks()
        elif POPUP_ANALYSIS_BTN_RECT.collidepoint(pos):
            self.show_analysis = True
        elif REMATCH_WHITE_BTN_RECT.collidepoint(pos):
            self.reset_game(human_is_white=True)
        elif REMATCH_BLACK_BTN_RECT.collidepoint(pos):
            self.reset_game(human_is_white=False)

    # ---------------- game analysis overlay ----------------

    def handle_analysis_click(self, pos):
        if ANALYSIS_CLOSE_BTN_RECT.collidepoint(pos):
            self.show_analysis = False

    def scroll_analysis(self, direction):
        self.analysis_scroll = max(0, min(self.analysis_scroll + direction, self.analysis_max_scroll))

    # ---------------- events ----------------

    def handle_mousedown(self, pos):
        if self.show_analysis:
            self.handle_analysis_click(pos)
            return

        if ANALYSIS_BTN_RECT.collidepoint(pos):
            self.show_analysis = True
            return

        if self.result_text is not None:
            self.handle_popup_click(pos)
            return

        if EVAL_TOGGLE_RECT.collidepoint(pos):
            self.show_eval_bar = not self.show_eval_bar
            return

        sq = pixel_to_square(pos, self.flipped)
        if sq is None:
            return

        if not self.human_turn():
            self.handle_premove_mousedown(sq)
            return

        # Clicking an already-highlighted destination executes the move.
        if self.selected_square is not None and sq in self.legal_targets:
            self.try_move(self.selected_square, sq)
            self.selected_square = None
            self.legal_targets = []
            return

        # Clicking the currently-selected piece again unselects it.
        if sq == self.selected_square:
            self.selected_square = None
            self.legal_targets = []
            self.dragging_square = None
            return

        piece = self.board.piece_at(sq)
        is_own_piece = piece is not None and (piece.color == chess.WHITE) == self.human_is_white
        if is_own_piece:
            self.selected_square = sq
            self.legal_targets = [m.to_square for m in self.legal_moves_from(sq)]
            self.dragging_square = sq
        else:
            # Had something selected and clicked a square that isn't a legal
            # destination and isn't another of the player's own pieces --
            # an attempted illegal move.
            if self.selected_square is not None:
                self.play_sound("illegal")
            self.selected_square = None
            self.legal_targets = []

    def handle_mouseup(self, pos):
        if self.dragging_square is None:
            return
        drag_square = self.dragging_square
        self.dragging_square = None
        if self.result_text is not None:
            return

        sq = pixel_to_square(pos, self.flipped)

        if self.human_turn():
            # It's genuinely the human's turn right now -- true even if it
            # WASN'T when this drag started (the model's move can land,
            # and any queued premove auto-fire with it, while a premove
            # drag is still in progress). Recompute everything fresh
            # against the live board rather than trusting self.legal_targets,
            # which premove-mode mousedown never populates (it would
            # incorrectly look empty) and which could otherwise be stale.
            piece = self.board.piece_at(drag_square)
            is_own_piece = piece is not None and (piece.color == chess.WHITE) == self.human_is_white
            if not is_own_piece:
                # The piece we picked up is no longer ours to move from
                # here -- e.g. the model's move captured it mid-drag.
                # Nothing sane to do but drop the attempt quietly.
                self.selected_square = None
                self.legal_targets = []
                self.premove_selected_square = None
                return

            legal_targets = [m.to_square for m in self.legal_moves_from(drag_square)]
            if sq == drag_square:
                # released right back where it was picked up -- treat as
                # a plain click, keep it selected so the next click can move it
                self.selected_square = drag_square
                self.legal_targets = legal_targets
                return
            if sq is not None and sq in legal_targets:
                self.try_move(drag_square, sq)
            else:
                # dropped on a square that isn't a legal destination
                self.play_sound("illegal")
            self.selected_square = None
            self.legal_targets = []
            return

        # Premove drag-and-drop: still genuinely not the human's turn.
        # Dropping on a square occupied by another of the player's own
        # pieces is allowed too -- it queues a premove there anyway (e.g.
        # a recapture queued before the opponent's capture that creates
        # it has landed); only the real legality check when it's about to
        # fire (see try_execute_premove) decides whether it actually plays.
        if sq is None or sq == drag_square:
            # released back where it was picked up -- plain click,
            # selection already set by handle_premove_mousedown
            return
        self.finalize_premove(drag_square, sq)

    # ---------------- drawing ----------------

    def draw_frame(self):
        self.screen.fill((0, 0, 0))
        draw_eval_bar(self.screen, self.eval_font, self.eval_white, self.show_eval_bar,
                      self.human_is_white)
        draw_board(self.screen, self.board_bg, self.board, self.images, self.flipped,
                   self.selected_square, self.legal_targets, self.last_move,
                   self.dragging_square, self.premoves, self.premove_selected_square)
        draw_dragging_piece(self.screen, self.images, self.board, self.premoves,
                             self.dragging_square, pygame.mouse.get_pos())
        self.moves_max_scroll = draw_sidebar(
            self.screen, self.font, self.big_font, self.board, self.human_is_white,
            self.args.model, self.thinking, self.san_history, self.result_text,
            self.premoves, self.show_eval_bar, self.moves_scroll)

        if self.result_text is not None:
            pgn_copied = self.pgn_copied_at is not None and \
                pygame.time.get_ticks() - self.pgn_copied_at < 2000
            draw_popup(self.screen, self.font, self.big_font, self.result_text, pgn_copied)

        if self.show_analysis:
            self.analysis_max_scroll = draw_analysis(
                self.screen, self.font, self.big_font, self.move_analysis, self.analysis_scroll)


def main():
    parser = argparse.ArgumentParser(description="Play chess against a trained model (GUI).")
    parser.add_argument("--model", type=str, default=train.BEST_MODEL_PATH,
                         help="Path to model weights (.pt). Defaults to best_model.pt.")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"],
                         help="Which side you play.")
    parser.add_argument("--sims", type=int, default=400,
                         help="MCTS simulations per move for the model (more = stronger but slower).")
    parser.add_argument("--threads", type=int, default=4,
                         help="Worker threads inside mcts_engine per search call.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--assets", type=str, default="assets",
                         help="Folder containing piece images and board backgrounds "
                              "(e.g. assets/black_king.png, assets/checkboard_white.png).")
    parser.add_argument("--sounds", type=str, default="sound",
                         help="Folder containing sound effect .mp3 files (e.g. sound/capture.mp3).")
    args = parser.parse_args()

    train.setup_logging()
    device = torch.device(args.device)

    model = train.DualHeadResNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except FileNotFoundError:
        print(f"Could not find model weights at '{args.model}'.")
        print("Train one first with train.py, or point --model at an existing checkpoint.")
        sys.exit(1)
    model.eval()

    train.compile_engine()
    proc = train.start_engine()

    # Deterministic, strongest-move play: no Dirichlet exploration noise,
    # no visit-count temperature -- always the model's best guess.
    train.SELF_PLAY_MODE = False
    train.CURRENT_MODEL = model
    train.CURRENT_DEVICE = device

    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Chess vs Model")
    # RESIZABLE lets the user drag the window to any size; SCALED keeps
    # all of the drawing code above working at its fixed logical
    # resolution (WINDOW_W x WINDOW_H) while pygame automatically scales
    # that to whatever the actual window/display size is -- including
    # fullscreen (toggled below with F11) -- and automatically translates
    # mouse event coordinates back into logical space, so none of the
    # click hit-testing elsewhere needs to change.
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE | pygame.SCALED)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)
    big_font = pygame.font.SysFont("arial", 26, bold=True)
    eval_font = pygame.font.SysFont("arial", 14, bold=True)

    images = load_piece_images(args.assets)
    sounds = load_sounds(args.sounds)
    board_backgrounds = load_board_backgrounds(args.assets)

    game = Game(args, screen, font, big_font, eval_font, images, sounds, board_backgrounds, proc)

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    game.handle_mousedown(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    game.handle_mouseup(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    if game.show_analysis:
                        game.scroll_analysis(event.y)
                    elif pygame.mouse.get_pos()[0] >= SIDEBAR_X:
                        game.scroll_moves(event.y)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r and game.result_text is not None:
                        game.reset_game()
                    elif event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                    elif event.key == pygame.K_a:
                        game.show_analysis = not game.show_analysis
                    elif event.key == pygame.K_ESCAPE and game.show_analysis:
                        game.show_analysis = False

            game.poll_model_move()
            game.draw_frame()
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
        train.shutdown_engine(proc)


if __name__ == "__main__":
    main()
