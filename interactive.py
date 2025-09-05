#!/usr/bin/env python3
"""
interactive.py

Like compilation.py but interactive: instead of updating every time step,
the canvas updates once per *mouse click*. Each click updates the clicked grid cell by advancing to the next image's piece (cycling through all input images), then overlays it onto the canvas.

Usage:
  python interactive.py img1.jpg img2.jpg ... [--rows 5 --cols 4]

Controls:
  - Left click anywhere in the window: perform one random update
  - 'r' : reset canvas to black
  - 'q' or ESC : quit
"""

import cv2
import numpy as np
import argparse
import sys
from ctypes import cdll, c_uint32
from ctypes import util as ctypes_util

def _screen_size_via_tk():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively reveal a mosaic by placing random grid pieces from input images on click.")
    parser.add_argument('images', nargs='+', help="List of input image paths (all images must be the same size).")
    parser.add_argument('--rows', type=int, default=5, help="Grid rows (default: 5).")
    parser.add_argument('--cols', type=int, default=4, help="Grid cols (default: 4).")
    parser.add_argument('--window', type=str, default='sleeping', help="Window name (default: 'sleeping').")
    parser.add_argument('--debug', action='store_true', help="Enable verbose debug logging to stdout.")
    return parser.parse_args()

def load_images(image_files):
    imgs = []
    print("Input images (WxH):")
    for file in image_files:
        img = cv2.imread(file)
        if img is None:
            print(f"Error: Unable to load image '{file}'", file=sys.stderr)
            sys.exit(1)
        h, w = img.shape[:2]
        print(f"  - {file}: {w}x{h}")
        imgs.append(img)
    return imgs

def resize_to_smallest(images):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_h = min(heights)
    target_w = min(widths)
    if any((h, w) != (target_h, target_w) for h, w in zip(heights, widths)):
        print(f"Resizing images to common size {target_w}x{target_h}")
        images = [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA) for img in images]
    return images

def make_edges(total, parts):
    """Return a list of length parts+1 with cell boundary edges (handles remainder in last cell)."""
    base = total // parts
    edges = [i * base for i in range(parts)]
    edges.append(total)
    return edges  # len = parts + 1

def split_images_to_grid(images, rows=5, cols=4):
    """
    Splits each image into grid pieces.
    Returns:
      grid_pieces: dict[(i,j)] -> [piece_from_img0, piece_from_img1, ...]
      row_edges, col_edges: cell boundary arrays for coordinate math
    """
    grid_pieces = {}
    h, w = images[0].shape[:2]
    row_edges = make_edges(h, rows)
    col_edges = make_edges(w, cols)

    for i in range(rows):
        for j in range(cols):
            y0, y1 = row_edges[i], row_edges[i+1]
            x0, x1 = col_edges[j], col_edges[j+1]
            pieces = [img[y0:y1, x0:x1].copy() for img in images]
            grid_pieces[(i, j)] = pieces

    return grid_pieces, row_edges, col_edges

def index_from_edges(edges, coord):
    """Return index i such that edges[i] <= coord < edges[i+1]; clamps to valid range."""
    for i in range(len(edges) - 1):
        if edges[i] <= coord < edges[i + 1]:
            return i
    return len(edges) - 2

def main():
    # Ensure stdout is line-buffered so prints appear promptly
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    debug = args.debug
    def dprint(*a):
        if debug:
            print(*a, flush=True)
    images = load_images(args.images)
    images = resize_to_smallest(images)
    # Report the unified working resolution after any resizing
    wh, ww = images[0].shape[:2]
    print(f"Working resolution (WxH): {ww}x{wh}")
    dprint(f"OpenCV version: {cv2.__version__}; platform: {sys.platform}")

    rows, cols = args.rows, args.cols
    grid_pieces, row_edges, col_edges = split_images_to_grid(images, rows=rows, cols=cols)

    # Blank canvas same size as inputs
    canvas = np.zeros_like(images[0])

    # Track which source index is currently used for each cell (-1 = black)
    current_idx = {(i, j): -1 for i in range(rows) for j in range(cols)}

    # Create window and attempt fullscreen; we will always scale to window size
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    ih, iw = images[0].shape[:2]
    # Determine fullscreen target size: prefer CoreGraphics (true pixels) on macOS,
    # then Tk (scaled for Retina), then window rect, then image size.
    def _screen_target_size():
        if sys.platform == 'darwin':
            # Try ApplicationServices (CoreGraphics) for real pixel dimensions
            try:
                app_path = ctypes_util.find_library('ApplicationServices')
                if app_path:
                    app = cdll.LoadLibrary(app_path)
                    app.CGMainDisplayID.restype = c_uint32
                    did = app.CGMainDisplayID()
                    app.CGDisplayPixelsWide.argtypes = [c_uint32]
                    app.CGDisplayPixelsWide.restype = c_uint32
                    app.CGDisplayPixelsHigh.argtypes = [c_uint32]
                    app.CGDisplayPixelsHigh.restype = c_uint32
                    w = int(app.CGDisplayPixelsWide(did))
                    h = int(app.CGDisplayPixelsHigh(did))
                    if w > 0 and h > 0:
                        return (w, h)
            except Exception:
                pass
        tk_size = _screen_size_via_tk()
        if tk_size:
            # Tk often returns logical points (e.g., 1440x900 on a 2x Retina 2880x1800)
            if sys.platform == 'darwin':
                tw, th = tk_size
                return (tw * 2, th * 2)
            return tk_size
        try:
            _, _, _w, _h = cv2.getWindowImageRect(args.window)
            if _w > 0 and _h > 0:
                return (int(_w), int(_h))
        except Exception:
            pass
        return (iw, ih)
    fs_size = _screen_target_size()
    fs_w, fs_h = fs_size
    # Set fullscreen with robust fallback
    try:
        fs_desktop = getattr(cv2, 'WINDOW_FULLSCREEN_DESKTOP', None)
        if fs_desktop is not None:
            cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, fs_desktop)
        else:
            cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception as e:
        if debug:
            print(f"Fullscreen set error: {e}", flush=True)
    try:
        fs_prop = cv2.getWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN)
        dprint(f"Initial fullscreen property: {fs_prop}; target size: {fs_w}x{fs_h}")
    except Exception as e:
        dprint(f"Could not read fullscreen property initially: {e}")
    # Always return the cached fullscreen size for drawing
    def draw_size():
        return fs_w, fs_h

    # Click flag shared with the mouse callback
    click_state = {"clicked": False, "x": None, "y": None}

    def on_mouse(event, x, y, flags, param):
        # Only act on left-button clicks; ignore right/middle clicks and drags
        if event == cv2.EVENT_LBUTTONDOWN:
            param["clicked"] = True
            param["x"] = x
            param["y"] = y

    cv2.setMouseCallback(args.window, on_mouse, click_state)

    print("Interactive mode:")
    print("  • Left click: update the CLICKED grid cell to the NEXT image (cycles 1→2→…→N→1)")
    print("  • 'r' to reset to a black canvas")
    print("  • 'q' or ESC to quit")

    try:
        frame = 0
        last_rect = None
        last_fs_prop = None
        while True:
            # Pump events first so window size is up-to-date
            key = cv2.waitKey(16) & 0xFF
            # Ensure fullscreen stays active and draw to the fullscreen target size
            try:
                fs_desktop = getattr(cv2, 'WINDOW_FULLSCREEN_DESKTOP', None)
                if fs_desktop is not None:
                    cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, fs_desktop)
                else:
                    cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception as e:
                dprint(f"setWindowProperty fullscreen error: {e}")
            # Observe window rect and fullscreen property (debug)
            rect_now = None
            try:
                rect_now = cv2.getWindowImageRect(args.window)
            except Exception as e:
                dprint(f"getWindowImageRect error: {e}")
            fs_prop_now = None
            try:
                fs_prop_now = cv2.getWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN)
            except Exception as e:
                dprint(f"getWindowProperty(WND_PROP_FULLSCREEN) error: {e}")
            # Compute draw size and render
            win_w, win_h = draw_size()
            display = cv2.resize(canvas, (win_w, win_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(args.window, display)
            # Log on interesting frames or changes
            if (frame < 10) or (rect_now != last_rect) or (fs_prop_now != last_fs_prop):
                dprint(f"frame={frame} rect={rect_now} fs_prop={fs_prop_now} draw={win_w}x{win_h} canvas={iw}x{ih}")
                last_rect = rect_now
                last_fs_prop = fs_prop_now
            frame += 1

            # Process one update on the clicked grid cell
            if click_state["clicked"]:
                cx, cy = click_state["x"], click_state["y"]
                # Map using the just-drawn frame's size
                dw, dh = display.shape[1], display.shape[0]
                # Clamp and scale
                ox = min(max(int(round(cx / max(dw, 1) * iw)), 0), iw - 1)
                oy = min(max(int(round(cy / max(dh, 1) * ih)), 0), ih - 1)
                # Map pixel coordinates to grid indices
                i = index_from_edges(row_edges, oy)
                j = index_from_edges(col_edges, ox)
                cell = (i, j)
                pieces = grid_pieces[cell]
                dprint(f"click: win=({cx},{cy}) draw={dw}x{dh} -> img=({ox},{oy}) cell=({i},{j}) idx_prev={current_idx[cell]}")

                # Cycle forward through images for this cell (wrap at the end)
                idx = (current_idx[cell] + 1) % len(pieces)

                piece = pieces[idx]
                y0, y1 = row_edges[i], row_edges[i+1]
                x0, x1 = col_edges[j], col_edges[j+1]
                canvas[y0:y1, x0:x1] = piece
                current_idx[cell] = idx

                # consume the click
                click_state["clicked"] = False
                click_state["x"], click_state["y"] = None, None

            # Handle keys
            if key in (ord('q'), 27):  # ESC or 'q'
                dprint("quit requested")
                break
            elif key == ord('r'):
                canvas[:] = 0
                for k in current_idx:
                    current_idx[k] = -1
                dprint("canvas reset")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
