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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively reveal a mosaic by placing random grid pieces from input images on click.")
    parser.add_argument('images', nargs='+', help="List of input image paths (all images must be the same size).")
    parser.add_argument('--rows', type=int, default=5, help="Grid rows (default: 5).")
    parser.add_argument('--cols', type=int, default=4, help="Grid cols (default: 4).")
    parser.add_argument('--window', type=str, default='sleeping', help="Window name (default: 'sleeping').")
    parser.add_argument('--stretch-px', type=int, default=None,
                        help="Extra pixels to add to display height (default: 10% of image height).")
    parser.add_argument('--bake-stretch', action='store_true',
                        help="Bake the vertical stretch into the canvas instead of scaling at display time.")
    return parser.parse_args()

def load_images(image_files):
    imgs = []
    for file in image_files:
        img = cv2.imread(file)
        if img is None:
            print(f"Error: Unable to load image '{file}'", file=sys.stderr)
            sys.exit(1)
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
    args = parse_args()
    images = load_images(args.images)
    images = resize_to_smallest(images)

    rows, cols = args.rows, args.cols
    grid_pieces, row_edges, col_edges = split_images_to_grid(images, rows=rows, cols=cols)

    # Blank canvas same size as inputs
    canvas = np.zeros_like(images[0])
    ih, iw = images[0].shape[:2]
    extra_px = args.stretch_px if args.stretch_px is not None else int(round(ih * 0.10))
    stretch_y = (ih + extra_px) / ih  # vertical stretch factor for display only

    # Track which source index is currently used for each cell (-1 = black)
    current_idx = {(i, j): -1 for i in range(rows) for j in range(cols)}

    # Create fixed-size autosized window; we will render a vertically stretched view
    cv2.namedWindow(args.window, cv2.WINDOW_AUTOSIZE)
    # If baking stretch, keep a separate stretched canvas to draw 1:1
    if args.bake_stretch:
        canvas_disp = np.zeros((ih + extra_px, iw, 3), dtype=canvas.dtype)
    else:
        canvas_disp = None

    # Click flag shared with the mouse callback
    click_state = {"clicked": False, "x": None, "y": None}

    def on_mouse(event, x, y, flags, param):
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
        while True:
            if args.bake_stretch:
                # Show baked stretched canvas 1:1 (no per-frame rescale)
                cv2.imshow(args.window, canvas_disp)
                out_h = ih + extra_px
                display = canvas_disp  # for click mapping dimensions
            else:
                # Render a taller view by a fixed extra_px (width unchanged)
                out_h = ih + extra_px
                display = cv2.resize(canvas, (iw, out_h), interpolation=cv2.INTER_LANCZOS4)
                cv2.imshow(args.window, display)

            # Process one update on the clicked grid cell
            if click_state["clicked"]:
                cx, cy = click_state["x"], click_state["y"]
                # Map pixel coordinates to grid indices
                # Invert the 10% vertical stretch for coordinate mapping
                oy = int(round(cy / stretch_y))
                ox = cx
                # Clamp to original bounds
                oy = max(0, min(ih - 1, oy))
                ox = max(0, min(iw - 1, ox))
                i = index_from_edges(row_edges, oy)
                j = index_from_edges(col_edges, ox)
                cell = (i, j)
                pieces = grid_pieces[cell]

                # Cycle forward through images for this cell (wrap at the end)
                idx = (current_idx[cell] + 1) % len(pieces)

                piece = pieces[idx]
                y0, y1 = row_edges[i], row_edges[i+1]
                x0, x1 = col_edges[j], col_edges[j+1]
                # Update base canvas
                canvas[y0:y1, x0:x1] = piece
                # If baking, also write the stretched piece into the stretched canvas
                if args.bake_stretch:
                    y0s = int(round(y0 * stretch_y))
                    y1s = int(round(y1 * stretch_y))
                    target_h = max(1, y1s - y0s)
                    target_w = max(1, x1 - x0)
                    piece_stretched = cv2.resize(piece, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    canvas_disp[y0s:y1s, x0:x1] = piece_stretched
                current_idx[cell] = idx

                # consume the click
                click_state["clicked"] = False
                click_state["x"], click_state["y"] = None, None

            # Handle keys
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):  # ESC or 'q'
                break
            elif key == ord('r'):
                canvas[:] = 0
                if args.bake_stretch:
                    canvas_disp[:] = 0
                for k in current_idx:
                    current_idx[k] = -1

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
