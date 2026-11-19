#!/usr/bin/env python3
"""Build the level assets for mini-game.html from the unslopped level sources.

Usage:
  python3 tools/make-level-assets.py [--src DIR] [--out-dir DIR] [--mid-darken 0.3]

Reads assets/level/src/level-tiles-unslop.png, level-bg-far-unslop.png and level-bg-mid-unslop.png.
Writes to assets/level/: level-tiles.png, level-tiles.json, level-bg-far.png, level-bg-mid.png and
level-assets.manifest.js (window.LEVEL_ASSETS, for file:// pages).
"""
import argparse
import json
import os
from collections import deque

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TILES_NATIVE = (768, 516)
TILES_ART = (449, 301)  # same pixel density as the character sprites

# Native bounding boxes (W, H, X, Y) on level-tiles-unslop.png.
PIECES = {
    "ground": (747, 99, 11, 401),
    "wall": (119, 354, 26, 21),
    "stonePlatform": (207, 192, 186, 27),
    "catwalkLeft": (108, 84, 175, 273),
    "catwalkMid": (328, 65, 291, 273),
    "catwalkRight": (108, 84, 627, 273),
    "pipeValve": (165, 65, 425, 39),
    "drainGrate": (106, 90, 613, 30),
    "barrel": (68, 102, 427, 137),
    "chainHook": (27, 81, 505, 160),
    "bones": (110, 53, 546, 183),
    "crate": (77, 71, 664, 165),
    "drip1": (19, 43, 525, 120),
    "drip2": (17, 43, 561, 120),
    "drip3": (23, 37, 597, 120),
}
# Native px trimmed from each end of a repeat tile, to drop its closed outline.
TRIM = {"ground": 4, "catwalkMid": 5}
# Native y on the source sheet where the wall pipe elbow starts; the wall fill uses the rows above it.
WALL_ELBOW_Y = 262


def components(alpha, w, h):
    seen = bytearray(w * h)
    comps = []
    for start in range(w * h):
        if seen[start] or not alpha[start]:
            continue
        seen[start] = 1
        q = deque([start])
        x0 = x1 = start % w
        y0 = y1 = start // w
        n = 0
        while q:
            i = q.popleft()
            n += 1
            x, y = i % w, i // w
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            for j in (i - 1 if x > 0 else -1, i + 1 if x < w - 1 else -1, i - w, i + w):
                if 0 <= j < w * h and not seen[j] and alpha[j]:
                    seen[j] = 1
                    q.append(j)
        comps.append((x0, y0, x1 + 1, y1 + 1, n))
    return comps


def surface_row(alpha, w, rect, ratio=0.7):
    x, y, rw, rh = rect
    for yy in range(rh):
        n = sum(1 for xx in range(rw) if alpha[(y + yy) * w + x + xx])
        if n >= ratio * rw:
            return yy
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "assets/level/src"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets/level"))
    ap.add_argument("--mid-darken", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    src = Image.open(os.path.join(args.src, "level-tiles-unslop.png")).convert("RGBA")
    assert src.size == TILES_NATIVE, src.size
    tiles = src.resize(TILES_ART, Image.NEAREST)
    tw, th = tiles.size
    alpha = [1 if v > 127 else 0 for v in tiles.split()[3].getdata()]
    comps = components(alpha, tw, th)
    sx, sy = TILES_ART[0] / TILES_NATIVE[0], TILES_ART[1] / TILES_NATIVE[1]

    rects = {}
    report = []
    for name, (nw, nh, nx, ny) in PIECES.items():
        bx0, by0, bx1, by1 = nx * sx, ny * sy, (nx + nw) * sx, (ny + nh) * sy
        inside = [c for c in comps if bx0 - 1 <= (c[0] + c[2]) / 2 <= bx1 + 1 and by0 - 1 <= (c[1] + c[3]) / 2 <= by1 + 1]
        if not inside:
            raise SystemExit(f"no pixels for {name}")
        x0, y0 = min(c[0] for c in inside), min(c[1] for c in inside)
        x1, y1 = max(c[2] for c in inside), max(c[3] for c in inside)
        rects[name] = [x0, y0, x1 - x0, y1 - y0]
        report.append(f"{name}: scaled {round(bx0)},{round(by0)} {round(bx1 - bx0)}x{round(by1 - by0)} -> cc {x0},{y0} {x1 - x0}x{y1 - y0} ({len(inside)} parts)")

    meta = {}
    for name, nat in TRIM.items():
        x, y, w, h = rects[name]
        t = max(1, round(nat * sx))
        rects[name + "Tile"] = [x + t, y, w - 2 * t, h]
    for name in ("ground", "stonePlatform", "catwalkLeft", "catwalkMid", "catwalkRight"):
        meta[name] = {"surfaceY": surface_row(alpha, tw, rects[name])}
    x, y, w, h = rects["wall"]
    upper = round(WALL_ELBOW_Y * sy) - y
    rects["wallUpper"] = [x, y, w, upper]
    # Chain-only rows on top of the stone platform, used to extend the chains up to the ceiling.
    sx_, sy_, sw, sh = rects["stonePlatform"]
    chain_rows = 0
    for yy in range(sh):
        n = sum(1 for xx in range(sw) if alpha[(sy_ + yy) * tw + sx_ + xx])
        if n > 0.25 * sw:
            break
        chain_rows = yy + 1
    meta["stonePlatform"]["chainRows"] = chain_rows

    tiles.save(os.path.join(args.out_dir, "level-tiles.png"), optimize=True)
    far = Image.open(os.path.join(args.src, "level-bg-far-unslop.png")).convert("RGBA")
    far.save(os.path.join(args.out_dir, "level-bg-far.png"), optimize=True)
    mid = Image.open(os.path.join(args.src, "level-bg-mid-unslop.png")).convert("RGBA")
    k = 1 - args.mid_darken
    r, g, b, a = mid.split()
    r, g, b = (ch.point(lambda v: int(v * k)) for ch in (r, g, b))
    Image.merge("RGBA", (r, g, b, a)).save(os.path.join(args.out_dir, "level-bg-mid.png"), optimize=True)

    data = {
        "id": "level",
        "tiles": {"image": "level-tiles.png", "width": tw, "height": th, "pieces": rects, "meta": meta},
        "far": {"image": "level-bg-far.png", "width": far.width, "height": far.height, "parallax": 0.15, "repeat": "mirror"},
        "mid": {"image": "level-bg-mid.png", "width": mid.width, "height": mid.height, "parallax": 0.45, "repeat": "plain", "darken": args.mid_darken},
    }
    with open(os.path.join(args.out_dir, "level-tiles.json"), "w") as f:
        json.dump(data, f, indent=1)
    with open(os.path.join(args.out_dir, "level-assets.manifest.js"), "w") as f:
        f.write("window.LEVEL_ASSETS = " + json.dumps(data, separators=(", ", ": ")) + ";\n")
    print("\n".join(report))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
