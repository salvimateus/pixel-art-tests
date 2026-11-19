#!/usr/bin/env python3
"""Build a character sprite sheet, its effect sheets and its manifest from key-pose sheets.

Usage:
  python3 tools/make-sprite-frames.py <id> [--out-dir DIR] [--name NAME] [--contact OUT.png]
                                           [--contact-scale 4] [--scale-tolerance 0.08] [--grid WxH]

<id> is a config in tools/characters/ (slime-drop, exterminator). The outputs go to the config
out_dir: <id>-anim.png, <id>-anim.json, <id>-anim.manifest.js and one <id>-<effect>.png per effect.
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "characters"))

import sprite_core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", help="character id, such as slime-drop or exterminator")
    ap.add_argument("--out-dir", help="default: the out_dir of the character config")
    ap.add_argument("--name", help="default: <id>-anim")
    ap.add_argument("--contact", help="also write a contact sheet on a dark background to this path")
    ap.add_argument("--contact-scale", type=int, default=4)
    ap.add_argument("--scale-tolerance", type=float, help="override the per-sheet limit (default 0.08)")
    ap.add_argument("--grid", help="override the art grid of every sheet, as WxH")
    args = ap.parse_args()

    try:
        cfg = importlib.import_module(args.id.replace("-", "_")).CONFIG
    except ModuleNotFoundError:
        sys.exit(f"No config for {args.id} in tools/characters/.")
    out_dir = args.out_dir or os.path.join(sprite_core.ROOT, cfg["out_dir"])
    grid = tuple(map(int, args.grid.split("x"))) if args.grid else None
    sprite_core.build(cfg, out_dir, args.name, args.contact, args.contact_scale, args.scale_tolerance, grid)


if __name__ == "__main__":
    main()
