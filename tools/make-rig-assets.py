#!/usr/bin/env python3
"""Build a skeletal cutout rig: the part atlas, the rig manifest and the baked animations.

Usage:
  python3 tools/make-rig-assets.py <id> [--out-dir DIR] [--contact DIR] [--contact-scale 4]

<id> is a config in tools/rigs/ (black-knight). The outputs go to the config out_dir:
<name>.png (atlas), <name>.json and <name>.manifest.js, which registers window.SPRITE_RIGS[id].
--contact also writes one frame strip per animation and a rest-vs-master check to DIR.
The manifest contract is documented at the top of tools/rig_core.py.
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rigs"))

import rig_core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", help="rig id, such as black-knight")
    ap.add_argument("--out-dir", help="default: the out_dir of the rig config")
    ap.add_argument("--contact", help="write animation strips and the rest check to this folder")
    ap.add_argument("--contact-scale", type=int, default=4)
    args = ap.parse_args()

    try:
        cfg = importlib.import_module(args.id.replace("-", "_")).CONFIG
    except ModuleNotFoundError:
        sys.exit(f"No config for {args.id} in tools/rigs/.")
    out_dir = args.out_dir or os.path.join(rig_core.ROOT, cfg["out_dir"])
    rig_core.build(cfg, out_dir, args.contact, args.contact_scale)


if __name__ == "__main__":
    main()
