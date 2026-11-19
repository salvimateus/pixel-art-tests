#!/usr/bin/env python3
"""Build a single-file copy of mini-game.html with every script and image inlined.

Usage:
  python3 tools/build-single-html.py [--src mini-game.html] [--out dist/exterminador-no-esgoto.html]

The page loads its manifests with loadScript('<path>') and its images with loadImage('<path>').
This build:
  - inlines each manifest script found in a loadScript('...') call, in source order, and marks it in
    window.EMBEDDED_SCRIPTS so the page does not request it;
  - reads each manifest to find the images it names (paths relative to the manifest folder) and puts
    them in window.EMBEDDED_ASSETS as data:image/png;base64 URIs, with the PNG bytes unchanged;
  - fails if the output still names a local asset file or an external URL.
Standard library only.
"""
import argparse
import base64
import json
import os
import posixpath
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SCRIPT = "<script>\n'use strict';"


def manifest_object(js_text):
    """Return the JSON object assigned by a one-line manifest script."""
    body = js_text[js_text.index("= {") + 2:].strip().rstrip(";").strip()
    return json.loads(body)


def manifest_images(obj):
    images = []
    if "image" in obj:
        images.append(obj["image"])
    for e in (obj.get("effects") or {}).values():
        images.append(e["image"])
    for key in ("tiles", "far", "mid"):
        if isinstance(obj.get(key), dict) and "image" in obj[key]:
            images.append(obj[key]["image"])
    return images


def safe_script(text):
    return text.replace("</script", "<\\/script")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "mini-game.html"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "exterminador-no-esgoto.html"))
    args = ap.parse_args()

    html = open(args.src, encoding="utf-8").read()
    src_dir = os.path.dirname(os.path.abspath(args.src))
    script_paths = re.findall(r"loadScript\('([^']+)'\)", html)
    if not script_paths:
        sys.exit("no loadScript('...') calls found")

    inline_scripts, embedded_scripts, assets = [], {}, {}
    for rel in script_paths:
        path = os.path.join(src_dir, rel)
        text = open(path, encoding="utf-8").read().strip()
        inline_scripts.append(f"<script>\n{safe_script(text)}\n</script>")
        embedded_scripts[rel] = True
        folder = posixpath.dirname(rel)
        for img in manifest_images(manifest_object(text)):
            img_rel = posixpath.normpath(posixpath.join(folder, img))
            data = open(os.path.join(src_dir, img_rel), "rb").read()
            if not data.startswith(b"\x89PNG"):
                sys.exit(f"not a PNG: {img_rel}")
            assets[img_rel] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    maps = (
        "<script>\nwindow.EMBEDDED_SCRIPTS = " + json.dumps(embedded_scripts) + ";\n"
        "window.EMBEDDED_ASSETS = " + json.dumps(assets) + ";\n</script>"
    )
    if html.count(MAIN_SCRIPT) != 1:
        sys.exit("main script marker not found")
    out = html.replace(MAIN_SCRIPT, "\n".join(inline_scripts + [maps]) + "\n" + MAIN_SCRIPT)

    # Every remaining asset path must only be a key of an embedded map, never a fetched file.
    stripped = re.sub(r'"data:image/png;base64,[A-Za-z0-9+/=]+"', '""', out)
    leftovers = [m for m in re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", stripped) if not m.startswith("data:")]
    external = re.findall(r"https?://[^\s\"'<>)]+", stripped)
    if leftovers or external:
        sys.exit(f"local or external references remain: {leftovers + external}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{args.out}: {os.path.getsize(args.out)} bytes, {len(inline_scripts)} scripts, {len(assets)} images")
    for k in assets:
        print("  embedded", k)


if __name__ == "__main__":
    main()
