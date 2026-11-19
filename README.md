# Pixel Art Tests

A study project about pixel art animation in the browser.

## Approaches

The project tests two ways to animate pixel art characters:

- Frame sheets: each animation is a row of complete frames. The `exterminator` and `slime-drop` characters use this approach. The level assets for the mini game use the same build pipeline.
- Skeletal cutout rig: the character is a set of separate parts on bones, and a small runtime moves the parts. The `black-knight` character uses this approach.

The source art comes from an image model. The build tools clean the art and snap it to a pixel grid.

## Folder layout

- `assets/`: the built sprite sheets, atlases, JSON data and manifest scripts that the pages load.
- `heroes/`: the source art and the image sidecars (`*.png.json`) that record how each image was made.
- `tools/`: the Python build tools and their shared code.
- `tools/characters/`: the configuration of each frame-sheet character.
- `tools/rigs/`: the configuration of each rig.
- `dist/`: a single-file build of the mini game, made by `tools/build-single-html.py`.

## Open the pages

Open each page directly in the browser by `file://`. You do not need a server.

- `sprite-viewer.html`: shows the frame-sheet animations.
- `rig-viewer.html`: shows the black knight rig and lets you play it.
- `mini-game.html`: a small level with the frame-sheet characters.

## Rig viewer keys

Click the stage first, then use these keys:

- `A` / `D`: walk.
- `Shift` + `A` / `D`: sprint.
- `Space`: jump.
- `S` (hold): crouch.
- `J`: attack.
- `K` (hold): block.
- `H`: play the hurt animation.
- `L`: play the death animation.
- `R`: reset.

## Rebuild the assets

You need Python 3 and Pillow. Run these commands from the project root:

```bash
python3 tools/make-sprite-frames.py exterminator
python3 tools/make-sprite-frames.py slime-drop
python3 tools/make-rig-assets.py black-knight
python3 tools/make-level-assets.py
```

## Asset contracts

`PRIMITIVES.md` documents the asset contracts: the data formats, the manifests and the rules that the tools and pages share.
