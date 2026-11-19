"""Shared core of the sprite frame generator.

Pose extraction, the resize to the common art grid, nearest-neighbor transforms, outline and
stray-pixel checks, effect split, packing and the manifest writer. Character configs live in
tools/characters/ and the command is tools/make-sprite-frames.py.
"""
import json
import math
import os
import sys
from collections import deque

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGIN = 2
DARK_LUM = 45  # outline colors are darker than this
MIN_DROP_PX = 3  # smaller detached bits are strays
ART_PX = 2048 / 449  # image px of one art px on the raw API canvas
HITBOX_SLICES = 4  # effect hitboxes: vertical slices per effect frame
HITBOX_MIN_PART = 6  # effect components smaller than this are ignored by the hitboxes
FEET_ROWS = 3  # bottom rows that define the feet for "feet" placement
SPRAY_STEPS = [(0, 0), (3, 0), (6, 1), (10, 3)]  # (distance along spray direction, gravity) per spray frame


def rnd(v):
    return int(math.floor(v + 0.5))


def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * u)


def F(pose, sx=1.0, sy=1.0, lean=0, tx=0, ty=0, anchor="back", drops="follow", **kw):
    """One frame as data.

    ty moves the pixels (ty < 0 lifts the frame above the base row). Optional keys: lean_power,
    bob {"waist": fraction of body height, "dy": px} to move only the rows above the waist,
    offset_y (manifest offsetY only, the pixels stay in place), fx {"effect", "frame"}.
    """
    d = {"pose": pose, "sx": sx, "sy": sy, "lean": lean, "tx": tx, "ty": ty, "anchor": anchor, "drops": drops}
    d.update(kw)
    return d


def spray(step):
    return {"mode": "spray", "dist": SPRAY_STEPS[step][0], "grav": SPRAY_STEPS[step][1]}


def grid_for_canvas(canvas):
    return rnd(canvas[0] / ART_PX), rnd(canvas[1] / ART_PX)


# ---- Source analysis --------------------------------------------------------
def lum(c):
    return (c[0] * 3 + c[1] * 6 + c[2]) / 10


def resize_sheet(im, size):
    """Nearest-neighbor resize to the common art grid. A silhouette edge pixel takes the darkest color
    of the source block it covers when that block holds outline color, so the 1 px outline survives."""
    if size is None or tuple(size) == im.size:
        return im
    w, h = size
    sw, sh = im.size
    rx, ry = sw / w, sh / h
    sp = im.load()
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    op = out.load()
    for y in range(h):
        for x in range(w):
            op[x, y] = sp[min(sw - 1, int((x + 0.5) * rx)), min(sh - 1, int((y + 0.5) * ry))]
    mask = {(x, y) for y in range(h) for x in range(w) if op[x, y][3]}
    for (x, y) in boundary_set(mask):
        block = [sp[i, j] for i in range(int(x * rx), min(sw, math.ceil((x + 1) * rx)))
                 for j in range(int(y * ry), min(sh, math.ceil((y + 1) * ry)))]
        dark = [c for c in block if c[3] and lum(c) < DARK_LUM]
        if dark and lum(op[x, y]) >= DARK_LUM:
            op[x, y] = min(dark, key=lum)
    return out


def load_sheet(path, grid=None):
    im = Image.open(path).convert("RGBA")
    px = im.load()
    alphas = {px[x, y][3] for y in range(im.height) for x in range(im.width)}
    if not alphas <= {0, 255}:
        sys.exit(f"{path}: alpha must be 0 or 255, found {sorted(alphas - {0, 255})[:5]}")
    return resize_sheet(im, grid)


def components(pixels):
    """8-connected components of a {(x, y): color} dict, in scan order."""
    seen = set()
    comps = []
    for p in sorted(pixels, key=lambda q: (q[1], q[0])):
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        pts = {}
        while stack:
            a, b = stack.pop()
            pts[(a, b)] = pixels[(a, b)]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (a + dx, b + dy)
                    if q in pixels and q not in seen:
                        seen.add(q)
                        stack.append(q)
        comps.append(pts)
    return comps


def bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def center(pts):
    b = bbox(pts)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def cells(src, size):
    """Cell rectangles {pose: (x0, y0, x1, y1)} for a cols x rows grid, in reading order."""
    w, h = size
    cols, rows = src.get("cols", 2), src.get("rows", 2)
    return {name: (k % cols * w / cols, k // cols * h / rows, (k % cols + 1) * w / cols, (k // cols + 1) * h / rows)
            for k, name in enumerate(src["poses"])}


def boundary_set(mask):
    out = set()
    for (x, y) in mask:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) not in mask:
                out.add((x, y))
                break
    return out


def nearest_color_map(region, sources, colors):
    """BFS from `sources` over `region`; returns the color of the nearest source for every region pixel."""
    res = {}
    q = deque()
    for p in sources:
        res[p] = colors[p]
        q.append(p)
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (x + dx, y + dy)
            if n in region and n not in res:
                res[n] = res[(x, y)]
                q.append(n)
    return res


def feet_center_x(pts):
    """Center column of the lowest FEET_ROWS rows."""
    y1 = max(p[1] for p in pts)
    xs = [x for x, y in pts if y > y1 - FEET_ROWS]
    return (min(xs) + max(xs)) // 2


def top_back_x(pts, frac):
    """Leftmost column of the top `frac` of the body (a rigid back part, such as a tank)."""
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    return min(x for x, y in pts if y <= y0 + frac * (y1 - y0))


def foot_span(pts):
    """Distance between the centers of the leftmost and rightmost foot blobs in the lowest rows."""
    y1 = max(p[1] for p in pts)
    cols_ = sorted({x for x, y in pts if y > y1 - FEET_ROWS})
    runs, start = [], cols_[0]
    for a, b in zip(cols_, cols_[1:] + [None]):
        if b is None or b != a + 1:
            runs.append((start, a))
            if b is not None:
                start = b
    if len(runs) < 2:
        return 0
    return (runs[-1][0] + runs[-1][1]) / 2 - (runs[0][0] + runs[0][1]) / 2


class Pose:
    """A key pose in rest world coords: x relative to the pivot column, y relative to the base row (0)."""

    def __init__(self, name, body, drops, ref_x, base_y, spray_origin=None, nozzle=None):
        self.name = name
        self.ref_x, self.base_y = ref_x, base_y
        self.px = {(x - ref_x, y - base_y): c for (x, y), c in body.items()}
        self.drops = [{(x - ref_x, y - base_y): c for (x, y), c in d.items()} for d in drops]
        self.outline = boundary_set(self.px)
        interior = set(self.px) - self.outline
        self.outline_near = nearest_color_map(set(self.px), self.outline, self.px)
        self.inward = nearest_color_map(set(self.px), interior, self.px)
        self.x0, self.y0, self.x1, self.y1 = bbox(self.px)
        self.nozzle = (nozzle[0] - ref_x, nozzle[1] - base_y) if nozzle else None
        if spray_origin:
            self.spray_origin = (spray_origin[0] - ref_x, spray_origin[1] - base_y)
        else:
            n = len(self.px)
            self.spray_origin = (sum(p[0] for p in self.px) / n, sum(p[1] for p in self.px) / n)

    @property
    def height(self):
        return self.y1 - self.y0 + 1

    @property
    def width(self):
        return self.x1 - self.x0 + 1

    def anchor_x(self, anchor):
        if isinstance(anchor, (int, float)):
            return anchor
        return {"back": self.x0, "front": self.x1, "center": 0 if self.name == "idle" else (self.x0 + self.x1) / 2,
                "pivot": 0}[anchor]


def check_pose(name, data, min_drop):
    """Drop stray bits of a key pose and report how much of its silhouette edge is not outline color."""
    body = data["body"]
    stray = [p for p in body if sum((p[0] + dx, p[1] + dy) in body
                                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy) <= 1]
    for p in stray:
        del body[p]
    small = [d for d in data["drops"] if len(d) < min_drop]
    data["drops"] = [d for d in data["drops"] if len(d) >= min_drop]
    edge = boundary_set(body)
    light = sum(lum(body[p]) >= DARK_LUM for p in edge)
    print(f"  pose {name}: edge {len(edge)} px, {light} not outline color ({light / len(edge):.0%}), "
          f"removed {len(stray)} stray px and {len(small)} tiny drops, kept {len(data['drops'])} drops")


def load_poses(cfg, grid_override=None):
    """Return (poses, palette, missing sheets, effect sources {pose: {"px": fire pixels, "nozzle"}})."""
    raw, palette, missing, fire = {}, set(), [], {}
    nozzles = cfg.get("fire", {}).get("nozzles", {})
    keep_px = cfg.get("fire", {}).get("keep_right_px", 0)
    for src in cfg["sources"]:
        path = os.path.join(ROOT, src["path"])
        if not os.path.exists(path):
            missing.append(src["path"])
            continue
        grid = grid_override or src.get("grid") or grid_for_canvas(src["canvas"])
        im = load_sheet(path, grid)
        raw_size = Image.open(path).size
        print(f"Load {src['path']}: {raw_size[0]}x{raw_size[1]} -> {im.width}x{im.height}")
        factor = raw_size[0] / im.width
        px = im.load()
        opaque = {(x, y): px[x, y] for y in range(im.height) for x in range(im.width) if px[x, y][3]}
        palette |= set(opaque.values())
        for name, (x0, y0, x1, y1) in cells(src, im.size).items():
            cell = {p: c for p, c in opaque.items() if x0 <= p[0] < x1 and y0 <= p[1] < y1}
            if name in nozzles:
                # Fire is the part right of the nozzle column; a few columns stay for the pilot flame.
                # A cut piece that touches the character across the cut, away from the nozzle (a boot toe),
                # goes back to the character; the flame itself touches only the nozzle.
                nx, ny = int(x0) + nozzles[name][0], int(y0) + nozzles[name][1]
                cut = nx + keep_px
                right = {p: c for p, c in cell.items() if p[0] > cut}
                left = {p: c for p, c in cell.items() if p[0] <= cut}
                pieces = components(right)
                main = min(pieces, key=lambda c: min(abs(x - cut) + abs(y - ny) for x, y in c))
                flame = {}
                for comp in pieces:
                    touches_char = any((cut, y + dy) in left for x, y in comp if x == cut + 1 for dy in (-1, 0, 1))
                    if comp is main or not touches_char:
                        flame.update(comp)
                    else:
                        left.update(comp)
                fire[name] = {"px": {(x - nx, y - ny): c for (x, y), c in flame.items()}, "nozzle": (nx, ny)}
                cell = left
            comps = [c for c in components(opaque) if x0 <= center(c)[0] < x1 and y0 <= center(c)[1] < y1] \
                if name not in nozzles else components(cell)
            if not comps:
                sys.exit(f"{src['path']}: no pixels in the {name} cell")
            body = max(comps, key=len)
            data = {"body": body, "drops": [c for c in comps if c is not body]}
            check_pose(name, data, cfg.get("min_drop_px", MIN_DROP_PX))
            origin = src.get("spray_origin", {}).get(name)
            data["origin"] = (origin[0] / factor, origin[1] / factor) if origin else None
            data["nozzle"] = fire[name]["nozzle"] if name in fire else None
            data["sheet"] = src["path"]
            data["cell"] = (int(x0), int(y0))
            raw[name] = data
    if "idle" not in raw:
        return None, palette, missing, fire
    if cfg.get("prepare"):
        # Config hook: rebuild raw key poses (for example, one shared upper body) before placement.
        cfg["prepare"](raw)
    placement = cfg.get("placement", {})
    ib = bbox(raw["idle"]["body"])
    if cfg.get("pivot") == "feet":
        pivot_x = feet_center_x(raw["idle"]["body"])
    else:
        pivot_x = (ib[0] + ib[2]) // 2
    back_world = ib[0] - pivot_x
    frac = cfg.get("top_back_frac", 0.4)
    top_back_world = top_back_x(raw["idle"]["body"], frac) - pivot_x
    poses = {}
    for name, data in raw.items():
        b = bbox(data["body"])
        base = max(bbox(c)[3] for c in [data["body"]] + data["drops"])
        mode = placement.get(name, cfg.get("default_placement", "center"))
        if name == "idle":
            ref = pivot_x
        elif mode == "back":
            ref = b[0] - back_world
        elif mode == "feet":
            ref = feet_center_x(data["body"])
        elif mode == "top_back":
            ref = top_back_x(data["body"], frac) - top_back_world
        else:
            ref = (b[0] + b[2]) // 2
        poses[name] = Pose(name, data["body"], data["drops"], ref, base, data["origin"], data["nozzle"])
        poses[name].sheet = data["sheet"]
        if name in cfg.get("layers", {}):
            layers = dict(cfg["layers"][name])
            if "flame" in layers:
                layers["flame"] = dict(layers["flame"])
            prepare_layers(poses[name], layers)
            if "flame" in layers:
                # Flicker colors: the pilot flame and the split fire sheet colors only.
                fire_colors = {c for f in fire.values() for c in f["px"].values()}
                layers["flame"]["palette"] = sorted(fire_colors | set(poses[name].flame.values()), key=lum)
    return poses, palette, missing, fire


def scale_check(poses, tolerance, sources):
    print("Key pose body sizes (art px):")
    for name, p in poses.items():
        print(f"  {name:15s} {p.width:4d} x {p.height:3d}  ({p.sheet})")
    idle = poses["idle"]
    by_sheet = {}
    for p in poses.values():
        if p.sheet != idle.sheet:
            by_sheet.setdefault(p.sheet, []).append(p)
    ok = True
    for sheet, ps in by_sheet.items():
        best = min(ps, key=lambda p: abs(p.height - idle.height))
        diff = abs(best.height - idle.height) / idle.height
        limit = tolerance or next((s.get("scale_tolerance") for s in sources if s["path"] == sheet), None) or 0.08
        print(f"  scale {sheet}: closest {best.name} {best.height} px vs idle {idle.height} px, "
              f"{diff:.1%} (limit {limit:.0%})")
        if diff > limit:
            ok = False
    if not ok:
        sys.exit("Scale check failed: a sheet does not match the idle art pixel scale. Nothing was written.")


# ---- Deformation ------------------------------------------------------------
def render_body(pose, anchor_x, sx, sy, lean, tx=0, lean_power=1.0, bob=None):
    """Inverse-map every output pixel to the rest pose with nearest-neighbor sampling.

    The base row (y = 0) is fixed. `lean` is the integer shear of the top row; it falls to 0 at the base.
    `bob` moves only the rows above the waist by dy px; the waist row stretches or loses one row.
    """
    out_h = rnd(pose.height * sy) + 1
    waist, bob_dy = None, 0
    if bob:
        waist, bob_dy = -rnd(pose.height * bob["waist"]), bob["dy"]
        out_h += abs(bob_dy)
    pad = abs(lean) + 4
    xa = int(min(anchor_x + (pose.x0 - anchor_x) * sx + tx, pose.x0)) - pad
    xb = int(max(anchor_x + (pose.x1 - anchor_x) * sx + tx, pose.x1)) + pad
    mask = {}
    for wy in range(-out_h - 1, 1):
        ry = rnd(wy / sy)
        if waist is not None and wy <= waist:
            ry = rnd((wy - bob_dy) / sy)
        hn = min(1.0, (-wy) / max(1, out_h - 1))
        off = rnd(lean * hn ** lean_power) if lean else 0
        for wx in range(xa, xb + 1):
            rx = rnd(anchor_x + (wx - tx - off - anchor_x) / sx)
            if (rx, ry) in pose.px:
                mask[(wx, wy)] = (rx, ry)
    return finish(pose, mask)


def finish(pose, mask):
    """Clean a {output: source} map and redraw the silhouette outline with source outline colors."""
    mask = clean_mask(mask)
    edge = boundary_set(mask)
    img = {}
    for p, s in mask.items():
        if p in edge:
            img[p] = pose.px[s] if s in pose.outline else pose.outline_near[s]
        else:
            img[p] = pose.inward[s] if s in pose.outline else pose.px[s]
    return img


def in_poly(x, y, poly):
    """Even-odd test of the pixel center against a polygon of (x, y) points."""
    cx, cy, inside = x + 0.5, y + 0.5, False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > cy) != (y2 > cy) and cx < x1 + (cy - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def prepare_layers(pose, layers):
    """Tag each pose pixel with its region and cut the flame pixels out into their own sprite."""
    pose.layers = layers
    pose.region = {}
    for p in pose.px:
        pose.region[p] = "torso"
        for name, reg in layers["regions"].items():
            if in_poly(p[0], p[1], reg["poly"]):
                pose.region[p] = name
    fl = layers.get("flame")
    pose.flame = {}
    if fl:
        pose.flame = {p: c for p, c in pose.px.items() if in_poly(p[0], p[1], fl["poly"]) and fl["is_flame"](c)}


def render_layered(pose, fr):
    """Breathing warp of everything above the waist, plus region layers moved on their own.

    The rows in a band just above the waist stretch or shrink, so the legs never move. Each region layer
    is drawn at its own offset; where it moves away, the warped copy below it stays, so nothing tears.
    Depth decides which pixel wins: higher region depth first, a moved layer before a warped copy.
    """
    L, v = pose.layers, fr["layered"]
    waist, band, rise = L["waist"], L["band"], v["rise"]
    depth = {"torso": 1}
    depth.update({n: r["depth"] for n, r in L["regions"].items()})

    def warp_y(oy):
        if oy > waist:
            return oy
        if oy >= waist - band - rise:
            return waist + rnd((oy - waist) * band / (band + rise))
        return oy + rise

    reach = max([abs(rise)] + [abs(d) for d in v.get("dy", {}).values()]) + 2
    best = {}
    for oy in range(pose.y0 - reach, 1):
        for ox in range(pose.x0 - 2, pose.x1 + 3):
            cands = []
            s = (ox, warp_y(oy))
            if s in pose.px and s not in pose.flame:
                cands.append((depth[pose.region[s]], 0, s))
            for name, d in v.get("dy", {}).items():
                s = (ox, oy - d)
                if s in pose.px and s not in pose.flame and pose.region[s] == name:
                    cands.append((depth[name], 1, s))
            if cands:
                best[(ox, oy)] = max(cands)[2]
    img = finish(pose, best)
    if pose.flame and "flame" in v:
        dy = v.get("dy", {}).get(L["flame"]["follows"], 0)
        for (x, y), c in flame_variant(pose, v["flame"]).items():
            img[(x, y + dy)] = c
    return img


def flame_variant(pose, k):
    """Flicker shape k of the pilot flame: nearest-neighbor stretch around its bottom-left root and a
    shift along the flame palette, so only existing flame colors are used."""
    fl = pose.flame
    sx, sy, shift = pose.layers["flame"]["variants"][k % len(pose.layers["flame"]["variants"])]
    x0, _, _, y1 = bbox(fl)
    pal = sorted({c for c in pose.layers["flame"]["palette"]}, key=lum)
    out = {}
    xs = [p[0] for p in fl]
    ys = [p[1] for p in fl]
    for y in range(y1 - rnd((y1 - min(ys)) * sy) - 1, y1 + 1):
        for x in range(x0, x0 + rnd((max(xs) - x0) * sx) + 2):
            q = (x0 + rnd((x - x0) / sx), y1 + rnd((y - y1) / sy))
            if q in fl:
                c = fl[q]
                i = min(len(pal) - 1, max(0, pal.index(c) + shift)) if c in pal else None
                out[(x, y)] = pal[i] if i is not None else c
    return {p: c for p, c in out.items()
            if any((p[0] + dx, p[1] + dy) in out for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)}


def forward_point(pose, fr, pt):
    """Where a rest-pose point lands after the frame transform (lean and bob counted at its row)."""
    ax = pose.anchor_x(fr["anchor"])
    x = ax + (pt[0] - ax) * fr["sx"] + fr["tx"]
    y = pt[1] * fr["sy"]
    bob = fr.get("bob")
    if bob and y <= -rnd(pose.height * bob["waist"]):
        y += bob["dy"]
    if fr["lean"]:
        out_h = rnd(pose.height * fr["sy"]) + 1
        x += rnd(fr["lean"] * min(1.0, -y / max(1, out_h - 1)) ** fr.get("lean_power", 1.0))
    return rnd(x), rnd(y) + fr["ty"]


def clean_mask(mask):
    """Drop isolated pixels and fill 1 px pinholes. Pinholes take the sample of an opaque neighbor."""
    changed = True
    while changed:
        changed = False
        for p in list(mask):
            x, y = p
            n8 = sum((x + dx, y + dy) in mask for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
            n4 = sum((x + dx, y + dy) in mask for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if n8 <= 1 or n4 == 0:
                del mask[p]
                changed = True
        holes = set()
        for (x, y) in mask:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + dx, y + dy)
                if q not in mask:
                    holes.add(q)
        for q in holes:
            x, y = q
            n4 = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (x + dx, y + dy) in mask]
            if len(n4) == 4 and y <= 0:
                mask[q] = mask[n4[0]]
                changed = True
    return mask


def stamp(img, drop, dx, dy):
    for (x, y), c in drop.items():
        img.setdefault((x + dx, y + dy), c)


def touches(img, drop, dx, dy=0):
    return any((x + dx + ax, y + dy + ay) in img
               for (x, y) in drop for ax in (-1, 0, 1) for ay in (-1, 0, 1))


def render_frame(poses, fr, spray_release):
    pose = poses[fr["pose"]]
    ax = pose.anchor_x(fr["anchor"])
    if fr.get("layered"):
        body = render_layered(pose, fr)
    else:
        body = render_body(pose, ax, fr["sx"], fr["sy"], fr["lean"], fr["tx"], fr.get("lean_power", 1.0),
                           fr.get("bob"))
    layer = {}
    mode = fr["drops"] if isinstance(fr["drops"], str) else fr["drops"]["mode"]
    for k, drop in enumerate(pose.drops):
        cx, cy = center(drop)
        if mode == "hide":
            break
        if mode == "follow":
            dx = rnd(ax + (cx - ax) * fr["sx"] + fr["tx"] - cx)
            dy = rnd(cy * fr["sy"] - cy)
        elif mode == "trail":
            # Trail drops follow the back edge with lag and a small per-drop wiggle.
            back, phase = fr["drops"]["back"], fr["drops"]["phase"]
            dx = rnd(back * (0.4 + 0.2 * k)) + rnd(0.6 * math.sin(2 * math.pi * phase + k * 2.1))
            dy = 0
        else:
            # Spray: released at the first spray frame transform, then free flight away from the origin.
            rel = spray_release.setdefault(fr["pose"], fr)
            rax = pose.anchor_x(rel["anchor"])
            ox, oy = pose.spray_origin
            vx, vy = cx - ox, cy - oy
            norm = math.hypot(vx, vy) or 1
            dist, grav = fr["drops"]["dist"], fr["drops"]["grav"]
            bx = rax + (cx - rax) * rel["sx"] + rel["tx"] - cx
            by = cy * rel["sy"] - cy + grav
            dx, dy = rnd(bx + dist * vx / norm), rnd(by + dist * vy / norm)
            # A flying drop never fuses with the body: push it further along its path.
            extra = 0
            while (touches(body, drop, dx, dy) or touches(layer, drop, dx, dy)) and extra < 20:
                extra += 1
                dx, dy = rnd(bx + (dist + extra) * vx / norm), rnd(by + (dist + extra) * vy / norm)
        # Keep a 1 px gap so a drop never fuses with the body outline.
        step = -1 if mode == "trail" else 0
        if step:
            while touches(body, drop, dx, dy) or touches(layer, drop, dx, dy):
                dx += step
        stamp(layer, drop, dx, dy)
    out = dict(layer)
    out.update(body)
    if fr["ty"]:
        out = {(x, y + fr["ty"]): c for (x, y), c in out.items()}
    return out


# ---- Effects ----------------------------------------------------------------
def shrink(px, scale):
    """Nearest-neighbor shrink of an effect frame around its anchor (0, 0)."""
    if scale == 1:
        return dict(px)
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    out = {}
    for y in range(rnd(min(ys) * scale) - 1, rnd(max(ys) * scale) + 2):
        for x in range(rnd(min(xs) * scale) - 1, rnd(max(xs) * scale) + 2):
            q = (rnd(x / scale), rnd(y / scale))
            if q in px:
                out[(x, y)] = px[q]
    return out


def build_effects(cfg, fire):
    """Return {effect: [frame pixel dicts with the anchor at (0, 0)]}."""
    out = {}
    for name, spec in cfg.get("effects", {}).items():
        frames = []
        for f in spec["frames"]:
            if f["pose"] not in fire:
                return {}
            px = shrink(fire[f["pose"]]["px"], f.get("scale", 1))
            if f.get("crop_x"):
                px = {p: c for p, c in px.items() if p[0] <= f["crop_x"]}
            frames.append(px)
        out[name] = frames
    return out


# ---- Self-checks ------------------------------------------------------------
def check_frames(name, frames, specs):
    """Fail on stray pixels or pixels below the base; report the biggest jump between neighbor frames."""
    prev = None
    worst = (0, 0)
    for i, (f, spec) in enumerate(zip(frames, specs)):
        if any(y > 0 for _, y in f):
            sys.exit(f"{name} frame {i}: pixels below the base row")
        if not spec["ty"] and not any(y == 0 for _, y in f):
            sys.exit(f"{name} frame {i}: base row missing")
        lonely = [p for p in f if not any((p[0] + dx, p[1] + dy) in f
                                          for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)]
        if lonely:
            sys.exit(f"{name} frame {i}: {len(lonely)} stray pixels")
        b = bbox(largest_part(f))
        if prev:
            jump = max(abs(a - c) for a, c in zip(b, prev))
            if jump > worst[0]:
                worst = (jump, i)
        prev = b
    print(f"  {name}: {len(frames)} frames, biggest body bbox jump {worst[0]} px (into frame {worst[1]})")


def largest_part(f):
    seen, best = set(), []
    for p in f:
        if p in seen:
            continue
        stack, part = [p], []
        seen.add(p)
        while stack:
            a = stack.pop()
            part.append(a)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (a[0] + dx, a[1] + dy)
                    if q in f and q not in seen:
                        seen.add(q)
                        stack.append(q)
        if len(part) > len(best):
            best = part
    return best


# ---- Build and write --------------------------------------------------------
def build(cfg, out_dir, name=None, contact=None, contact_scale=4, scale_tolerance=None, grid=None):
    cid = cfg["id"]
    name = name or f"{cid}-anim"
    poses, palette, missing, fire = load_poses(cfg, grid)
    for m in missing:
        print(f"Skip: {m} does not exist yet; its animations are left out.")
    if poses is None:
        sys.exit("No idle pose: the sheet with idle is missing. Nothing was written.")
    scale_check(poses, scale_tolerance, cfg["sources"])
    effects = build_effects(cfg, fire)

    rows = []
    for anim in cfg["animations"]:
        specs = anim["build"](poses)
        need = sorted({s["pose"] for s in specs} - set(poses))
        if need:
            print(f"Skip animation {anim['name']}: missing poses {', '.join(need)}.")
            continue
        fx_need = sorted({s["fx"]["effect"] for s in specs if s.get("fx")} - set(effects))
        if fx_need:
            print(f"Skip animation {anim['name']}: missing effects {', '.join(fx_need)}.")
            continue
        release = {}
        frames = [render_frame(poses, s, release) for s in specs]
        fps = anim["fps"](poses) if callable(anim["fps"]) else anim["fps"]
        entry = {"name": anim["name"], "frames": len(frames), "fps": fps, "loop": anim["loop"]}
        if "stride" in anim:
            entry["stride"] = anim["stride"](poses)
        if "phases" in anim:
            # Inclusive frame ranges the game uses (hold, loop, final state); the viewer ignores them.
            entry["phases"] = {k: list(v) for k, v in anim["phases"].items()}
        rows.append((entry, frames, specs))

    print("Self-check:")
    for entry, frames, specs in rows:
        check_frames(entry["name"], frames, specs)

    allpts = [p for _, fr, _ in rows for f in fr for p in f]
    minx = min(p[0] for p in allpts)
    maxx = max(p[0] for p in allpts)
    miny = min(p[1] for p in allpts)
    piv_x = MARGIN - minx
    base_row = MARGIN - miny
    fw = maxx - minx + 1 + 2 * MARGIN
    fh = base_row + 1 + MARGIN
    cols = max(len(fr) for _, fr, _ in rows)
    sheet = Image.new("RGBA", (cols * fw, len(rows) * fh), (0, 0, 0, 0))
    spx = sheet.load()
    for r, (_, fr, _) in enumerate(rows):
        for i, f in enumerate(fr):
            for (x, y), c in f.items():
                if c not in palette:
                    sys.exit(f"row {r} frame {i}: color {c} is not in the source")
                spx[i * fw + piv_x + x, r * fh + base_row + y] = c

    # Optional per-animation keys: offsetY (manifest only) and fx (effect frame on the nozzle point).
    for entry, frames, specs in rows:
        if any(s.get("offset_y") for s in specs):
            entry["offsetY"] = [int(s.get("offset_y", 0)) for s in specs]
        if any(s.get("fx") for s in specs):
            fx = []
            for s in specs:
                if not s.get("fx"):
                    fx.append(None)
                    continue
                nx, ny = forward_point(poses[s["pose"]], s, poses[s["pose"]].nozzle)
                fx.append({"effect": s["fx"]["effect"], "frame": s["fx"]["frame"],
                           "x": piv_x + nx, "y": base_row + ny})
            entry["fx"] = fx

    os.makedirs(out_dir, exist_ok=True)
    image = name + ".png"
    png = os.path.join(out_dir, image)
    sheet.save(png)
    manifest = {"id": cid, "image": image, "frameWidth": fw, "frameHeight": fh,
                # ground_sink: the bottom rows of a soft body sit that many px into the floor surface.
                "pivot": {"x": piv_x, "y": base_row + 1 - cfg.get("ground_sink", 0)},
                "animations": [{"name": e["name"], "row": r, **{k: v for k, v in e.items() if k != "name"}}
                               for r, (e, _, _) in enumerate(rows)]}
    fx_sheets = {}
    if effects:
        manifest["effects"] = {}
        for ename, frames in effects.items():
            pts = [p for f in frames for p in f]
            ex0, ey0, ex1, ey1 = bbox(pts)
            efw, efh = ex1 - ex0 + 1 + 2 * MARGIN, ey1 - ey0 + 1 + 2 * MARGIN
            ax, ay = MARGIN - ex0, MARGIN - ey0
            esheet = Image.new("RGBA", (efw * len(frames), efh), (0, 0, 0, 0))
            ep = esheet.load()
            for k, f in enumerate(frames):
                for (x, y), c in f.items():
                    ep[k * efw + ax + x, ay + y] = c
            eimage = f"{cid}-{ename}.png"
            esheet.save(os.path.join(out_dir, eimage))
            fx_sheets[ename] = (esheet, efw, efh, ax, ay)
            manifest["effects"][ename] = {"image": eimage, "frameWidth": efw, "frameHeight": efh,
                                          "frames": len(frames), "anchor": {"x": ax, "y": ay},
                                          "hitboxes": [hitboxes({(x + ax, y + ay): c for (x, y), c in f.items()})
                                                       for f in frames]}
            print(f"Wrote effect {eimage} ({esheet.width}x{esheet.height}, {len(frames)} frames)")
    text = json.dumps(manifest)
    with open(os.path.join(out_dir, name + ".json"), "w") as fh_:
        fh_.write(text + "\n")
    with open(os.path.join(out_dir, name + ".manifest.js"), "w") as fh_:
        fh_.write(f'(window.SPRITE_ANIMS = window.SPRITE_ANIMS || {{}})["{cid}"] = {text};\n')

    if contact:
        write_contact(contact, contact_scale, sheet, rows, cols, fw, fh, piv_x, base_row, manifest, fx_sheets)
    print(f"Wrote {png} ({sheet.width}x{sheet.height})")
    print(text)


def hitboxes(px, slices=HITBOX_SLICES, min_part=HITBOX_MIN_PART):
    """Up to `slices` vertical slices along an effect frame, each the bounding rectangle of the main flame
    component in its column range. Components smaller than `min_part` px are never the main body."""
    # Only the main flame body counts; detached embers and stray bits would stretch the slices.
    parts = [c for c in components(px) if len(c) >= min_part]
    keep = list(max(parts, key=len)) if parts else []
    if not keep:
        return []
    x0, _, x1, _ = bbox(keep)
    n = min(slices, x1 - x0 + 1)
    out = []
    for k in range(n):
        a = x0 + (x1 - x0 + 1) * k // n
        b = x0 + (x1 - x0 + 1) * (k + 1) // n - 1
        part = [p for p in keep if a <= p[0] <= b]
        if part:
            bx0, by0, bx1, by1 = bbox(part)
            out.append({"x": bx0, "y": by0, "w": bx1 - bx0 + 1, "h": by1 - by0 + 1})
    return out


def write_contact(path, scale, sheet, rows, cols, fw, fh, piv_x, base_row, manifest, fx_sheets):
    """Dark background, cell grid, base line and pivot column. Effects are composited on top, and the
    columns are spread apart so an effect never covers the next cell."""
    pad = 0
    for a in manifest["animations"]:
        for e in a.get("fx") or []:
            if e:
                _, efw, _, ax, _ = fx_sheets[e["effect"]]
                pad = max(pad, e["x"] - ax + efw - fw)
    step = fw + pad
    bg = Image.new("RGBA", (cols * step, sheet.height), (34, 34, 40, 255))
    g = bg.load()
    for r in range(len(rows)):
        for i in range(cols):
            for x in range(fw):
                g[i * step + x, r * fh + base_row + 1] = (70, 70, 90, 255)
            for y in range(fh):
                g[i * step, r * fh + y] = (60, 60, 60, 255)
                g[i * step + piv_x, r * fh + y] = (55, 55, 75, 255)
    for r in range(len(rows)):
        for i in range(cols):
            cell = sheet.crop((i * fw, r * fh, (i + 1) * fw, (r + 1) * fh))
            layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
            layer.paste(cell, (i * step, r * fh))
            bg.alpha_composite(layer)
    for r, a in enumerate(manifest["animations"]):
        for i, e in enumerate(a.get("fx") or []):
            if not e:
                continue
            es, efw, efh, ax, ay = fx_sheets[e["effect"]]
            crop = es.crop((e["frame"] * efw, 0, (e["frame"] + 1) * efw, efh))
            layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
            layer.paste(crop, (i * step + e["x"] - ax, r * fh + e["y"] - ay))
            bg.alpha_composite(layer)
    bg.resize((bg.width * scale, bg.height * scale), Image.NEAREST).save(path)
