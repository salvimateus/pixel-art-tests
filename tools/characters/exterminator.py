"""Exterminator config: key sheets, nozzle points, fire effect and animation data.

The human body is never squashed: frames use the drawn poses, a 1 px upper-body bob (`bob`) and
manifest-only `offset_y` for the jump. The flamethrower fire is split into its own effect sheet.
"""
import math

from sprite_core import F, foot_span, lum, rnd

ID = "exterminator"
OUT_DIR = "assets/exterminator"

# The target grid comes from the raw API canvas (2048/449 image px per art px). The rigid boots measure
# the same width on the idle and the attack poses at this grid, so no extra scale is applied.
SOURCES = [
    {"path": "assets/exterminator/keys/exterminator-keys-jump-unslop.png", "canvas": (2560, 1024), "cols": 4, "rows": 1,
     "poses": ["idle", "jump_crouch", "jump_air", "landing"]},
    # 8-frame walk: 2x4 grid of 640x1024 cells, frames 1 to 4 in the top row.
    {"path": "assets/exterminator/keys/exterminator-keys-walk8-unslop.png", "canvas": (2560, 2048), "cols": 4,
     "rows": 2, "poses": [f"walk8_{k}" for k in range(1, 9)]},
    {"path": "assets/exterminator/keys/exterminator-keys-attack-unslop.png", "canvas": (3200, 1600), "cols": 2, "rows": 2,
     "poses": ["aim", "burst", "full_a", "full_b"]},
]

# Nozzle points in cell-local grid px, measured at the barrel tip where the fire starts.
# Every pixel right of the nozzle column is fire; the recoil moves the nozzle between poses.
FIRE = {"nozzles": {"burst": (124, 85), "full_a": (122, 90), "full_b": (120, 89)}, "keep_right_px": 0}

# Fire effect frames: 0 burst, 1 full A, 2 full B, 3 and 4 the fire that dies down (shrunk full A).
EFFECTS = {"fire": {"frames": [{"pose": "burst"}, {"pose": "full_a"}, {"pose": "full_b"},
                               {"pose": "full_a", "scale": 0.6}, {"pose": "full_a", "scale": 0.35}]}}

WAIST = 0.5  # the jump and attack bob moves the rows above half the body height

# Idle layers on the idle key pose, in rest world coords (x from the pivot, y up is negative, 0 = base row).
# The legs stay below the waist row; the rows of the band above it stretch for the breath.
IDLE_LAYERS = {
    "waist": -64,
    "band": 8,
    "regions": {
        "tank": {"depth": 0, "poly": [(-46, -164), (0, -164), (0, -150), (-12, -146), (-12, -86), (-46, -86)]},
        "head": {"depth": 2, "poly": [(2, -165), (44, -165), (44, -118), (12, -118), (2, -128)]},
        "arms": {"depth": 3, "poly": [(-16, -98), (44, -98), (60, -86), (60, -69), (-16, -69)]},
    },
    "flame": {
        "poly": [(48, -84), (60, -84), (60, -70), (48, -70)],
        "is_flame": lambda c: (c[0] >= 200 and c[2] <= 75) or (c[0] >= 245 and c[1] >= 245),
        "follows": "arms",
        # (width scale, height scale, palette shift) for each flicker shape.
        "variants": [(1.0, 1.0, 0), (0.85, 1.35, 1), (1.15, 0.8, -1), (0.7, 1.15, 2), (1.0, 0.65, -2)],
    },
}

IDLE_N = 12
IDLE_FLAME = [0, 1, 3, 2, 4, 1, 0, 3, 2, 1, 4, 3]


def build_idle(poses):
    """Layered breath: the torso rises up to 3 px; the tank lags 1 frame with 1 px more; the head nods
    at the top of the inhale; the arms and nozzle dip at the end of the exhale; the pilot flame flickers."""
    frames = []
    for i in range(IDLE_N):
        rise = rnd(3 * (0.5 - 0.5 * math.cos(2 * math.pi * i / IDLE_N)))
        tank = rnd(4 * (0.5 - 0.5 * math.cos(2 * math.pi * (i - 1) / IDLE_N)))
        nod = 1 if i in (6, 7) else 0
        dip = 1 if i in (10, 11) else 0
        frames.append(F("idle", anchor="pivot", layered={"rise": rise, "flame": IDLE_FLAME[i],
                                                         "dy": {"tank": -tank, "head": -rise + nod, "arms": -rise + dip}}))
    return frames


def P(pose, **kw):
    return F(pose, anchor="pivot", **kw)


WALK8 = [f"walk8_{k}" for k in range(1, 9)]


def walk_stride(poses):
    """Planted-foot travel over the cycle: the foot planted at a contact frame ends at the back of the next
    contact frame, so each step is the heel-to-heel spread of a contact pose (frames 1 and 5)."""
    return rnd(foot_span(poses["walk8_1"].px) + foot_span(poses["walk8_5"].px))


def build_walk(poses):
    return [P(n) for n in WALK8]


# ---- Walk upper body ---------------------------------------------------------
# The model redrew the upper body in each walk cell and cut it with a straight hem. Frame 1's upper body
# goes on every frame at that frame's own bob, the legs come from each frame, and a torn hem hides the seam.
HEM_Y = 147  # cell-local row of the straight hem line in walk frame 1 (2560x2048 canvas grid)
MATCH_ROWS = 135  # rows above this match the upper body of each frame against frame 1
HEM_DEPTH = [2, 3, 4, 3, 2, 2, 3, 4, 4, 3, 2, 3, 4, 3, 2, 2, 4, 3, 2, 3, 3, 4, 2, 2, 3, 4, 3, 2]


def _local(data):
    cx, cy = data["cell"]
    return {(x - cx, y - cy): c for (x, y), c in data["body"].items()}


def _offset(ref, px):
    """(dx, dy) that best lands the frame-1 upper body on this frame (color match on rigid rows)."""
    def score(dx, dy):
        n = 0
        for (x, y), c in ref.items():
            d = px.get((x + dx, y + dy))
            if d and abs(c[0] - d[0]) + abs(c[1] - d[1]) + abs(c[2] - d[2]) < 60:
                n += 1
        return n
    return max(((score(dx, dy), -abs(dx) - abs(dy), dx, dy) for dx in range(-4, 5) for dy in range(-5, 4)))[2:]


def _torn_hem(upper, extra=0):
    """Extend the jacket 2 to 4 px below the straight hem with an uneven edge, a dark outline at the new
    edge and a small fold highlight, using jacket colors from the rows just above."""
    cols = sorted(x for x, y in upper if y == HEM_Y)
    jacket = {}
    for x in cols:
        jacket[x] = [upper[(x, y)] for y in range(HEM_Y - 1, HEM_Y - 8, -1)
                     if (x, y) in upper and lum(upper[(x, y)]) >= 70]
    shades = sorted({c for v in jacket.values() for c in v}, key=lum)
    dark = min((c for (x, y), c in upper.items() if y == HEM_Y), key=lum)
    light = shades[-max(1, len(shades) // 8)] if shades else None
    hem = {}
    for i, x in enumerate(cols):
        h = HEM_DEPTH[i % len(HEM_DEPTH)]
        if i < 2 or i >= len(cols) - 2:
            h = 2  # the hem tapers at the sides
        h += extra
        src = jacket[x] or shades[len(shades) // 2:len(shades) // 2 + 1] or [dark]
        for j in range(0, h):
            hem[(x, HEM_Y + j)] = src[j % len(src)]
        hem[(x, HEM_Y + h)] = dark
        if light and h >= 4 and HEM_DEPTH[(i - 1) % len(HEM_DEPTH)] >= 3:
            hem[(x, HEM_Y + h - 1)] = light
    return hem


def prepare(raw):
    if not all(n in raw for n in WALK8):
        return
    first = _local(raw[WALK8[0]])
    upper = {p: c for p, c in first.items() if p[1] <= HEM_Y}
    ref = {p: c for p, c in first.items() if p[1] <= MATCH_ROWS}
    base1 = max(y for _, y in first)
    WALK_BOB.clear()
    for k, n in enumerate(WALK8):
        px = _local(raw[n])
        dx, dy = _offset(ref, px)
        # The torso sits WALK_BOB_PX[k] px from its frame-1 height above the feet (negative is up).
        place = WALK_BOB_PX[k] + max(y for _, y in px) - base1
        WALK_BOB.append((dx, dy, place))
        # The legs start 2 rows below the old straight hem, so no trace of that line stays under the new hem.
        legs_top = HEM_Y + dy + 3
        # The shallowest hem column ends at HEM_Y + place + 2; deepen the hem when the legs start lower.
        extra = max(0, legs_top - (HEM_Y + place + 3))
        torso = dict(upper)
        torso.update(_torn_hem(upper, extra))
        frame = {p: c for p, c in px.items() if p[1] >= legs_top}
        frame.update({(x + dx, y + place): c for (x, y), c in torso.items()})
        cx, cy = raw[n]["cell"]
        raw[n]["body"] = {(x + cx, y + cy): c for (x, y), c in frame.items()}


WALK_BOB_PX = [0, 0, 0, -2, 0, 0, 0, -2]  # designed torso bob per walk frame; both steps rise the same
WALK_BOB = []  # filled by prepare: (dx, measured dy, placed dy) of each walk frame's upper body


def build_jump(poses):
    # The pixels stay in the cell; offset_y is the suggested rise and fall for the game.
    return [
        P("idle"),
        P("jump_crouch"),
        P("jump_crouch", bob={"waist": WAIST, "dy": 1}),
        P("jump_air", offset_y=-10),
        P("jump_air", offset_y=-18),
        P("jump_air", offset_y=-21),
        P("jump_air", offset_y=-17),
        P("jump_air", offset_y=-8),
        P("landing"),
        P("idle"),
    ]


def fire(frame):
    return {"effect": "fire", "frame": frame}


def build_attack(poses):
    return [
        P("aim"),
        P("aim", bob={"waist": WAIST, "dy": 1}),
        P("burst", fx=fire(0)),
        P("full_a", fx=fire(1)),
        P("full_b", fx=fire(2)),
        P("full_a", fx=fire(1)),
        P("full_b", fx=fire(2)),
        P("full_a", fx=fire(1)),
        P("full_b", fx=fire(2)),
        P("full_a", fx=fire(3)),
        P("burst", fx=fire(4)),
        P("aim"),
        P("aim", bob={"waist": WAIST, "dy": -1}),
        P("idle"),
    ]


ANIMATIONS = [
    {"name": "idle", "fps": 10, "loop": True, "build": build_idle},
    {"name": "walk", "fps": 10, "loop": True, "build": build_walk, "stride": walk_stride},
    {"name": "jump", "fps": 12, "loop": False, "build": build_jump,
     "phases": {"takeoff": [0, 2], "air": [3, 7], "land": [8, 9]}},
    {"name": "attack", "fps": 12, "loop": False, "build": build_attack,
     "phases": {"start": [0, 2], "sustain": [3, 8], "end": [9, 13]}},
]

CONFIG = {"id": ID, "out_dir": OUT_DIR, "sources": SOURCES, "pivot": "feet", "default_placement": "top_back",
          "fire": FIRE, "effects": EFFECTS, "layers": {"idle": IDLE_LAYERS}, "prepare": prepare,
          "animations": ANIMATIONS}
