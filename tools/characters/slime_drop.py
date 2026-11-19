"""Slime "Drop" config: key sheets, placement and animation data."""
import math

from sprite_core import F, ease, rnd, spray

ID = "slime-drop"
OUT_DIR = "assets/slime"

# Every key sheet is resized with nearest-neighbor to one art grid. This grid keeps the earlier output
# pixel-identical; the canvas formula would give 449x302 for the 2048x1376 canvas.
ART_GRID = (449, 301)
SOURCES = [
    {"path": "assets/slime/keys/slime-drop-keys-base-unslop.png", "grid": ART_GRID,
     "poses": ["idle", "crawl", "attack_mid", "attack_full"]},
    {"path": "assets/slime/keys/slime-drop-keys-heavy-unslop.png", "grid": ART_GRID,
     "poses": ["heavy_windup", "heavy_jump", "heavy_impact", "heavy_recover"]},
    # death_hit is drawn bigger than idle on purpose, so this sheet gets a wider scale tolerance.
    {"path": "assets/slime/keys/slime-drop-keys-death-unslop.png", "grid": ART_GRID, "scale_tolerance": 0.12,
     "poses": ["death_hit", "death_collapse", "death_puddle", "death_remains"]},
]

# How each pose sits on the shared pivot: "center" puts the body bbox center on the pivot,
# "back" puts the body back (left) edge on the idle back edge.
PLACEMENT = {"attack_mid": "back", "attack_full": "back"}

IDLE = {"frames": 8, "fps": 8, "breath_sy": 0.05, "breath_sx": 0.035, "wobble_px": 2, "wobble_power": 1.6}
CRAWL = {"frames": 8, "fps": 10, "stride_ratio": 8 / 126, "lower_sy": 0.05, "lean_px": 1}


def build_idle(poses):
    frames = []
    n = IDLE["frames"]
    for i in range(n):
        t = 2 * math.pi * i / n
        s = math.sin(t)
        frames.append(F("idle", 1 - IDLE["breath_sx"] * s, 1 + IDLE["breath_sy"] * s,
                        rnd(IDLE["wobble_px"] * math.sin(t + math.pi / 2)), anchor="center",
                        lean_power=IDLE["wobble_power"]))
    return frames


def crawl_stride(poses):
    p = poses["crawl"]
    return max(2, rnd((p.x1 - p.x0) * CRAWL["stride_ratio"]))


def build_crawl(poses):
    p = poses["crawl"]
    n = CRAWL["frames"]
    stride = crawl_stride(poses)
    length = p.x1 - p.x0
    half = n // 2
    frames = []
    for i in range(n):
        # Treadmill: the planted end slides back at ground speed, so the cycle loops in place.
        drift = -stride * i / n
        if i < half:
            back, front = 0.0, stride * ease(i / half)
        else:
            back, front = stride * ease((i - half) / half), float(stride)
        back += drift
        front += drift
        stretch = (front - back) / stride
        lean = rnd(CRAWL["lean_px"] * stretch) * (-1 if i < half else 1)
        frames.append(F("crawl", (length + front - back) / length, 1 - CRAWL["lower_sy"] * stretch, lean,
                        tx=back, drops={"mode": "trail", "back": back, "phase": i / n}))
    return frames


def build_attack(poses):
    # attack_mid and attack_full bridge the size change; stretch keeps the front edge moving by small steps.
    return [
        F("idle"),
        F("idle", 1.03, 0.95, -2),
        F("idle", 1.06, 0.90, -3),
        F("attack_mid", 1.06, 0.93, 1),
        F("attack_full", 1.02, 0.92, 1, drops=spray(0)),
        F("attack_full", 1.05, 0.97, 2, drops=spray(1)),
        F("attack_full", 1.02, 1.01, 1, drops=spray(2)),
        F("attack_full", 1.00, 1.00, 0, drops=spray(3)),
        F("attack_full", 0.97, 0.94, -1, drops="hide"),
        F("attack_mid", 1.02, 1.00, -1, drops="hide"),
        F("idle", 1.03, 0.95, 0),
        F("idle", 0.99, 1.02, 0),
    ]


def build_heavy(poses):
    # Windup trembles with an alternating lean; the jump lifts with ty; impact drops fly out over 3 frames.
    return [
        F("idle", anchor="center"),
        F("idle", 1.04, 0.92, 1, anchor="center"),
        F("idle", 1.07, 0.87, -1, anchor="center"),
        F("heavy_windup", 1.10, 0.68, 1, anchor="center"),
        F("heavy_windup", 1.07, 0.78, -1, anchor="center"),
        F("heavy_windup", 1.04, 0.88, 1, anchor="center"),
        F("heavy_windup", 1.01, 0.96, -1, anchor="center"),
        F("heavy_windup", 1.00, 1.00, 1, anchor="center"),
        F("heavy_jump", 0.92, 1.12, anchor="center", ty=-8),
        F("heavy_jump", 1.00, 1.00, anchor="center", ty=-22),
        F("heavy_jump", 0.96, 1.06, anchor="center", ty=-9),
        F("heavy_impact", 1.04, 0.85, anchor="center", drops=spray(0)),
        F("heavy_impact", 1.00, 1.00, anchor="center", drops=spray(1)),
        F("heavy_impact", 1.00, 1.02, anchor="center", drops=spray(2)),
        F("heavy_recover", 1.15, 0.90, anchor="center"),
        F("heavy_recover", 1.04, 1.06, anchor="center"),
        F("idle", 1.02, 0.95, anchor="center"),
    ]


def build_death(poses):
    # death_hit is drawn about 10% bigger than idle; the squash on its first frame bridges the size.
    return [
        F("idle", anchor="center"),
        F("death_hit", 0.96, 0.92, -1, anchor="center", drops=spray(0)),
        F("death_hit", 1.00, 1.00, -1, anchor="center", drops=spray(2)),
        F("death_hit", 1.01, 0.97, 0, anchor="center", drops=spray(3)),
        F("death_hit", 1.03, 0.88, 0, anchor="center", drops="hide"),
        F("death_collapse", 0.92, 1.28, anchor="center"),
        F("death_collapse", 0.97, 1.12, anchor="center"),
        F("death_collapse", 1.00, 1.00, anchor="center"),
        F("death_collapse", 1.03, 0.88, anchor="center"),
        F("death_puddle", 0.97, 1.25, anchor="center"),
        F("death_puddle", 1.02, 1.00, anchor="center"),
        F("death_remains", 1.02, 1.20, anchor="center"),
        F("death_remains", 1.00, 1.00, anchor="center"),
    ]


def build_hurt(poses):
    # A short flinch for each hit: death_hit squashed near idle size, without the burst drops.
    return [
        F("idle", 1.04, 0.93, -1, anchor="center"),
        F("death_hit", 0.92, 0.85, -2, anchor="center", drops="hide"),
        F("death_hit", 0.90, 0.88, -1, anchor="center", drops="hide"),
        F("idle", 1.06, 0.88, -1, anchor="center"),
        F("idle", 1.03, 0.96, 0, anchor="center"),
        F("idle", 0.99, 1.02, 0, anchor="center"),
    ]


ANIMATIONS = [
    {"name": "idle", "fps": IDLE["fps"], "loop": True, "build": build_idle},
    {"name": "crawl", "fps": CRAWL["fps"], "loop": True, "build": build_crawl, "stride": crawl_stride},
    {"name": "attack", "fps": 12, "loop": False, "build": build_attack,
     "phases": {"windup": [0, 3], "strike": [4, 7], "recover": [8, 11]}},
    {"name": "heavy", "fps": 12, "loop": False, "build": build_heavy,
     "phases": {"windup": [0, 7], "air": [8, 10], "impact": [11, 13], "recover": [14, 16]}},
    {"name": "death", "fps": 10, "loop": False, "build": build_death, "phases": {"remains": [12, 12]}},
    {"name": "hurt", "fps": 14, "loop": False, "build": build_hurt, "phases": {"hit": [0, 2], "recover": [3, 5]}},
]


# The goo foot sits 4 art px into the floor surface, like the level props, so the slime does not float.
GROUND_SINK = 4

CONFIG = {"id": ID, "out_dir": OUT_DIR, "sources": SOURCES, "pivot": "bbox", "placement": PLACEMENT,
          "ground_sink": GROUND_SINK,
          "animations": ANIMATIONS}
