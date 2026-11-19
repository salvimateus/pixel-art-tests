"""Shared core of the skeletal cutout rig generator.

Part loading, baked far-side variants, atlas packing, forward kinematics, 2-bone leg IK, animation
baking, a Pillow renderer that matches rig-runtime.js, checks and the rig manifest writer.
Rig configs live in tools/rigs/ and the command is tools/make-rig-assets.py.

Conventions (model space): art px, origin on the ground line under the feet, +x forward (the
character faces +x), +y down. Rotations are degrees, positive is clockwise on screen. A bone
local transform is translate(x, y) then rotate(rot), applied after its parent.

Rig manifest contract (window.SPRITE_RIGS[id], data only):
  id, image                      atlas file, relative to the manifest folder
  frame {w, h}, pivot {x, y}     1x canvas that holds every frame of every animation; pivot is the
                                 model origin (ground contact point) in that canvas; no opaque pixel
                                 is below pivot.y
  parts {name: {x, y, w, h, pivot {x, y}}}   atlas rect; pivot is the joint point in part px
  bones [{name, parent, part, x, y, rot, z}] rest pose in parent space, parents before children;
                                 part may be null; higher z draws in front
  animations {name: {fps, frames, duration, loop, tracks, phases?, stride?}}
      tracks {bone: {rot|x|y: [[t, v], ...]}}  deltas added to the rest values, t in seconds,
                                 linear between keys; a loop wraps at duration, a one-shot holds the
                                 last key
      frames/fps                 frame i shows the pose at t = i / fps (the runtime can hold frames)
      z {bone: z}                optional draw order overrides for this animation
Config-only animation keys: pose(t, frame, rig) -> {bone: {rot, x, y}, "ik": {chain: target}};
  settle [first, last] frames whose lowest pixel is moved onto the ground (a body that lies down).
      phases {name: [first, last]}  inclusive frame ranges, same meaning as the sprite manifest
      stride                     art px the entity moves toward +x per cycle, so feet do not slide;
                                 measured on the planted feet when the config says "feet"
      bboxes                     one [x, y, w, h] per frame: the opaque bounds in frame coords (facing +x),
                                 baked from the tool renders, so a game can size the body box (a crouch)
                                 without pixel reads; the height of a frame is pivot.y - y
      hitboxes                   optional, one entry per frame: null or [{x, y, w, h}] in frame coords
                                 (facing +x), rects along the sword blade, baked from the sword bone
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sprite_core import ROOT, rnd  # noqa: E402

ATLAS_GAP = 2
ATLAS_MAX_W = 256
FRAME_MARGIN = 2
KEY_TOLERANCE = 0.05
REST_KNEE_SLACK = 0.6  # px of leg length kept bent at rest, so IK never locks the knee


# ---- 2D affine math (canvas order: x' = a x + c y + e, y' = b x + d y + f) ----

def mat_mul(m, n):
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2,
            a * e2 + c * f2 + e, b * e2 + d * f2 + f)


def mat_local(x, y, rot):
    r = math.radians(rot)
    co, si = math.cos(r), math.sin(r)
    return (co, si, -si, co, x, y)


def mat_apply(m, p):
    a, b, c, d, e, f = m
    return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)


def mat_inv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return (ia, ib, ic, id_, -(ia * e + ic * f), -(ib * e + id_ * f))


def mat_rot(m):
    return math.degrees(math.atan2(m[1], m[0]))


def dir_rot(vx, vy):
    """Rotation that turns the rest direction (0, 1) (a limb that hangs down) toward (vx, vy)."""
    return math.degrees(math.atan2(-vx, vy))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def lerp(a, b, u):
    return a + (b - a) * u


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def keyed(keys, t, ease=smooth):
    """Value at t from [(t, v), ...] sorted by t, eased between keys, held outside."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return lerp(v0, v1, ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0)
    return keys[-1][1]


# ---- parts ----

def load_part(path, erase=None):
    im = Image.open(path).convert("RGBA")
    alphas = set(im.getchannel("A").getdata())
    if not alphas <= {0, 255}:
        sys.exit(f"{path}: alpha must be 0 or 255, found {sorted(alphas - {0, 255})[:5]}")
    if erase:
        mask = Image.new("L", im.size, 0)
        dr = ImageDraw.Draw(mask)
        for poly in erase:
            dr.polygon(poly, fill=255)
        im.paste((0, 0, 0, 0), mask=mask)
    return im


def darken(im, factor, tint=(1.0, 1.0, 1.0)):
    out = im.copy()
    px = out.load()
    for y in range(out.height):
        for x in range(out.width):
            r, g, b, a = px[x, y]
            if a:
                px[x, y] = tuple(min(255, rnd(c * factor * t)) for c, t in zip((r, g, b), tint)) + (a,)
    return out


def pack_atlas(images):
    """Shelf packing, tallest first. Returns (atlas, {name: (x, y, w, h)})."""
    order = sorted(images, key=lambda n: (-images[n].height, n))
    rects, x, y, shelf_h, width = {}, ATLAS_GAP, ATLAS_GAP, 0, ATLAS_GAP
    max_w = max(ATLAS_MAX_W, max(im.width for im in images.values()) + 2 * ATLAS_GAP)
    for n in order:
        im = images[n]
        if x + im.width + ATLAS_GAP > max_w:
            x, y, shelf_h = ATLAS_GAP, y + shelf_h + ATLAS_GAP, 0
        rects[n] = (x, y, im.width, im.height)
        x += im.width + ATLAS_GAP
        shelf_h = max(shelf_h, im.height)
        width = max(width, x)
    atlas = Image.new("RGBA", (width, y + shelf_h + ATLAS_GAP), (0, 0, 0, 0))
    for n, (rx, ry, _, _) in rects.items():
        atlas.paste(images[n], (rx, ry))
    return atlas, rects


def sole_points(im, pivot):
    """Heel and toe points of a foot part on its lowest opaque row, relative to its pivot."""
    a = im.getchannel("A")
    bottom = a.getbbox()[3] - 1
    xs = [x for x in range(im.width) if a.getpixel((x, bottom))]
    sole_y = bottom + 1 - pivot[1]
    return (min(xs) - pivot[0], sole_y), (max(xs) + 1 - pivot[0], sole_y)


# ---- skeleton ----

class Rig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.parts = {}
        for name, p in cfg["parts"].items():
            self.parts[name] = load_part(os.path.join(ROOT, p["src"]), p.get("erase"))
        for name, v in cfg.get("variants", {}).items():
            self.parts[name] = darken(self.parts[v["from"]], v["factor"], tuple(v.get("tint", (1, 1, 1))))
        self.pivots = {n: tuple(cfg["parts"][n]["pivot"]) if n in cfg["parts"]
                       else tuple(cfg["parts"][cfg["variants"][n]["from"]]["pivot"]) for n in self.parts}
        self.bones = [dict(b) for b in cfg["bones"]]
        self.index = {b["name"]: i for i, b in enumerate(self.bones)}
        for b in self.bones:
            if b["parent"] and self.index[b["parent"]] > self.index[b["name"]]:
                sys.exit(f"bone {b['name']}: its parent must come first")
        self._bind_rest()

    def _bind_rest(self):
        """Rest pose from placements measured on the master. A rest entry is either
        {"center": (x, y), "rot"}: the part center in master px, or {"joint": (x, y), "rot"}: the point
        in the parent part px where this bone pivot sits. "rot" is the world rotation. Then both feet
        are planted flat on one ground row by IK, and the origin moves to the ground contact point:
        the ground line under the soles, x at the center of the soles."""
        rest = self.cfg["rest"]
        mats = {}
        for b in self.bones:
            r = rest.get(b["name"])
            parent = mats[b["parent"]] if b["parent"] else (1, 0, 0, 1, 0, 0)
            if r is None:
                if b["parent"] is None:
                    b["x"] = b["y"] = b["rot"] = 0.0
                    mats[b["name"]] = (1, 0, 0, 1, 0, 0)
                    continue
                sys.exit(f"bone {b['name']}: no rest placement")
            wrot = r.get("rot", 0.0)
            if "center" in r:
                im = self.parts[b["part"]]
                px, py = self.pivots[b["part"]]
                wx, wy = mat_apply(mat_local(r["center"][0], r["center"][1], wrot),
                                   (px - im.width / 2, py - im.height / 2))
                b["x"], b["y"] = mat_apply(mat_inv(parent), (wx, wy))
            else:
                ppx, ppy = self.pivots[self.bone(b["parent"])["part"]]
                b["x"], b["y"] = r["joint"][0] - ppx, r["joint"][1] - ppy
            b["rot"] = wrap180(wrot - mat_rot(parent))
            mats[b["name"]] = mat_mul(parent, mat_local(b["x"], b["y"], b["rot"]))

        chains = self.cfg["ik"]
        soles = {}
        for c, ch in chains.items():
            foot = self.bone(ch["foot"])
            soles[c] = sole_points(self.parts[foot["part"]], self.pivots[foot["part"]])
        rest_w = self.world({})
        # The lowest current sole, raised when a leg cannot reach it with a small knee bend left.
        reach = []
        for c, ch in chains.items():
            hip = mat_apply(rest_w[ch["thigh"]], (0, 0))
            shin, foot = self.bone(ch["shin"]), self.bone(ch["foot"])
            ln = math.hypot(shin["x"], shin["y"]) + math.hypot(foot["x"], foot["y"])
            dx = mat_apply(rest_w[ch["foot"]], (0, 0))[0] - hip[0]
            reach.append(hip[1] + math.sqrt(max(0.0, (ln - REST_KNEE_SLACK) ** 2 - dx * dx)) + soles[c][0][1])
        lowest = max(mat_apply(rest_w[chains[c]["foot"]], (0, 0))[1] + soles[c][0][1] for c in chains)
        ground = math.floor(min(lowest, min(reach)))
        xs = []
        for c, ch in chains.items():
            ax = mat_apply(rest_w[ch["foot"]], (0, 0))[0]
            pose = {}
            self.solve_leg(c, pose, {"x": ax, "y": ground - soles[c][0][1]})
            for name, d in pose.items():
                self.bone(name)["rot"] += d["rot"]
            xs += [ax + soles[c][0][0], ax + soles[c][1][0]]
        cx = rnd((min(xs) + max(xs)) / 2)
        for b in self.bones:
            if b["parent"] == self.bones[0]["name"]:
                b["x"] -= cx
                b["y"] -= ground
        for b in self.bones:
            b["x"], b["y"], b["rot"] = round(b["x"], 3), round(b["y"], 3), round(b["rot"], 3)
        self.master_offset = (cx, ground)
        img, _, oy = self.render(self.world({}))
        bb = img.getchannel("A").getbbox()
        self.body_height = bb[3] - bb[1]
        rest = self.world({})
        self.foot_rest = {c: mat_apply(rest[chains[c]["foot"]], (0, 0))[0] for c in chains}
        self.hip_rest = {c: mat_apply(rest[chains[c]["thigh"]], (0, 0)) for c in chains}
        self.leg_len = {c: math.hypot(self.bone(chains[c]["shin"])["x"], self.bone(chains[c]["shin"])["y"])
                        + math.hypot(self.bone(chains[c]["foot"])["x"], self.bone(chains[c]["foot"])["y"])
                        for c in chains}

    def bone(self, name):
        return self.bones[self.index[name]]

    def world(self, pose):
        """pose {bone: {rot, x, y}} deltas from rest. Returns {bone: matrix}."""
        mats = {}
        for b in self.bones:
            d = pose.get(b["name"], {})
            local = mat_local(b["x"] + d.get("x", 0.0), b["y"] + d.get("y", 0.0), b["rot"] + d.get("rot", 0.0))
            mats[b["name"]] = mat_mul(mats[b["parent"]], local) if b["parent"] else local
        return mats

    # ---- IK ----

    def ankle_target(self, chain, target):
        """Ankle model position for an IK target, after the heel or toe roll."""
        foot = self.bone(self.cfg["ik"][chain]["foot"])
        heel, toe = sole_points(self.parts[foot["part"]], self.pivots[foot["part"]])
        pitch = target.get("pitch", 0.0)
        ax = target["x"]
        ay = target["y"] if "y" in target else -heel[1] - target.get("lift", 0.0)
        if pitch and "y" not in target:
            pt = toe if pitch > 0 else heel
            moved = mat_apply(mat_local(0, 0, pitch), pt)
            ax, ay = ax + pt[0] - moved[0], ay + pt[1] - moved[1]
        return ax, ay

    def hip(self, pose, chain):
        thigh = self.bone(self.cfg["ik"][chain]["thigh"])
        m = self.world(pose)[thigh["parent"]]
        d = pose.get(thigh["name"], {})
        return mat_apply(m, (thigh["x"] + d.get("x", 0.0), thigh["y"] + d.get("y", 0.0)))

    def reach_drop(self, pose, targets, reach=0.995):
        """Px the pelvis must move down so every leg reaches its target with `reach` of its length."""
        need = 0.0
        for chain, tgt in targets.items():
            ax, ay = self.ankle_target(chain, tgt)
            hx, hy = self.hip(pose, chain)
            r = reach * self.leg_len[chain]
            dx = ax - hx
            if abs(dx) >= r:
                continue
            need = max(need, (ay - math.sqrt(r * r - dx * dx)) - hy)
        return need

    def straight_pitch(self, pose, chain, target, reach=0.995, limit=75.0):
        """Toe roll in [0, limit] at which the leg reaches exactly `reach` of its length, so a back foot
        rolls onto its toe and the leg stays straight. 0 when the flat foot is already within reach."""
        hx, hy = self.hip(pose, chain)
        r = reach * self.leg_len[chain]

        def dist(p):
            ax, ay = self.ankle_target(chain, dict(target, pitch=p))
            return math.hypot(ax - hx, ay - hy)

        lo, hi = 0.0, limit
        if dist(lo) <= r:
            return lo
        if dist(hi) > r:
            return hi
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if dist(mid) > r else (lo, mid)
        return hi

    def solve_leg(self, chain, pose, target):
        """Bake one leg. target {x, lift, pitch} or {x, y, pitch}: x is the ankle x when the foot is flat,
        lift the sole height above the ground, pitch the foot world rotation (+ is toe down: roll on the
        toe; - is toe up: roll on the heel). With y, the ankle goes exactly to (x, y), with no roll.
        Writes rot deltas for thigh, shin and foot into pose. Returns the px the target is out of reach."""
        c = self.cfg["ik"][chain]
        thigh, shin, foot = self.bone(c["thigh"]), self.bone(c["shin"]), self.bone(c["foot"])
        foot_rest_rot = 0.0
        pitch = target.get("pitch", 0.0)
        ax, ay = self.ankle_target(chain, target)

        mats = self.world(pose)
        parent_m = mats[thigh["parent"]]
        hip = mat_apply(mat_mul(parent_m, mat_local(thigh["x"] + pose.get(thigh["name"], {}).get("x", 0),
                                                     thigh["y"] + pose.get(thigh["name"], {}).get("y", 0), 0)), (0, 0))
        k = (shin["x"], shin["y"])
        a = (foot["x"], foot["y"])
        l1, l2 = math.hypot(*k), math.hypot(*a)
        dx, dy = ax - hip[0], ay - hip[1]
        d = math.hypot(dx, dy)
        reach = l1 + l2 - 1e-3
        short = d - reach
        d = max(abs(l1 - l2) + 1e-3, min(reach, d))
        base = dir_rot(dx, dy)
        alpha = math.degrees(math.acos(max(-1, min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)))))
        thigh_dir = base - alpha  # knee bends forward (+x)
        knee = (hip[0] - l1 * math.sin(math.radians(thigh_dir)), hip[1] + l1 * math.cos(math.radians(thigh_dir)))
        shin_dir = dir_rot(ax - knee[0], ay - knee[1])
        thigh_w = thigh_dir - dir_rot(*k)
        shin_w = shin_dir - dir_rot(*a)
        foot_w = foot_rest_rot + pitch
        parent_rot = mat_rot(parent_m)
        for bone, local in ((thigh, thigh_w - parent_rot), (shin, shin_w - thigh_w), (foot, foot_w - shin_w)):
            pose.setdefault(bone["name"], {})["rot"] = wrap180(local - bone["rot"])
        return max(0.0, short)

    # ---- rendering (matches rig-runtime.js: nearest neighbor, 1x) ----

    def render(self, mats, pad=0, size=None, origin=None, z=None):
        if size is None:
            size = (640, 640)
            origin = (320, 320)
        ox, oy = origin
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        z = z or {}
        for b in sorted((b for b in self.bones if b.get("part")), key=lambda b: z.get(b["name"], b["z"])):
            im = self.parts[b["part"]]
            px, py = self.pivots[b["part"]]
            m = mat_mul((1, 0, 0, 1, ox, oy), mat_mul(mats[b["name"]], (1, 0, 0, 1, -px, -py)))
            inv = mat_inv(m)
            a, bb, c, d, e, f = inv
            layer = im.transform(size, Image.AFFINE, (a, c, e, bb, d, f), resample=Image.NEAREST)
            canvas.alpha_composite(layer)
        return canvas, ox, oy


# ---- animation baking ----

def bake(rig, anim):
    """Sample the config pose function at every frame, solve the legs, apply followers.
    Returns (frames [pose], warnings)."""
    fps, n = anim["fps"], anim["frames"]
    frames, warn = [], []
    for i in range(n):
        t = i / fps
        spec = anim["pose"](t, i, rig)
        pose = {k: dict(v) for k, v in spec.items() if k != "ik"}
        for chain, target in spec.get("ik", {}).items():
            short = rig.solve_leg(chain, pose, target)
            if short > 0.5:
                warn.append(f"{anim['name']} frame {i}: {chain} misses its target by {short:.1f} px")
        frames.append(pose)
    for fol in rig.cfg.get("followers", []):
        # A follower copies part of the local rot delta of a sibling bone, some time later.
        src, dst = fol["source"], fol["bone"]
        rots = [p.get(src, {}).get("rot", 0.0) for p in frames]
        lag = fol["delay"] * fps
        for i, pose in enumerate(frames):
            j = (i - lag) % n if anim["loop"] else max(0.0, i - lag)
            j0 = int(math.floor(j))
            j1 = (j0 + 1) % n if anim["loop"] else min(n - 1, j0 + 1)
            v = fol["gain"] * lerp(rots[j0], rots[j1], j - j0) + fol.get("offset", 0.0)
            v = max(fol.get("min", -180), min(fol.get("max", 180), v))
            d = pose.setdefault(dst, {})
            d["rot"] = d.get("rot", 0.0) + v
    if anim.get("settle"):
        # Frames that lie on the ground: move the root so the lowest pixel is on the row above the ground line.
        first, last = anim["settle"]
        for i in range(first, last + 1):
            img, ox, oy = rig.render(rig.world(frames[i]), z=anim.get("z"))
            bottom = img.getchannel("A").getbbox()[3] - oy
            d = frames[i].setdefault(rig.bones[0]["name"], {})
            d["y"] = d.get("y", 0.0) - bottom
    return frames, warn


def tracks_from(frames, fps, loop):
    n = len(frames)
    bones = sorted({b for p in frames for b in p})
    out = {}
    for b in bones:
        chans = {}
        for ch in ("rot", "x", "y"):
            vals = [round(p.get(b, {}).get(ch, 0.0), 2) for p in frames]
            if all(abs(v) < 0.005 for v in vals):
                continue
            keys = [[round(i / fps, 4), v] for i, v in enumerate(vals)]
            keys.append([round(n / fps, 4), vals[0] if loop else vals[-1]])
            chans[ch] = simplify(keys)
        if chans:
            out[b] = chans
    return out


def simplify(keys):
    if len(keys) <= 2:
        return keys
    out = [keys[0]]
    for k in range(1, len(keys) - 1):
        (t0, v0), (t1, v1), (t2, v2) = out[-1], keys[k], keys[k + 1]
        if abs(v0 + (v2 - v0) * (t1 - t0) / (t2 - t0) - v1) > KEY_TOLERANCE:
            out.append(keys[k])
    out.append(keys[-1])
    return out


# ---- gait helpers for configs ----

def gait_foot(u, stride, duty, lift, center=0.0, toe_off=0.0, heel_strike=0.0, lift_peak=0.4, swing_ease=1.0):
    """Foot target for a cycle phase u in [0, 1): stance first (u < duty), then swing.
    In stance the flat-foot ankle x moves back at stride per cycle, the ground speed of the entity, so the
    planted foot keeps its world position. The stance starts on the heel (toe up by heel_strike degrees)
    and ends on the toe (toe_off degrees); both roll on the contact point. In the swing the foot rises to
    `lift` at `lift_peak` of the swing, then lowers to the next heel strike. swing_ease < 1 brings the foot
    forward early in the swing (a run drives the knee forward instead of kicking the heel back)."""
    u %= 1.0
    half = stride * duty / 2
    if u < duty:
        s = u / duty
        x = center + half - stride * duty * s
        pitch = -heel_strike * (1 - smooth(s / 0.2)) + toe_off * smooth((s - 0.55) / 0.45)
        return {"x": x, "lift": 0.0, "pitch": pitch}
    s = (u - duty) / (1 - duty)
    x = center - half + 2 * half * smooth(s ** swing_ease)
    if s < lift_peak:
        h = lift * math.sin(0.5 * math.pi * s / lift_peak)
    else:
        h = lift * math.cos(0.5 * math.pi * (s - lift_peak) / (1 - lift_peak))
    pitch = toe_off * (1 - smooth(s / 0.4)) - heel_strike * smooth((s - 0.55) / 0.45)
    return {"x": x, "lift": max(0.0, h), "pitch": pitch}


# ---- outputs ----

def build(cfg, out_dir, contact_dir=None, contact_scale=4):
    rig = Rig(cfg)
    anims_out, rendered, warnings = {}, {}, []
    for anim in cfg["animations"]:
        frames, warn = bake(rig, anim)
        warnings += warn
        rendered[anim["name"]] = frames
        entry = {"fps": anim["fps"], "frames": anim["frames"], "duration": round(anim["frames"] / anim["fps"], 4),
                 "loop": anim["loop"], "tracks": tracks_from(frames, anim["fps"], anim["loop"])}
        if anim.get("phases"):
            entry["phases"] = anim["phases"]
        if anim.get("stride") == "feet":
            entry["stride"] = measure_stride(rig, anim, frames)
            print(f"{anim['name']}: stride {entry['stride']} px per cycle, measured on the planted feet")
        elif anim.get("stride") is not None:
            entry["stride"] = anim["stride"]
        if anim.get("hitbox"):
            entry["hitboxes"] = blade_hitboxes(rig, anim, frames)
        if anim.get("z"):
            entry["z"] = anim["z"]
        anims_out[anim["name"]] = entry

    # Frame box over every frame of every animation, from the real renders.
    size, origin = (480, 400), (240, 300)
    x0 = y0 = 10 ** 9
    x1 = y1 = -10 ** 9
    images = {}
    zmap = {a["name"]: a.get("z") for a in cfg["animations"]}
    boxes = {}
    for name, frames in rendered.items():
        images[name] = []
        for pose in frames:
            img, ox, oy = rig.render(rig.world(pose), size=size, origin=origin, z=zmap[name])
            images[name].append(img)
            bb = img.getchannel("A").getbbox()
            boxes.setdefault(name, []).append((bb[0] - ox, bb[1] - oy, bb[2] - bb[0], bb[3] - bb[1]))
            if bb[3] - oy > 0:
                warnings.append(f"{name} frame {len(images[name]) - 1}: pixels reach {bb[3] - oy} px below the ground line")
            x0, y0, x1, y1 = min(x0, bb[0] - ox), min(y0, bb[1] - oy), max(x1, bb[2] - ox), max(y1, bb[3] - oy)
    piv = {"x": -x0 + FRAME_MARGIN, "y": -y0 + FRAME_MARGIN}
    fw, fh = x1 - x0 + 2 * FRAME_MARGIN, piv["y"] + max(0, y1)

    warnings += check_feet(rig, cfg, rendered, {n: a.get("stride", 0) for n, a in anims_out.items()})
    for name, entry in anims_out.items():
        entry["bboxes"] = [[bx + piv["x"], by + piv["y"], bw, bh] for bx, by, bw, bh in boxes[name]]
        for f, hbs in enumerate(entry.get("hitboxes", [])):
            for r in hbs or []:
                r["x"] += piv["x"]
                r["y"] += piv["y"]

    atlas, rects = pack_atlas(rig.parts)
    os.makedirs(out_dir, exist_ok=True)
    name = cfg["name"]
    atlas.save(os.path.join(out_dir, name + ".png"))
    manifest = {
        "id": cfg["id"], "image": name + ".png",
        "frame": {"w": fw, "h": fh}, "pivot": piv,
        "parts": {n: {"x": r[0], "y": r[1], "w": r[2], "h": r[3],
                      "pivot": {"x": rig.pivots[n][0], "y": rig.pivots[n][1]}} for n, r in sorted(rects.items())},
        "bones": [{"name": b["name"], "parent": b["parent"], "part": b.get("part"), "x": b["x"], "y": b["y"],
                   "rot": b["rot"], "z": b["z"]} for b in rig.bones],
        "animations": anims_out,
    }
    text = json.dumps(manifest, separators=(",", ":"))
    with open(os.path.join(out_dir, name + ".json"), "w") as fh_:
        fh_.write(text + "\n")
    with open(os.path.join(out_dir, name + ".manifest.js"), "w") as fh_:
        fh_.write(f'(window.SPRITE_RIGS = window.SPRITE_RIGS || {{}})["{cfg["id"]}"] = {text};\n')
    print(f"Wrote {os.path.join(out_dir, name)}.png/.json/.manifest.js atlas {atlas.width}x{atlas.height}, "
          f"frame {fw}x{fh}, pivot ({piv['x']}, {piv['y']})")

    if contact_dir:
        os.makedirs(contact_dir, exist_ok=True)
        for an, imgs in images.items():
            write_strip(os.path.join(contact_dir, f"rigdev-{cfg['id']}-{an}.png"), imgs, origin, x0, y0, x1, y1,
                        contact_scale)
        write_rest_check(rig, cfg, os.path.join(contact_dir, f"rigdev-{cfg['id']}-rest-check.png"), contact_scale)
    for w in warnings:
        print("WARN", w)
    return manifest


def planted_ankles(rig, anim, frames, chain):
    """Per frame: the ankle model x when the target is a flat planted foot, else None."""
    out = []
    for i, pose in enumerate(frames):
        tgt = anim["pose"](i / anim["fps"], i, rig).get("ik", {}).get(chain)
        if not tgt or tgt.get("lift", 0) > 0 or tgt.get("pitch", 0) or "y" in tgt:
            out.append(None)
        else:
            out.append(mat_apply(rig.world(pose)[rig.cfg["ik"][chain]["foot"]], (0, 0))[0])
    return out


def measure_stride(rig, anim, frames):
    """Art px per cycle: the speed of the planted feet times the cycle duration."""
    steps = []
    for chain in rig.cfg["ik"]:
        xs = planted_ankles(rig, anim, frames, chain)
        steps += [a - b for a, b in zip(xs, xs[1:]) if a is not None and b is not None]
    if not steps:
        sys.exit(f"{anim['name']}: no planted foot to measure the stride")
    return round(sum(steps) / len(steps) * anim["frames"], 1)


def blade_hitboxes(rig, anim, frames):
    """Per frame, rects along a bone segment (the blade) in model px, or None outside the listed frames.
    The caller moves them to frame coords. Never read from pixels: the runtime cannot read a tainted canvas."""
    hb = anim["hitbox"]
    part = rig.bone(hb["bone"])["part"]
    piv = rig.pivots[part]
    alpha = rig.parts[part].getchannel("A")
    tip = max(x for x in range(alpha.width) if alpha.getpixel((x, piv[1]))) + 1
    (ax, ay), (bx, by) = (hb["from_x"], piv[1]), (tip, piv[1])
    out = []
    for i, pose in enumerate(frames):
        if i not in hb["frames"]:
            out.append(None)
            continue
        m = rig.world(pose)[hb["bone"]]
        rects = []
        n, pad = hb.get("slices", 4), hb.get("pad", 2)
        for k in range(n):
            pts = [mat_apply(m, (ax - piv[0] + (bx - ax) * u, ay - piv[1] + (by - ay) * u)) for u in (k / n, (k + 1) / n)]
            x0, x1 = min(p[0] for p in pts) - pad, max(p[0] for p in pts) + pad
            y0, y1 = min(p[1] for p in pts) - pad, max(p[1] for p in pts) + pad
            rects.append({"x": math.floor(x0), "y": math.floor(y0), "w": math.ceil(x1) - math.floor(x0),
                          "h": math.ceil(y1) - math.floor(y0)})
        out.append(rects)
    return out


def check_feet(rig, cfg, rendered, strides):
    """A planted flat foot must keep its world x across frames, given the stride (baked ankles, not targets)."""
    out = []
    for anim in cfg["animations"]:
        stride = strides.get(anim["name"]) or 0
        fps, n = anim["fps"], anim["frames"]
        speed = stride / (n / fps)
        for chain in cfg["ik"]:
            xs = planted_ankles(rig, anim, rendered[anim["name"]], chain)
            for i in range(1, n):
                if xs[i] is None or xs[i - 1] is None:
                    continue
                slide = (xs[i] - xs[i - 1]) + speed / fps
                if abs(slide) > 0.51:
                    out.append(f"{anim['name']} frame {i}: {chain} slides {slide:+.2f} px")
    return out


def write_strip(path, imgs, origin, x0, y0, x1, y1, scale):
    ox, oy = origin
    w, h = x1 - x0 + 4, y1 - y0 + 6
    strip = Image.new("RGBA", (w * len(imgs), h), (40, 40, 48, 255))
    dr = ImageDraw.Draw(strip)
    for k, img in enumerate(imgs):
        cell = img.crop((ox + x0 - 2, oy + y0 - 2, ox + x0 - 2 + w, oy + y0 - 2 + h))
        strip.alpha_composite(cell, (k * w, 0))
        gy = -y0 + 2
        dr.line([(k * w, gy), (k * w + w - 1, gy)], fill=(90, 200, 90, 255))
        dr.line([(k * w + 2 - x0, gy - 2), (k * w + 2 - x0, gy + 2)], fill=(255, 220, 0, 255))
        dr.line([(k * w, 0), (k * w, h)], fill=(70, 70, 80, 255))
    strip.resize((strip.width * scale, strip.height * scale), Image.NEAREST).save(path)


def write_rest_check(rig, cfg, path, scale):
    """Rest pose next to the master and a 50% overlay."""
    master = Image.open(os.path.join(ROOT, cfg["master"])).convert("RGBA")
    cx, ground = rig.master_offset
    size = (master.width + 20, master.height + 20)
    origin = (cx + 10, ground + 10)
    rest, _, _ = rig.render(rig.world({}), size=size, origin=origin)
    m = Image.new("RGBA", size, (0, 0, 0, 0))
    m.paste(master, (10, 10))
    over = Image.blend(Image.alpha_composite(Image.new("RGBA", size, (40, 40, 48, 255)), m),
                       Image.alpha_composite(Image.new("RGBA", size, (40, 40, 48, 255)), rest), 0.5)
    out = Image.new("RGBA", (size[0] * 3, size[1]), (40, 40, 48, 255))
    out.alpha_composite(m, (0, 0))
    out.alpha_composite(rest, (size[0], 0))
    out.alpha_composite(over, (size[0] * 2, 0))
    out.resize((out.width * scale, out.height * scale), Image.NEAREST).save(path)
