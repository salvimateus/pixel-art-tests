"""Black knight rig: parts, skeleton, rest pose and animations as data.

Build: python3 tools/make-rig-assets.py black-knight
Rest placements are measured on the master knight (piece-01, same art px scale as the parts),
in master px: "center" is the part center and "rot" its rotation (+ is clockwise).
Pose functions return rot/x/y deltas from rest per bone and leg IK targets per chain.
"""
import math

from rig_core import gait_foot, keyed, mat_apply, mat_rot, smooth, sole_points, wrap180

ID = "black-knight"
OUT_DIR = "assets/black-knight"
P = "heroes/black-knight/parts/"

PARTS = {
    "helmet": {"src": P + "piece-02.png", "pivot": (13, 31)},
    "torso": {"src": P + "piece-03.png", "pivot": (15, 37)},
    "hips": {"src": P + "piece-04.png", "pivot": (15, 3)},
    "tabard_front": {"src": P + "piece-05.png", "pivot": (13, 2)},
    "tabard_back": {"src": P + "piece-06.png", "pivot": (15, 2)},
    "upper_arm": {"src": P + "piece-07.png", "pivot": (10, 6)},
    "forearm": {"src": P + "piece-08.png", "pivot": (6, 2)},
    "gauntlet": {"src": P + "piece-09.png", "pivot": (3, 6)},
    "thigh": {"src": P + "piece-10.png", "pivot": (8, 4)},
    "shin": {"src": P + "piece-11.png", "pivot": (7, 2)},
    "sabaton": {"src": P + "piece-12.png", "pivot": (8, 3)},
    "sword": {"src": P + "piece-13.png", "pivot": (9, 9)},
    "shield": {"src": P + "piece-14.png", "pivot": (13, 22)},
}

SWORD_GUARD_X = 15  # sword part px where the blade starts, after the cross guard

FAR_SHADE = {"factor": 0.62, "tint": (0.96, 0.98, 1.08)}
VARIANTS = {f"{n}_far": dict(FAR_SHADE, **{"from": n}) for n in ("upper_arm", "forearm", "thigh", "shin", "sabaton")}

# Parents come before children. Higher z draws in front.
BONES = [
    {"name": "root", "parent": None, "part": None, "z": 0},
    {"name": "pelvis", "parent": "root", "part": "hips", "z": 52},
    {"name": "torso", "parent": "pelvis", "part": "torso", "z": 60},
    {"name": "helmet", "parent": "torso", "part": "helmet", "z": 70},
    {"name": "arm_far", "parent": "torso", "part": "upper_arm_far", "z": 10},
    {"name": "forearm_far", "parent": "arm_far", "part": "forearm_far", "z": 12},
    {"name": "shield", "parent": "forearm_far", "part": "shield", "z": 80},
    {"name": "arm_near", "parent": "torso", "part": "upper_arm", "z": 90},
    {"name": "forearm_near", "parent": "arm_near", "part": "forearm", "z": 92},
    {"name": "hand_near", "parent": "forearm_near", "part": "gauntlet", "z": 96},
    {"name": "sword", "parent": "hand_near", "part": "sword", "z": 94},
    {"name": "tabard_back", "parent": "pelvis", "part": "tabard_back", "z": 15},
    {"name": "thigh_far", "parent": "pelvis", "part": "thigh_far", "z": 20},
    {"name": "shin_far", "parent": "thigh_far", "part": "shin_far", "z": 17},
    {"name": "foot_far", "parent": "shin_far", "part": "sabaton_far", "z": 18},
    {"name": "thigh_near", "parent": "pelvis", "part": "thigh", "z": 50},
    {"name": "shin_near", "parent": "thigh_near", "part": "shin", "z": 47},
    {"name": "foot_near", "parent": "shin_near", "part": "sabaton", "z": 48},
    {"name": "tabard_front", "parent": "pelvis", "part": "tabard_front", "z": 55},
]

REST = {
    "pelvis": {"center": (22, 82), "rot": -8},
    "torso": {"joint": (22, -2), "rot": -8},
    "helmet": {"joint": (21, 2), "rot": 0},
    "arm_far": {"joint": (27, 8), "rot": -15},
    "forearm_far": {"joint": (7, 30), "rot": -75},
    "shield": {"center": (51, 58), "rot": 0},
    "arm_near": {"center": (12, 49), "rot": 12},
    "forearm_near": {"joint": (7, 30), "rot": -24},
    "hand_near": {"center": (16, 82), "rot": -16},
    "sword": {"center": (45, 76), "rot": -10},
    "tabard_back": {"center": (27, 90), "rot": 0},
    "tabard_front": {"center": (27, 89), "rot": 8},
    "thigh_far": {"joint": (8, 19), "rot": 10},
    "shin_far": {"joint": (8, 26), "rot": 12},
    "foot_far": {"joint": (7, 30), "rot": 0},
    "thigh_near": {"joint": (25, 19), "rot": -8},
    "shin_near": {"joint": (8, 26), "rot": -4},
    "foot_near": {"joint": (7, 30), "rot": 0},
}

IK = {
    "near": {"thigh": "thigh_near", "shin": "shin_near", "foot": "foot_near"},
    "far": {"thigh": "thigh_far", "shin": "shin_far", "foot": "foot_far"},
}

FOLLOWERS = [
    {"bone": "tabard_front", "source": "thigh_near", "gain": 0.45, "delay": 0.08, "min": -25, "max": 12},
    {"bone": "tabard_back", "source": "thigh_far", "gain": 0.45, "delay": 0.08, "min": -25, "max": 12},
]


def planted(rig, dx=0.0):
    return {c: {"x": rig.foot_rest[c] + dx} for c in IK}


def K(i, keys):
    """Value at frame i from {frame: value}, eased between keys."""
    return keyed(sorted(keys.items()), i)


def hip(rig, pose, chain):
    m = rig.world(pose)[IK[chain]["thigh"]]
    return m[4], m[5]


def aim(rig, pose, bone, world, via=None):
    """Turn `via` (default: the bone itself) so that `bone` has this world rotation."""
    via = via or bone
    cur = mat_rot(rig.world(pose)[bone])
    d = pose.setdefault(via, {})
    d["rot"] = d.get("rot", 0.0) + wrap180(world - cur)


def idle(t, i, rig):
    b = 0.5 - 0.5 * math.cos(2 * math.pi * t / 2.0)  # 0 exhale, 1 inhale
    sway = math.sin(2 * math.pi * t / 2.0)
    return {
        "torso": {"y": -b, "rot": -0.6 * b},
        "helmet": {"y": -0.4 * b, "rot": 0.8 * b},
        "arm_near": {"rot": 0.8 * b},
        "hand_near": {"rot": -2.0 * b},
        "arm_far": {"rot": -0.6 * b},
        "tabard_front": {"rot": 1.5 * sway},
        "tabard_back": {"rot": 1.5 * math.sin(2 * math.pi * t / 2.0 - 0.6)},
        "ik": planted(rig),
    }


def rest_rot(rig, bone):
    return mat_rot(rig.world({})[bone])


def legs(rig, pose, ik, reach=0.995):
    """Set the IK targets and lower the pelvis only as much as a leg needs to reach its target."""
    p = pose.setdefault("pelvis", {})
    p["y"] = p.get("y", 0.0) + rig.reach_drop(pose, ik, reach)
    pose["ik"] = ik
    return pose


def gait_pose(u, g, rig, upper, swing_pitch=True):
    """Legs of a walk or a run at cycle phase u. The pelvis starts at the passing height and drops only
    when a front stance leg cannot reach, plus a small dip just after each contact: the bob comes from
    the leg geometry. A back stance foot rolls onto its toe just enough to keep its leg straight; the
    swing starts from that toe angle."""
    stride = g["stride_steps"] * 2 * WALK["step"] * rig.body_height
    pose = upper(u)
    pel = pose.setdefault("pelvis", {})
    pel.setdefault("x", 0.0)
    # Level the hips, so both legs get the same reach; the torso is aimed in world space anyway.
    pel["rot"] = pel.get("rot", 0.0) - rest_rot(rig, "pelvis")
    pel["y"] = 0.0
    ankle_y = rig.ankle_target("near", {"x": 0})[1]
    pel["y"] = ankle_y - g["pass"] * rig.leg_len["near"] - rig.hip(pose, "near")[1]
    pel["y"] += g["dip"] * (0.5 + 0.5 * math.cos(4 * math.pi * (u - g["down"])))
    ik, phases = {}, {}
    for chain, phase in (("near", 0.0), ("far", 0.5)):
        center = rig.hip_rest[chain][0] + pel["x"] + g["center"]
        phases[chain] = (u + phase) % 1.0
        ik[chain] = gait_foot(u + phase, stride, g["duty"], g["lift"], center, 0.0, g["heel"], g["lift_peak"],
                              g.get("swing_ease", 1.0))
    front = {c: t for c, t in ik.items() if phases[c] < g["duty"] / 2}
    pel["y"] += rig.reach_drop(pose, front, g["reach"])
    for chain, v in phases.items():
        t = ik[chain]
        if g["duty"] / 2 <= v < g["duty"]:
            t["pitch"] = rig.straight_pitch(pose, chain, t, g["reach"])
        elif v >= g["duty"] and swing_pitch:
            s = (v - g["duty"]) / (1 - g["duty"])
            end = gait_pose((u - (v - g["duty"]) - 1e-4) % 1.0, g, rig, upper, False)["ik"][chain]["pitch"]
            t["pitch"] += end * (1 - smooth(s / 0.4))
    pose["ik"] = ik
    return pose


# Classic walk: contact, down, passing, up, then the same for the other leg. The step is a fraction of
# the body height; the manifest stride is measured on the planted feet.
WALK = {"step": 0.3, "stride_steps": 1.0, "duty": 0.52, "lift": 4.5, "lift_peak": 0.3, "heel": 18,
        "pass": 0.98, "reach": 0.995, "dip": 0.6, "down": 0.1, "center": 1.0, "lean": 2.5}


def walk(t, i, rig):
    w, u = WALK, t / ANIM["walk"]["duration"]

    def upper(u):
        c = math.cos(2 * math.pi * u)
        pose = {"pelvis": {"rot": 0.6 * c},
                "arm_near": {"rot": 6.0 * c},
                "forearm_near": {"rot": -3.0 * math.cos(2 * math.pi * u - 0.6)},
                "hand_near": {"rot": 2.0 * math.cos(2 * math.pi * u - 0.9)},
                "arm_far": {"rot": -1.5 * c}}
        aim(rig, pose, "torso", w["lean"] + 0.6 * math.cos(4 * math.pi * (u - w["down"])))
        aim(rig, pose, "helmet", rest_rot(rig, "helmet") - 1.0)
        return pose

    return gait_pose(u, w, rig, upper)


# Sprint: a stride that is a multiple of the walk stride, a short flight, a high swing knee, a strong lean.
SPRINT = {"stride_steps": 1.8, "duty": 0.34, "lift": 22, "lift_peak": 0.5, "swing_ease": 0.6, "heel": 0, "pass": 0.93,
          "reach": 0.985, "dip": 2.0, "down": 0.17, "center": -10.0, "lean": 18, "head": 14, "sword": 24}


def sprint(t, i, rig):
    w, u = SPRINT, t / ANIM["sprint"]["duration"]

    def upper(u):
        c = math.cos(2 * math.pi * u)
        pose = {"pelvis": {"x": 5.0, "rot": 10.0 + 1.5 * c},
                "arm_near": {"rot": -4.0 + 20.0 * c},
                "forearm_near": {"rot": -30.0 + 10.0 * math.cos(2 * math.pi * u + 0.5)},
                "arm_far": {"rot": -4.0 - 6.0 * c},
                "forearm_far": {"rot": -3.0 * math.cos(2 * math.pi * u + 0.5)}}
        aim(rig, pose, "torso", w["lean"] + 1.0 * math.cos(4 * math.pi * u))
        aim(rig, pose, "helmet", w["head"])
        aim(rig, pose, "sword", w["sword"] + 6.0 * math.cos(2 * math.pi * u + 0.8), "hand_near")
        return pose

    return gait_pose(u, w, rig, upper)


CROUCH = {"lean": 40, "head": 12, "pelvis_x": -8, "far_knee_dx": 15, "far_pitch": 55, "shield_arm": -55,
          "shield_fore": -95, "sword": 14}


def part_bottom(rig, mats, bone):
    """Lowest model y of the opaque pixels of a bone part (for ground contact, never read at runtime)."""
    b = rig.bone(bone)
    im = rig.parts[b["part"]]
    px, py = rig.pivots[b["part"]]
    a = im.getchannel("A")
    m = mats[bone]
    return max(mat_apply(m, (x + 0.5 - px, y + 1 - py))[1]
               for y in range(im.height) for x in range(im.width) if a.getpixel((x, y)))


def crouch_targets(rig, pose):
    """One-knee crouch: the near foot flat and planted, the far foot on its toe behind its knee."""
    _, toe = sole_points(rig.parts["sabaton_far"], rig.pivots["sabaton_far"])
    p = math.radians(CROUCH["far_pitch"])
    ankle_y = -(toe[0] * math.sin(p) + toe[1] * math.cos(p))  # the toe tip just touches the ground
    hx, hy = rig.hip(pose, "far")
    shin = rig.bone(IK["far"]["foot"])
    l2 = math.hypot(shin["x"], shin["y"])
    knee_x = hx + CROUCH["far_knee_dx"]
    ankle_x = knee_x - math.sqrt(max(0.0, l2 * l2 - (ankle_y + 5) ** 2))
    return {"near": {"x": rig.foot_rest["near"]}, "far": {"x": ankle_x, "y": ankle_y, "pitch": CROUCH["far_pitch"]}}


def crouch_hold_drop(rig):
    """Pelvis drop that puts the far knee on the ground in the hold pose."""
    lo, hi = 0.0, 60.0
    for _ in range(25):
        mid = (lo + hi) / 2
        pose = {"pelvis": {"x": CROUCH["pelvis_x"], "y": mid}}
        tg = crouch_targets(rig, pose)
        for c in IK:
            rig.solve_leg(c, pose, tg[c])
        lo, hi = (mid, hi) if part_bottom(rig, rig.world(pose), IK["far"]["thigh"]) < -0.5 else (lo, mid)
    return lo


def crouch(t, i, rig):
    # 0-3 down, 4-15 hold (looped by the game while the key is held), 16-19 up.
    g = CROUCH
    c = K(i, {0: 0, 3: 1, 16: 1, 19: 0})
    breath = 0.5 - 0.5 * math.cos(2 * math.pi * (i - 4) / 12) if 4 <= i <= 15 else 0.0  # 0..1, rises only
    if "drop" not in g:
        g["drop"] = crouch_hold_drop(rig)
    pose = {
        "pelvis": {"x": g["pelvis_x"] * c, "y": g["drop"] * c - breath},
        "tabard_front": {"rot": -24 * c},
        "tabard_back": {"rot": 16 * c},
    }
    rest = {b: rest_rot(rig, b) for b in ("torso", "helmet", "arm_far", "forearm_far", "shield", "arm_near",
                                          "forearm_near", "sword")}
    goal = {"torso": g["lean"], "helmet": g["head"], "arm_far": g["shield_arm"], "forearm_far": g["shield_fore"],
            "shield": 6, "arm_near": -10, "forearm_near": -60, "sword": g["sword"]}
    for b, v in goal.items():
        aim(rig, pose, b, rest[b] + wrap180(v - rest[b]) * c, "hand_near" if b == "sword" else None)
    planted_far = {"x": rig.foot_rest["far"], "y": rig.ankle_target("far", {"x": 0})[1], "pitch": 0.0}
    kneel = crouch_targets(rig, pose)
    far = {k: planted_far[k] + (kneel["far"][k] - planted_far[k]) * c for k in ("x", "y", "pitch")}
    # The far foot lifts a little while it moves back, so its toe never digs into the ground.
    far["y"] -= 6 * math.sin(math.pi * c)
    pose["ik"] = {"near": kneel["near"], "far": far}
    return pose


SWING = {  # world rotations per frame for the slash; the last key is the rest pose
    "arm_near": {2: -150, 3: -165, 4: -125, 5: -70, 6: -40, 8: -20},
    "forearm_near": {2: 170, 3: 165, 4: 210, 5: 280, 6: 305, 8: 320},
    "sword": {2: -100, 3: -160, 4: -100, 5: -10, 6: 40, 8: 30},
}


def attack(t, i, rig):
    # 0-3 windup, 4-6 strike, 7-11 recover. The feet stay planted; the body steps into the swing.
    pose = {
        "pelvis": {"x": K(i, {0: 0, 2: -2, 4: 3, 5: 5, 6: 5, 8: 3, 11: 0}), "y": K(i, {0: 0, 2: 1, 5: 4, 6: 4, 11: 0})},
        "arm_far": {"rot": K(i, {0: 0, 2: 6, 5: -10, 8: -6, 11: 0})},
        "tabard_front": {"rot": K(i, {0: 0, 3: 4, 5: -8, 8: -3, 11: 0})},
        "tabard_back": {"rot": K(i, {0: 0, 3: 5, 5: -6, 8: -2, 11: 0})},
    }
    rt = rest_rot(rig, "torso")
    aim(rig, pose, "torso", K(i, {0: rt, 2: rt - 6, 4: rt + 10, 5: 14, 6: 16, 9: 5, 11: rt}))
    aim(rig, pose, "helmet", K(i, {0: rest_rot(rig, "helmet"), 2: -4, 5: 6, 8: 3, 11: rest_rot(rig, "helmet")}))
    for bone, keys in SWING.items():
        r = rest_rot(rig, bone)
        unwrapped = r if bone != "forearm_near" else r + 360
        via = "hand_near" if bone == "sword" else None
        aim(rig, pose, bone, K(i, {**keys, 0: r, 11: unwrapped}), via)
    pose["ik"] = planted(rig)
    return pose


def block(t, i, rig):
    # 0-3 raise, 4-9 hold (looped by the game while the key is held), 10-13 lower.
    r = K(i, {0: 0, 3: 1, 10: 1, 13: 0})
    brace = math.sin(2 * math.pi * (i - 4) / 6) if 4 <= i <= 9 else 0.0
    pose = {
        "pelvis": {"x": -1.5 * r, "y": 5 * r + 0.5 * brace, "rot": 3 * r},
        "tabard_front": {"rot": -5 * r},
        "tabard_back": {"rot": 3 * r},
    }
    rest = {b: rest_rot(rig, b) for b in ("torso", "helmet", "arm_far", "forearm_far", "shield", "arm_near",
                                          "forearm_near", "sword")}
    goal = {"torso": 8 + 0.8 * brace, "helmet": 6, "arm_far": -105, "forearm_far": -150 + 2 * brace, "shield": -2,
            "arm_near": 28, "forearm_near": 0, "sword": 35}
    for b, g in goal.items():
        aim(rig, pose, b, rest[b] + wrap180(g - rest[b]) * r, "hand_near" if b == "sword" else None)
    pose["ik"] = planted(rig)
    return pose


def jump(t, i, rig):
    pelvis_y = K(i, {0: 0, 1: 4, 2: 8, 3: -1, 4: -1, 5: 0, 7: 0, 9: -1, 10: 9, 11: 7, 12: 3, 13: 0})
    torso = K(i, {0: 0, 2: 10, 3: 2, 4: 0, 6: 4, 8: 2, 9: 0, 10: 9, 12: 3, 13: 0})
    pitch = K(i, {0: 0, 2: 0, 3: 26, 4: 20, 5: 10, 7: 4, 8: -4, 9: -6, 10: 0})
    lift_n = K(i, {3: 0, 4: 4, 5: 12, 6: 16, 7: 16, 8: 9, 9: 2, 10: 0})
    lift_f = K(i, {3: 0, 4: 2, 5: 9, 6: 13, 7: 14, 8: 10, 9: 4, 10: 0})
    tuck = K(i, {4: 0, 6: 5, 7: 5, 9: 0})
    pose = {
        "pelvis": {"y": pelvis_y, "rot": K(i, {0: 0, 2: 4, 3: -2, 5: 2, 9: 0, 10: 4, 13: 0})},
        "torso": {"rot": torso},
        "helmet": {"rot": K(i, {0: 0, 2: -6, 3: -2, 6: -3, 10: -5, 13: 0})},
        "arm_near": {"rot": K(i, {0: 0, 2: 16, 3: -30, 5: -22, 7: -18, 9: -8, 10: 6, 12: 2, 13: 0})},
        "forearm_near": {"rot": K(i, {0: 0, 2: 6, 3: -20, 6: -24, 9: -10, 10: 4, 13: 0})},
        "hand_near": {"rot": K(i, {0: 0, 3: 10, 6: 14, 10: 0})},
        "arm_far": {"rot": K(i, {0: 0, 2: 10, 3: -18, 6: -14, 9: -6, 10: 4, 13: 0})},
        "forearm_far": {"rot": K(i, {0: 0, 3: -6, 10: 2, 13: 0})},
        "tabard_front": {"rot": K(i, {3: 0, 4: 6, 6: -8, 8: -12, 9: -6, 10: 4, 13: 0})},
        "tabard_back": {"rot": K(i, {3: 0, 4: 8, 6: -4, 8: -10, 10: 4, 13: 0})},
    }
    pose["ik"] = {
        "near": {"x": rig.foot_rest["near"] + tuck, "lift": lift_n, "pitch": pitch},
        "far": {"x": rig.foot_rest["far"] + tuck * 0.6, "lift": lift_f, "pitch": pitch},
    }
    return pose


def hurt(t, i, rig):
    # 0-1 hit: the torso and the helmet snap back, the sword arm flings back, the knees give; 2-7 recover.
    pose = {
        "pelvis": {"x": K(i, {0: -4, 1: -5, 3: -2, 7: 0}), "y": K(i, {0: 3, 1: 4, 3: 2, 7: 0}),
                   "rot": K(i, {0: -5, 1: -7, 3: -3, 7: 0})},
        "torso": {"rot": K(i, {0: -12, 1: -14, 3: -6, 5: -2, 7: 0})},
        "helmet": {"rot": K(i, {0: -6, 1: -8, 2: -5, 4: -1, 7: 0})},
        "arm_near": {"rot": K(i, {0: 14, 1: 18, 3: 6, 7: 0})},
        "forearm_near": {"rot": K(i, {0: 10, 1: 12, 4: 2, 7: 0})},
        "hand_near": {"rot": K(i, {0: -8, 1: -10, 4: -2, 7: 0})},
        "arm_far": {"rot": K(i, {0: 10, 1: 12, 4: 2, 7: 0})},
        "forearm_far": {"rot": K(i, {0: 6, 1: 8, 4: 1, 7: 0})},
        "tabard_front": {"rot": K(i, {0: -8, 2: 5, 4: -1, 7: 0})},
        "tabard_back": {"rot": K(i, {0: -10, 2: 4, 4: -1, 7: 0})},
        "ik": planted(rig),
    }
    return pose


KNEEL_HIP_Y = -32  # far hip height when the far knee is on the ground
LIE_ANKLE_Y = -19   # ankle height when lying: the toes point down and just touch the ground
LIE_HIP_Y = -15     # hip height when lying, close to the ankles, so the legs lie almost level
LIE_REACH = 0.97    # lying legs are almost straight


def death(t, i, rig):
    # 0-3 stagger back, 4-8 the far knee drops while the near foot stays planted,
    # 9-15 collapse forward with the legs straight back, 16-17 held remains.
    pose = {
        "pelvis": {"x": K(i, {0: -3, 1: -6, 2: -7, 3: -6, 5: -3, 8: -2, 11: 0, 15: 6}),
                   "rot": K(i, {0: -4, 2: -6, 3: 2, 6: 8, 8: 10, 10: 30, 13: 74, 15: 88})},
        "torso": {"rot": K(i, {0: -10, 1: -14, 2: -8, 3: 6, 6: 10, 8: 14, 11: 8, 15: 0})},
        "tabard_front": {"rot": K(i, {0: -6, 2: 4, 5: -2, 8: -18, 12: -40, 15: -70})},
        "tabard_back": {"rot": K(i, {0: -8, 2: 3, 5: -2, 8: -10, 12: -40, 15: -70})},
    }
    # Arms by world angle, so they never point into the ground.
    aim(rig, pose, "arm_near", K(i, {0: 0, 1: -6, 3: 12, 8: 14, 11: -20, 13: -60, 15: -78}))
    aim(rig, pose, "forearm_near", K(i, {0: -40, 1: -44, 3: -10, 8: 0, 11: -30, 13: -70, 15: -84}))
    aim(rig, pose, "sword", K(i, {0: -26, 1: -30, 2: -10, 3: 18, 6: 22, 8: 22, 11: 10, 13: 0, 15: -2}), "hand_near")
    aim(rig, pose, "arm_far", K(i, {0: 0, 3: 4, 8: 6, 11: 40, 13: 70, 15: 84}))
    aim(rig, pose, "forearm_far", K(i, {0: -70, 3: -50, 8: -40, 11: 0, 13: 60, 15: 90}))
    aim(rig, pose, "shield", K(i, {0: 0, 8: 6, 11: 40, 13: 80, 15: 90}))
    aim(rig, pose, "helmet", K(i, {0: -18, 1: -22, 2: -10, 3: 8, 8: 22, 11: 30, 13: 60, 15: 80}))
    # Pelvis height: the far knee reaches the ground at frame 7; the lying frames settle on the ground.
    pose["pelvis"]["y"] = 0.0
    kneel_drop = KNEEL_HIP_Y - hip(rig, pose, "far")[1]
    lie_drop = LIE_HIP_Y - (hip(rig, pose, "far")[1] + hip(rig, pose, "near")[1]) / 2
    pose["pelvis"]["y"] = K(i, {0: 1, 2: 2, 3: 3, 5: kneel_drop * 0.45, 7: kneel_drop, 8: kneel_drop,
                                10: kneel_drop + 4, 15: lie_drop})
    lie = K(i, {9: 0, 15: 1})  # legs move from the kneel to straight back along the ground
    ik = {}
    for c in IK:
        hx, hy = hip(rig, pose, c)
        if c == "far":
            step = K(i, {0: 0, 1: -5, 2: -9})
            sx, sy = rig.foot_rest[c] + step, -11 - K(i, {0: 0, 1: 5, 2: 0})
            kx, ky = hx - 24, -6
            knee = K(i, {3: 0, 7: 1})
            x, y = sx + (kx - sx) * knee, sy + (ky - sy) * knee - 6 * math.sin(math.pi * knee)
            pitch = K(i, {3: 0, 4: 25, 5: 100, 6: 150, 7: 168})
            p = math.radians(min(90.0, pitch))
            _, (tx, ty) = sole_points(rig.parts["sabaton_far"], rig.pivots["sabaton_far"])
            y -= max(0.0, tx * math.sin(p) + ty * math.cos(p) - ty)  # the toe clears the ground while it turns
        else:
            x, y, pitch = rig.foot_rest[c], -11.0, 0.0  # planted flat until the collapse
        dy = LIE_ANKLE_Y - hy
        lx = hx - math.sqrt(max(0.0, (LIE_REACH * rig.leg_len[c]) ** 2 - dy * dy))
        ik[c] = {"x": x + (lx - x) * lie, "y": y + (LIE_ANKLE_Y - y) * lie, "pitch": pitch + (90 - pitch) * lie}
    pose["ik"] = ik
    return pose


ANIMATIONS = [
    {"name": "idle", "fps": 8, "frames": 16, "loop": True, "pose": idle},
    {"name": "walk", "fps": 12, "frames": 12, "loop": True, "pose": walk, "stride": "feet"},
    {"name": "sprint", "fps": 15, "frames": 10, "loop": True, "pose": sprint, "stride": "feet"},
    {"name": "jump", "fps": 12, "frames": 14, "loop": False, "pose": jump,
     "phases": {"takeoff": [0, 3], "air": [4, 9], "land": [10, 13]}},
    {"name": "hurt", "fps": 12, "frames": 8, "loop": False, "pose": hurt, "phases": {"hit": [0, 1], "recover": [2, 7]}},
    {"name": "death", "fps": 10, "frames": 18, "loop": False, "pose": death, "z": {"shield": 8}, "settle": [9, 17],
     "phases": {"stagger": [0, 3], "knees": [4, 8], "collapse": [9, 15], "remains": [16, 17]}},
    {"name": "crouch", "fps": 12, "frames": 20, "loop": False, "pose": crouch,
     "phases": {"down": [0, 3], "hold": [4, 15], "up": [16, 19]}},
    {"name": "attack", "fps": 14, "frames": 12, "loop": False, "pose": attack,
     "phases": {"windup": [0, 3], "strike": [4, 6], "recover": [7, 11]},
     "hitbox": {"bone": "sword", "from_x": SWORD_GUARD_X, "frames": [4, 5, 6], "slices": 4, "pad": 2}},
    {"name": "block", "fps": 12, "frames": 14, "loop": False, "pose": block,
     "phases": {"raise": [0, 3], "hold": [4, 9], "lower": [10, 13]}},
]
ANIM = {a["name"]: dict(a, duration=a["frames"] / a["fps"]) for a in ANIMATIONS}

CONFIG = {
    "id": ID, "name": f"{ID}-rig", "out_dir": OUT_DIR, "master": P + "piece-01.png",
    "parts": PARTS, "variants": VARIANTS, "bones": BONES, "rest": REST, "ik": IK,
    "followers": FOLLOWERS, "animations": ANIMATIONS,
}
