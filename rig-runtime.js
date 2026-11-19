// Skeletal cutout rig runtime. Plain script, no build step, works on file://.
// Reads a rig manifest (window.SPRITE_RIGS[id], contract in tools/rig_core.py), computes bone world
// transforms (FK) and draws every part on a 1x offscreen canvas with nearest-neighbor sampling. The page
// then scales that canvas by a whole number, so rotated parts keep one pixel grid. Never reads pixels.
(() => {
  const DEG = Math.PI / 180;

  function sampleKeys(keys, t) {
    if (t <= keys[0][0]) return keys[0][1];
    const last = keys.length - 1;
    if (t >= keys[last][0]) return keys[last][1];
    let lo = 0;
    let hi = last;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (keys[mid][0] <= t) lo = mid; else hi = mid;
    }
    const [t0, v0] = keys[lo];
    const [t1, v1] = keys[hi];
    return v0 + (v1 - v0) * ((t - t0) / (t1 - t0));
  }

  function createRig(manifest, image) {
    const index = new Map(manifest.bones.map((b, i) => [b.name, i]));
    const bones = manifest.bones.map((b) => ({ ...b, parentIndex: b.parent == null ? -1 : index.get(b.parent) }));
    const anims = {};
    for (const [name, a] of Object.entries(manifest.animations)) {
      const tracks = [];
      for (const [bone, chans] of Object.entries(a.tracks)) {
        const i = index.get(bone);
        if (i !== undefined) tracks.push({ i, rot: chans.rot, x: chans.x, y: chans.y });
      }
      const z = a.z ?? {};
      const order = bones
        .map((b, i) => ({ i, z: z[b.name] ?? b.z, part: b.part }))
        .filter((b) => b.part && manifest.parts[b.part])
        .sort((p, q) => p.z - q.z)
        .map((b) => b.i);
      anims[name] = { ...a, name, tracks, order };
    }
    const restOrder = bones
      .map((b, i) => ({ i, z: b.z, part: b.part }))
      .filter((b) => b.part && manifest.parts[b.part])
      .sort((p, q) => p.z - q.z)
      .map((b) => b.i);
    return { manifest, image, bones, index, anims, restOrder };
  }

  class RigInstance {
    constructor(rig) {
      this.rig = rig;
      const { frame } = rig.manifest;
      this.canvas = document.createElement('canvas');
      this.canvas.width = frame.w;
      this.canvas.height = frame.h;
      this.ctx = this.canvas.getContext('2d');
      const n = rig.bones.length;
      this.local = new Float64Array(n * 3);
      this.world = new Float64Array(n * 6);
      this.anim = null;
      this.time = 0;
      this.stepped = true;
      this.finished = false;
    }

    // keepPhase: switch between two loops at the same cycle fraction (walk and sprint), with no pop.
    play(name, { restart = false, time = 0, keepPhase = false } = {}) {
      const anim = this.rig.anims[name];
      if (!anim) return false;
      if (this.anim !== anim || restart) {
        const prev = this.anim;
        this.anim = anim;
        this.time = keepPhase && prev && prev.loop && anim.loop ? (this.time / prev.duration) * anim.duration : time;
        this.finished = false;
      }
      return true;
    }

    // dt in seconds, already scaled by any speed factor.
    update(dt) {
      const a = this.anim;
      if (!a) return;
      this.time += dt;
      if (a.loop) {
        this.time %= a.duration;
      } else if (this.time >= a.duration) {
        this.time = a.duration;
        this.finished = true;
      }
    }

    // The time actually shown: with stepped on, frame i holds from i/fps to (i+1)/fps.
    sampleTime() {
      const a = this.anim;
      if (!a) return 0;
      if (!this.stepped) return this.time;
      return Math.min(a.frames - 1, Math.floor(this.time * a.fps + 1e-6)) / a.fps;
    }

    frameIndex() {
      const a = this.anim;
      return a ? Math.min(a.frames - 1, Math.floor(this.time * a.fps + 1e-6)) : 0;
    }

    phase() {
      const a = this.anim;
      if (!a || !a.phases) return null;
      const f = this.frameIndex();
      for (const [name, [first, last]] of Object.entries(a.phases)) if (f >= first && f <= last) return name;
      return null;
    }

    pose() {
      const { bones } = this.rig;
      const L = this.local;
      for (let i = 0; i < bones.length; i++) {
        L[i * 3] = bones[i].x;
        L[i * 3 + 1] = bones[i].y;
        L[i * 3 + 2] = bones[i].rot;
      }
      const a = this.anim;
      if (a) {
        const t = this.sampleTime();
        for (const tr of a.tracks) {
          if (tr.x) L[tr.i * 3] += sampleKeys(tr.x, t);
          if (tr.y) L[tr.i * 3 + 1] += sampleKeys(tr.y, t);
          if (tr.rot) L[tr.i * 3 + 2] += sampleKeys(tr.rot, t);
        }
      }
      const W = this.world;
      for (let i = 0; i < bones.length; i++) {
        const r = L[i * 3 + 2] * DEG;
        const co = Math.cos(r);
        const si = Math.sin(r);
        const x = L[i * 3];
        const y = L[i * 3 + 1];
        const p = bones[i].parentIndex;
        const o = i * 6;
        if (p < 0) {
          W[o] = co; W[o + 1] = si; W[o + 2] = -si; W[o + 3] = co; W[o + 4] = x; W[o + 5] = y;
        } else {
          const q = p * 6;
          const a0 = W[q], b0 = W[q + 1], c0 = W[q + 2], d0 = W[q + 3];
          W[o] = a0 * co + c0 * si;
          W[o + 1] = b0 * co + d0 * si;
          W[o + 2] = -a0 * si + c0 * co;
          W[o + 3] = -b0 * si + d0 * co;
          W[o + 4] = a0 * x + c0 * y + W[q + 4];
          W[o + 5] = b0 * x + d0 * y + W[q + 5];
        }
      }
    }

    // Pose, then draw the parts on the 1x offscreen canvas.
    render() {
      this.pose();
      const { manifest, image, bones } = this.rig;
      const { parts, pivot } = manifest;
      const ctx = this.ctx;
      const W = this.world;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.imageSmoothingEnabled = false;
      const order = this.anim ? this.anim.order : this.rig.restOrder;
      for (const i of order) {
        const part = parts[bones[i].part];
        const o = i * 6;
        ctx.setTransform(W[o], W[o + 1], W[o + 2], W[o + 3], W[o + 4] + pivot.x, W[o + 5] + pivot.y);
        ctx.drawImage(image, part.x, part.y, part.w, part.h, -part.pivot.x, -part.pivot.y, part.w, part.h);
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    // (x, y) is the ground contact point in art px of the target; scale is a whole number of target px
    // per art px; facing 1 is +x, -1 mirrors. The root lands on whole art px.
    draw(ctx, x, y, scale, facing = 1) {
      const { frame, pivot } = this.rig.manifest;
      const rx = Math.round(x);
      const ry = Math.round(y);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.translate(rx * scale, ry * scale);
      if (facing < 0) ctx.scale(-1, 1);
      ctx.drawImage(this.canvas, -pivot.x * scale, -pivot.y * scale, frame.w * scale, frame.h * scale);
      ctx.restore();
    }

    // Bone lines and pivots, for debugging. Call after render().
    drawBones(ctx, x, y, scale, facing = 1) {
      const { bones } = this.rig;
      const W = this.world;
      const rx = Math.round(x);
      const ry = Math.round(y);
      const at = (i) => [(rx + facing * W[i * 6 + 4]) * scale, (ry + W[i * 6 + 5]) * scale];
      ctx.save();
      ctx.lineWidth = Math.max(1, scale / 3);
      ctx.strokeStyle = 'rgb(0 230 255)';
      ctx.beginPath();
      for (let i = 0; i < bones.length; i++) {
        const p = bones[i].parentIndex;
        if (p <= 0) continue;
        const [x0, y0] = at(p);
        const [x1, y1] = at(i);
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
      }
      ctx.stroke();
      const r = Math.max(2, scale * 0.7);
      for (let i = 0; i < bones.length; i++) {
        const [px, py] = at(i);
        ctx.fillStyle = i === 0 ? 'rgb(120 255 90)' : 'rgb(255 230 0)';
        ctx.fillRect(px - r, py - r, r * 2, r * 2);
      }
      ctx.restore();
    }
  }

  window.RigRuntime = {
    createRig,
    createInstance: (rig) => new RigInstance(rig),
    sampleKeys,
  };
})();
