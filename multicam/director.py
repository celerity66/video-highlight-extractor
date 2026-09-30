"""Choosing the camera closest to the action, moment by moment.

Each camera gets a score per sample from the motion in the part of its view that counts
(the ice it covers well, weighted per grid row; the goalie and net blocked out for
cameras behind a net), scaled by that camera's own typical level. A choice that follows
the best-scoring camera while paying a penalty for every switch is found for the whole
game at once, then shots shorter than a minimum are merged into a neighbour.
"""
import numpy as np

from .motion import FPS


def weight_map(rows, block=None):
    w = np.repeat(np.array(rows, np.float32)[:, None], 16, axis=1)
    if block:
        r0, r1, c0, c1 = block
        w[r0:r1, c0:c1] = 0
    return w


def scores(grids, offsets, t0, n, config):
    """Per-camera score on the shared timeline (samples from reference time t0).
    grids: {camera: motion grid of its recording}; offsets: {camera: seconds into its
    recording at reference time 0}."""
    d = config["director"]
    cams = list(grids)
    k = max(1, int(d["smooth_s"] * FPS))
    lead = int(round(d["lead_s"] * FPS))
    near_rows = weight_map([0, 0, 0, 0, 0, 0, 1, 1, 1])
    # long-view fallbacks: one camera's settings, or a list of them
    fv = d.get("far_view") or []
    far_views = fv if isinstance(fv, list) else [fv]

    def norm(x):
        return x / (np.percentile(x, d["norm_pct"]) + 1e-6)

    def to_timeline(c, x):
        """Smooth, look LEAD s ahead (cut as the play starts to move), place on the timeline."""
        x = np.convolve(np.minimum(x, 3.0), np.ones(k) / k, mode="same")
        x = np.concatenate([x[lead:], np.repeat(x[-1:], lead)])
        out = np.full(n, -np.inf, np.float32)
        first = int(round((-offsets[c] - t0) * FPS))
        src0 = max(0, -first)
        dst0 = max(0, first)
        m = min(len(x) - src0, n - dst0)
        if m > 0:
            out[dst0:dst0 + m] = x[src0:src0 + m]
        return out

    sc = np.full((len(cams), n), -np.inf, np.float32)
    near = {}
    for i, c in enumerate(cams):
        g = np.asarray(grids[c]).astype(np.float32)
        cam = config["cameras"][c]
        w = weight_map(cam["rows"], cam.get("block"))
        x = norm((g * w).sum(axis=(1, 2)) / w.sum())
        for f in far_views:
            if f["camera"] == c:
                fw = weight_map(f["rows"], f.get("block"))
                x = np.maximum(x, f["factor"] * norm((g * fw).sum(axis=(1, 2)) / fw.sum()))
        sc[i] = to_timeline(c, x)
        near[c] = to_timeline(c, norm((g * near_rows).sum(axis=(1, 2)) / near_rows.sum()))

    # of two cameras watching the same net from opposite sides, the one with players
    # closer to it than the other loses the difference: the other sees them across the ice
    for a, b, penalty in d.get("across_pairs", []):
        if a in near and b in near:
            ia, ib = cams.index(a), cams.index(b)
            both = np.isfinite(near[a]) & np.isfinite(near[b])
            diff = np.zeros(n, np.float32)
            diff[both] = near[a][both] - near[b][both]
            sc[ia, both] = np.maximum(sc[ia, both] - penalty * np.maximum(diff[both], 0), 0)
            sc[ib, both] = np.maximum(sc[ib, both] - penalty * np.maximum(-diff[both], 0), 0)
    # a preferred camera: the other only wins when clearly better
    for p, o, margin in d.get("prefer", []):
        if p in cams and o in cams:
            ip, io = cams.index(p), cams.index(o)
            both = np.isfinite(sc[ip]) & np.isfinite(sc[io])
            loses = both & (sc[io] <= margin * sc[ip])
            sc[io, loses] = np.minimum(sc[io, loses], 0.99 * sc[ip, loses])
    return cams, sc


def shots(path):
    out, start = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[start]:
            out.append([start, t, int(path[start])])
            start = t
    return out


def choose(sc, config):
    """Best camera sequence: most score, minus switch_cost score-seconds per switch."""
    d = config["director"]
    ncam, n = sc.shape
    per = np.where(np.isfinite(sc), sc / FPS, -1e9)
    best = per[:, 0].copy()
    back = np.zeros((ncam, n), np.int16)
    idx = np.arange(ncam)
    for t in range(1, n):
        j = int(np.argmax(best))
        switch = best[j] - d["switch_cost"]
        take = switch > best
        back[:, t] = np.where(take, j, idx)
        best = np.maximum(best, switch) + per[:, t]
    path = np.zeros(n, np.int16)
    path[-1] = int(np.argmax(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    # merge shots shorter than the minimum into the neighbour that scores better there
    min_len = int(np.ceil(d["min_shot_s"] * FPS))
    while True:
        sh = shots(path)
        short = [i for i, (a, b, _) in enumerate(sh) if b - a < min_len]
        if not short or len(sh) == 1:
            return path
        i = min(short, key=lambda i: sh[i][1] - sh[i][0])
        a, b, _ = sh[i]
        options = [sh[j][2] for j in (i - 1, i + 1) if 0 <= j < len(sh)]
        path[a:b] = max(options, key=lambda c: np.where(np.isfinite(sc[c, a:b]), sc[c, a:b], -1e9).sum())
