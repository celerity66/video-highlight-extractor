"""Finding and following scoreboard clocks in any camera's view, automatically.

A running game clock is the one thing in a rink camera's view that changes exactly once
a second, at the same moment each second. Players move all the time and lights flicker,
but neither keeps that beat. So each pixel is scored by how much of its frame-to-frame
change falls at one fixed point within the second: the seconds digit of a clock stands
out. The clock found is then followed through the whole recording (the frame is lined up
on the scoreboard around it, since cameras on nets get bumped), giving per sample whether
the clock is ticking. That pattern of starts and stops is the same for every camera that
sees a clock, so it lines cameras up in time, and it shows when the game is being played.
"""
import os
import subprocess

import cv2
import numpy as np

from .motion import FPS, _key

HALF_W, HALF_H = 1280, 720      # search at half resolution
WINDOW_S = 60                   # judge the beat over one-minute stretches
SAMPLES = 8                     # stretches sampled across a recording
DIGIT = (14, 24)                # seconds digit size at half resolution, generously (w, h)


def _decode(path, start, dur, w, h, crop=None):
    vf = f"fps={FPS}," + (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else "") + f"scale={w}:{h},format=gray"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-i", path, "-t", f"{dur:.2f}", "-an",
                          "-vf", vf, "-f", "rawvideo", "-"], capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w)


def beat_map_stream(path, start, dur, w, h):
    """Per pixel: how strongly its changes keep a once-a-second beat (0 = none).
    Frames are processed as they're decoded, keeping only running totals (a minute of
    half-resolution video would otherwise need over a gigabyte of memory)."""
    vf = f"fps={FPS},scale={w}:{h},format=gray"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-i", path, "-t", f"{dur:.2f}", "-an",
                          "-vf", vf, "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    per_pos = np.zeros((FPS, h, w), np.float32)         # total change at each position in the second
    prev, n = None, 0
    size = w * h
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        cur = np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)
        if prev is not None:
            per_pos[(n - 1) % FPS] += np.abs(cur - prev)   # change since the previous sample
        prev, n = cur, n + 1
    p.wait()
    if n < FPS * 10:
        return None
    per_pos /= max(1, (n - 1) // FPS)
    peak = per_pos.max(axis=0)
    rest = np.median(per_pos, axis=0)
    # a clock: one position changes a lot, the others hardly at all
    return np.clip(peak - 2 * rest, 0, None)


def find_clock(files, lengths, cache_dir, min_strength=2.5):
    """Look for a ticking clock in a camera's recording. Returns a dict with the clock's
    box in full-resolution pixels and its strength, or None if no clock was seen."""
    key = os.path.join(cache_dir, _key(files[0].path, f"clockfind{len(files)}") + ".json")
    if os.path.exists(key):
        import json
        found = json.load(open(key))
        return found or None
    # sample several minutes across the recording; keep the one where a clock stands out
    # most (cameras on nets drift, so the clock isn't in the same place all game)
    best = None
    for t_rec in np.linspace(0.1, 0.9, SAMPLES) * sum(lengths):
        t = t_rec
        for f, n in zip(files, lengths):              # which file holds time t
            if t < n:
                bm = beat_map_stream(f.path, t, WINDOW_S, HALF_W, HALF_H)
                if bm is not None:
                    # light smoothing only: a distant clock's digits are just a few pixels across
                    smooth = cv2.blur(bm, (3, 3))
                    _, peak, _, (x, y) = cv2.minMaxLoc(smooth)
                    strength = peak / (np.percentile(smooth, 99.9) + 1)
                    if best is None or strength > best[0]:
                        best = (strength, peak, x, y, smooth, float(t_rec))
                break
            t -= n
    found = None
    if best and best[0] >= min_strength:
        strength, peak, x, y, smooth, t_best = best
        # the digit(s): the connected area around the peak that still scores well
        mask = (smooth > 0.3 * peak).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        bx, by, bw, bh, _ = stats[labels[y, x]]
        pad = 1                                   # tight: static board around the digit dilutes its changes
        found = {"x": int(max(0, bx - pad) * 2), "y": int(max(0, by - pad) * 2),
                 "w": int((bw + 2 * pad) * 2), "h": int((bh + 2 * pad) * 2),
                 "strength": round(float(strength), 1),
                 "t": round(t_best + WINDOW_S / 2, 1)}    # when it was seen there (recording time)
    import json
    json.dump(found or {}, open(key, "w"))
    return found


MARGIN = (120, 80)          # search this far around the clock for it in later frames (bumped cameras)
CONTEXT = 24                # scoreboard around the digits used to line frames up


def _norm(im):
    im = im.astype(np.float32)
    return (im - im.mean()) / (im.std() + 1e-6)


def track_clock(files, lengths, clock, cache_dir):
    """Per sample: shape similarity of the clock's digits to the previous sample
    (1 = unchanged, NaN = clock not found), over the whole recording."""
    key = os.path.join(cache_dir, _key(files[0].path, f"clocktrack{len(files)}_{clock['x']}_{clock['y']}"))
    if os.path.exists(key):
        return np.load(key)
    W, H = 2560, 1440
    cx0 = max(0, clock["x"] - MARGIN[0])
    cy0 = max(0, clock["y"] - MARGIN[1])
    cx1 = min(W, clock["x"] + clock["w"] + MARGIN[0])
    cy1 = min(H, clock["y"] + clock["h"] + MARGIN[1])
    crop = (cx0, cy0, cx1 - cx0, cy1 - cy0)
    # the scoreboard around the digits, at the moment the clock was found there: used to
    # line every frame up (the camera may drift, so a different moment could show elsewhere)
    t = clock["t"]
    for f, n in zip(files, lengths):
        if t < n:
            sample = _decode(f.path, max(0, t - 5), 10, crop[2], crop[3], crop)
            break
        t -= n
    tx0 = max(0, clock["x"] - cx0 - CONTEXT)
    ty0 = max(0, clock["y"] - cy0 - CONTEXT)
    tw, th = clock["w"] + 2 * CONTEXT, clock["h"] + 2 * CONTEXT
    template = np.median(sample, axis=0).astype(np.float32)[ty0:ty0 + th, tx0:tx0 + tw]
    dx0, dy0 = clock["x"] - cx0 - tx0, clock["y"] - cy0 - ty0     # the digits' place in the template
    out = []
    for f, n in zip(files, lengths):
        frames = _stream(f.path, crop)
        sims, prev = [], None
        for fr in frames:
            fr = fr.astype(np.float32)
            r = cv2.matchTemplate(fr, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (bx, by) = cv2.minMaxLoc(r)
            d = fr[by + dy0:by + dy0 + clock["h"], bx + dx0:bx + dx0 + clock["w"]]
            ok = score > 0.4 and d.shape == (clock["h"], clock["w"]) and np.percentile(d, 95) - np.median(d) > 15
            cur = _norm(d) if ok else None
            sims.append(float((cur * prev).mean()) if cur is not None and prev is not None else np.nan)
            prev = cur
        s = np.array(sims, np.float32)
        m = int(round(n * FPS))
        out.append(s[:m] if len(s) >= m else np.concatenate([s, np.full(m - len(s), np.nan, np.float32)]))
    sims = np.concatenate(out)
    np.save(key, sims)
    return sims


def clock_state(files, lengths, cache_dir):
    """For one camera's recording: (per-sample tick state 1/0/-1, clock info) if a real
    game clock is seen, else (None, reason). A clock must tick for a fair share of the
    game and be seen stopped for some of it; a static sign or a constantly changing
    display is rejected."""
    from . import scoreboard
    ck = find_clock(files, lengths, cache_dir)
    if not ck:
        return None, "no clock in view"
    st = scoreboard.tick_state(track_clock(files, lengths, ck, cache_dir))
    judged = float(np.mean(st >= 0))
    ticking = float(np.mean(st == 1))
    stopped = float(np.mean(st == 0))
    ck.update(judged=round(judged, 2), ticking=round(ticking, 2), stopped=round(stopped, 2))
    if judged < 0.15 or ticking < 0.05 or stopped < 0.05:
        return None, f"not a usable clock (judged {judged:.0%}, ticking {ticking:.0%}, stopped {stopped:.0%})"
    return st, ck


def _stream(path, crop):
    x, y, w, h = crop
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-an",
                          "-vf", f"fps={FPS},crop={w}:{h}:{x}:{y},format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    size = w * h
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w)
    p.wait()
