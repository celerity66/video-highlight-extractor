"""Reading each video once and caching small per-frame summaries of it."""
import concurrent.futures as cf
import hashlib
import os
import subprocess

import numpy as np

FPS = 5                  # samples per second kept from every video
W, H = 256, 144          # frame size for measuring motion
GRID_W, GRID_H = 16, 9   # motion is summarised per grid cell
PIXEL_THRESH = 15        # grey-level change that counts as "moved"


def _key(path, tag):
    st = os.stat(path)
    h = hashlib.sha1(f"{path}|{st.st_size}|{tag}".encode()).hexdigest()[:12]
    return f"{os.path.basename(path)}.{tag}.{h}.npy"


def _frames(path, vf, w, h):
    """Yields grey frames of size (h, w) at FPS per second."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"fps={FPS},{vf},format=gray",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {path}")


def motion_grid(path, cache_dir):
    """(N, 9, 16) float16: share of pixels that moved in each grid cell, per sample."""
    out = os.path.join(cache_dir, _key(path, "motion"))
    if os.path.exists(out):
        return np.load(out, mmap_mode="r")
    prev, rows = None, []
    for f in _frames(path, f"scale={W}:{H}:flags=area", W, H):
        f = f.astype(np.int16)
        if prev is None:
            rows.append(np.zeros((GRID_H, GRID_W), np.float16))
        else:
            moved = np.abs(f - prev) > PIXEL_THRESH
            rows.append(moved.reshape(GRID_H, H // GRID_H, GRID_W, W // GRID_W).mean(axis=(1, 3)).astype(np.float16))
        prev = f
    np.save(out + ".tmp.npy", np.stack(rows))
    os.replace(out + ".tmp.npy", out)
    return np.load(out, mmap_mode="r")


def crop_frames(path, crop, cache_dir):
    """(N, h, w) uint8: a full-resolution crop (x, y, w, h) of every sample."""
    x, y, w, h = crop
    out = os.path.join(cache_dir, _key(path, f"crop{x}_{y}_{w}_{h}"))
    if os.path.exists(out):
        return np.load(out, mmap_mode="r")
    a = np.stack(list(_frames(path, f"crop={w}:{h}:{x}:{y}", w, h)))
    np.save(out + ".tmp.npy", a)
    os.replace(out + ".tmp.npy", out)
    return np.load(out, mmap_mode="r")


def recording_motion(rec, cache_dir):
    """A recording's motion, its files joined end to end by their real length."""
    return np.concatenate([np.asarray(motion_grid(f.path, cache_dir)) for f in rec.files])


def recording_crop(rec, crop, cache_dir):
    """A recording's crops, padded or trimmed per file to line up with its motion samples."""
    parts = []
    for f in rec.files:
        a = np.asarray(crop_frames(f.path, crop, cache_dir))
        n = len(motion_grid(f.path, cache_dir))
        if len(a) < n:
            a = np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)])
        parts.append(a[:n])
    return np.concatenate(parts)


def file_lengths(rec, cache_dir):
    """Length in seconds of each file of a recording, as sampled."""
    return [len(motion_grid(f.path, cache_dir)) / FPS for f in rec.files]


def extract_all(recordings, cache_dir, jobs=6, crops=None, progress=print):
    """Caches motion (and any requested crops) for every file, several files at a time.
    crops: {camera: [(x, y, w, h), ...]}."""
    os.makedirs(cache_dir, exist_ok=True)
    tasks = [(f, cam) for cam, recs in recordings.items() for r in recs for f in r.files]

    def work(task):
        f, cam = task
        motion_grid(f.path, cache_dir)
        for c in (crops or {}).get(cam, []):
            crop_frames(f.path, c, cache_dir)
        return f

    done = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in ex.map(work, tasks):
            done += 1
            progress(f"  [{done}/{len(tasks)}] cam {f.camera}: {f.name}")
