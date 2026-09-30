"""Processing one game: sync, scoreboard, camera choice, and rendering."""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import time

import cv2
import numpy as np

from . import director, motion, scoreboard, sync
from .motion import FPS

OUT_FPS = 30


def log(msg):
    print(msg, flush=True)


class Game:
    def __init__(self, game, config, rink_dir, cache_dir, trusted):
        self.g = game
        self.config = config
        self.rink_dir = rink_dir
        self.cache = cache_dir
        self.recs = game["recordings"]
        self.trusted = trusted
        self.grids = {c: motion.recording_motion(r, cache_dir) for c, r in self.recs.items()}
        self.lengths = {c: motion.file_lengths(r, cache_dir) for c, r in self.recs.items()}
        self.notes = []

    # -- time ------------------------------------------------------------------------------
    def expected_lag(self, ref, cam):
        """Lag (samples) from the camera clocks when both are believable, else None."""
        r, c = self.recs[ref], self.recs[cam]
        if r.id in self.trusted and c.id in self.trusted:
            return (r.clock - c.clock).total_seconds() * FPS
        return None

    def motion_lag(self, ref, cam, spread_s=600):
        e = self.expected_lag(ref, cam)
        lags = sync.lags_around(e, spread_s) if e is not None else None
        return sync.best_lag(sync.sync_signal(self.grids[ref]), sync.sync_signal(self.grids[cam]), lags)

    def sync_end(self, anchor, cams):
        """Offsets (s) of each camera at this end relative to the anchor camera."""
        out = {anchor: 0.0}
        for c in cams:
            if c == anchor:
                continue
            lag, strength = self.motion_lag(anchor, c)
            out[c] = lag / FPS
            log(f"    cam {c} vs cam {anchor}: {lag / FPS:+.1f} s (match {strength:.2f}x)")
            if strength < 1.3:
                self.notes.append(f"weak sync for cam {c} ({strength:.2f}x): check the sync image")
        return out

    def locate(self, cam, t):
        """(file path, seconds into file) for time t on a camera's recording, or (None, None)."""
        if t < 0:
            return None, None
        for f, n in zip(self.recs[cam].files, self.lengths[cam]):
            if t < n:
                return f.path, t
            t -= n
        return None, None

    # -- the whole game ------------------------------------------------------------------------
    def run(self, out_dir, stem):
        t_start = time.time()
        cfg = self.config
        ends = {c: cfg["cameras"][c]["end"] for c in self.recs if c in cfg["cameras"]}
        sb_cam = cfg["scoreboard"]["camera"] if cfg["scoreboard"]["camera"] in self.recs else None
        far_cam = cfg["far_scoreboard"]["camera"] if cfg["far_scoreboard"]["camera"] in self.recs else None
        report = {"game": self.g["game"], "name": self.g["name"], "start": self.g["start"]}

        # 1. scoreboard: running/stopped, periods, goals (on the scoreboard camera's time)
        mask = goals = periods = None
        if sb_cam:
            log(f"  reading the scoreboard (cam {sb_cam})")
            board = scoreboard.Scoreboard(cfg, self.rink_dir)
            a = motion.recording_crop(self.recs[sb_cam], cfg["scoreboard"]["crop"], self.cache)
            sh = board.shifts(a)
            # reading every frame is slow: keep the per-frame results for re-runs
            key = hashlib.sha1(json.dumps([self.recs[sb_cam].id, cfg["scoreboard"], len(a)]).encode()).hexdigest()[:12]
            path = os.path.join(self.cache, f"clock_{key}.npz")
            if os.path.exists(path):
                saved = np.load(path)
                vals, sims = saved["vals"], saved["sims"]
            else:
                vals, sims = board.clock_values(a, sh), board.seconds_similarity(a, sh)
                np.savez(path, vals=vals, sims=sims)
            pts = scoreboard.trusted_points(vals)
            tstate = scoreboard.tick_state(sims)
            mask, periods = scoreboard.play_mask(pts, tstate, board.period_s)
            goals = scoreboard.goal_stoppages(board, a, sh, mask)
            report["clock_running_min"] = round(mask.sum() / FPS / 60, 1)
            report["periods"] = [[k0 / FPS, k1 / FPS] for k0, k1 in periods]
            report["goals"] = [[k / FPS, team] for k, team in goals]
            log(f"    clock ran {mask.sum() / FPS / 60:.1f} min in {len(periods)} period(s); "
                f"goals: {', '.join(f'{t} at {k / FPS / 60:.1f} min' for k, t in goals) or 'none found'}")
            if len(periods) == 0:
                self.notes.append("no periods found on the scoreboard: action-only version skipped")
                mask = None
        else:
            self.notes.append(f"no footage from the scoreboard camera {cfg['scoreboard']['camera']}: "
                              "action-only version skipped")

        # 2. sync: within each end by motion, between the ends by the scoreboards
        log("  syncing cameras")
        right = [c for c in self.recs if ends.get(c) == ends.get(sb_cam or "", "right")]
        left = [c for c in self.recs if c not in right]
        anchor_r = sb_cam or (right[0] if right else None)
        anchor_l = far_cam if far_cam in left else (left[0] if left else None)
        offsets = {}
        if anchor_r:
            offsets.update(self.sync_end(anchor_r, right))
        if anchor_l:
            rel = self.sync_end(anchor_l, left)
            if anchor_r is None:
                offsets.update(rel)
            else:
                bridge = None
                if mask is not None and far_cam == anchor_l:
                    sims = np.concatenate([
                        _fit(scoreboard.far_board_similarity(f.path, cfg, self.rink_dir, self.cache), int(n * FPS))
                        for f, n in zip(self.recs[far_cam].files, self.lengths[far_cam])])
                    far_run = scoreboard.fill_unknown(scoreboard.tick_state(sims)).astype(np.float32)
                    e = self.expected_lag(anchor_r, far_cam)
                    lags = sync.lags_around(e, 300) if e is not None else None
                    m = mask.astype(np.float32)
                    # running/stopped stretches last tens of seconds, so compare the best
                    # match with the best one at least 30 s away
                    lag, strength = sync.best_lag(m - m.mean(), far_run - far_run.mean(), lags, separation_s=30)
                    log(f"    cam {far_cam} vs cam {anchor_r} by the scoreboard clocks: {lag / FPS:+.1f} s (match {strength:.2f}x)")
                    if strength >= 1.3:
                        bridge = lag / FPS
                    else:
                        self.notes.append(f"scoreboard clocks didn't line the ends up clearly ({strength:.2f}x)")
                if bridge is None:
                    lag, strength = self.motion_lag(anchor_r, anchor_l)
                    bridge = lag / FPS
                    self.notes.append(f"ends lined up by motion only ({strength:.2f}x): check the sync image")
                offsets.update({c: bridge + v for c, v in rel.items()})
        report["offsets"] = offsets

        # 3. the game's time span: where the believable-clock (or all) cameras overlap the game
        spans = {c: (-offsets[c], sum(self.lengths[c]) - offsets[c]) for c in offsets}
        core = [c for c in offsets if self.recs[c].id in self.trusted] or list(offsets)
        t0 = max(min(spans[c][0] for c in core), min(s[0] for s in spans.values()))
        t1 = min(max(spans[c][1] for c in core), max(s[1] for s in spans.values()))
        n = int((t1 - t0) * FPS)

        # 4. camera choice
        log("  choosing cameras")
        cams, sc = director.scores({c: self.grids[c] for c in offsets}, offsets, t0, n, cfg)
        path = director.choose(sc, cfg)
        shot_list = [[t0 + a / FPS, t0 + b / FPS, cams[c]] for a, b, c in director.shots(path)]
        report["shots"] = len(shot_list)
        _chart(cams, sc, path, os.path.join(out_dir, f"{stem}_camera_chart.png"))

        # 5. what to keep for the action-only version
        keep = None
        if mask is not None:
            k = cfg["keep"]
            keep = [[a / FPS - k["pad_before_s"], b / FPS + k["pad_after_s"]] for a, b in scoreboard.segments(mask)]
            keep += [[s / FPS, s / FPS + k["goal_after_s"]] for s, _ in goals]
            keep = _merge(sorted(keep), k["join_gap_s"])
            report["action_min"] = round(sum(min(b, t1) - max(a, t0) for a, b in keep if b > t0 and a < t1) / 60, 1)

        # 6. check image, then render
        self.sync_image(offsets, t0, t1, os.path.join(out_dir, f"{stem}_sync_check.jpg"))
        log("  rendering the full game")
        self.render(shot_list, None, os.path.join(out_dir, f"{stem}_multicam.mp4"), offsets)
        if keep:
            log("  rendering the action-only version")
            self.render(shot_list, keep, os.path.join(out_dir, f"{stem}_multicam_action.mp4"), offsets)
        report["notes"] = self.notes
        report["minutes_taken"] = round((time.time() - t_start) / 60, 1)
        json.dump(report, open(os.path.join(out_dir, f"{stem}_report.json"), "w"), indent=1)
        return report

    # -- output ----------------------------------------------------------------------------------
    def sync_image(self, offsets, t0, t1, out):
        """All cameras at the same moment (a third of the way into the game), for checking sync."""
        t = t0 + (t1 - t0) / 3
        tiles = []
        for c in sorted(offsets, key=lambda c: (self.config["cameras"].get(c, {}).get("end", ""), c)):
            path, ft = self.locate(c, t + offsets[c])
            if path is None:
                continue
            r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ft:.2f}", "-i", path, "-frames:v", "1",
                                "-vf", "scale=640:360", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               capture_output=True)
            if len(r.stdout) == 640 * 360 * 3:
                im = np.frombuffer(r.stdout, np.uint8).reshape(360, 640, 3).copy()
                cv2.putText(im, f"cam {c}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                tiles.append(im)
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))

    def render(self, shot_list, keep, out, offsets, jobs=3):
        spans = []
        for a, b, cam in shot_list:
            for k0, k1 in (keep or [[a, b]]):
                lo, hi = max(a, k0), min(b, k1)
                if hi > lo:
                    spans.append((lo, hi, cam))
        pieces = []
        for a, b, cam in sorted(spans):
            t = a
            while t < b - 1e-6:
                path, ft = self.locate(cam, t + offsets[cam])
                if path is None:
                    break
                n = dict(zip((f.path for f in self.recs[cam].files), self.lengths[cam]))[path]
                piece = min(b - t, n - ft)
                if piece <= 1e-6:
                    break
                frames = int(round(piece * OUT_FPS))
                if frames > 0:
                    pieces.append((path, ft, frames))
                t += piece
        work = os.path.abspath(out + ".parts")
        os.makedirs(work, exist_ok=True)

        def cut(i):
            path, ft, frames = pieces[i]
            # named after what's in it, so an interrupted render can reuse finished pieces
            tag = hashlib.sha1(f"{path}|{ft:.3f}|{frames}".encode()).hexdigest()[:10]
            p = os.path.join(work, f"part{i:05d}_{tag}.mp4")
            if not os.path.exists(p):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{ft:.3f}", "-i", path, "-frames:v", str(frames),
                                "-an", "-vf", "scale=1920:1080", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                                "-r", str(OUT_FPS), "-f", "mp4", p + ".tmp"], check=True)
                os.replace(p + ".tmp", p)
            return p

        t_start = time.time()
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            parts = []
            for i, p in enumerate(ex.map(cut, range(len(pieces))), 1):
                parts.append(p)
                if i % 50 == 0 or i == len(pieces):
                    log(f"    {i}/{len(pieces)} pieces ({time.time() - t_start:.0f}s)")
        lst = os.path.join(work, "list.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{p}'\n" for p in parts)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", "-movflags", "+faststart", out], check=True)
        for name in os.listdir(work):      # finished: remove the pieces (and any from older runs)
            os.remove(os.path.join(work, name))
        os.rmdir(work)
        log(f"    saved {out}")


def _fit(a, n):
    """Pad (with NaN) or trim a per-sample array to n samples."""
    a = np.asarray(a, np.float32)
    return a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), np.nan, np.float32)])


def _merge(spans, join_gap):
    out = []
    for a, b in spans:
        if out and a <= out[-1][1] + join_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


COLOURS = [(180, 90, 20), (230, 160, 60), (140, 200, 90), (30, 110, 230), (60, 170, 250), (90, 60, 200),
           (120, 120, 120), (200, 80, 160)]


def _chart(cams, sc, path, out):
    """One row per camera: its score as grey, the chosen camera as a coloured bar."""
    n = sc.shape[1]
    W, row_h, pad = 1800, 40, 110
    img = np.full((row_h * len(cams) + 60, W + pad + 10, 3), 255, np.uint8)
    cols = np.linspace(0, n, W + 1).astype(int)
    for i, c in enumerate(cams):
        y = 10 + i * row_h
        cv2.putText(img, f"cam {c}", (10, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        for x in range(W):
            seg = sc[i, cols[x]:cols[x + 1]]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                g = int(235 - 150 * float(np.clip(seg.mean() / 2, 0, 1)))
                img[y + 2:y + row_h // 2, pad + x] = (g, g, g)
            if (path[cols[x]:cols[x + 1]] == i).mean() > 0.5:
                img[y + row_h // 2:y + row_h - 4, pad + x] = COLOURS[i % len(COLOURS)]
    for m in range(0, int(n / FPS / 60) + 1, 5):
        x = pad + int(m * 60 * FPS / n * W)
        cv2.line(img, (x, 10), (x, img.shape[0] - 45), (200, 200, 200), 1)
        cv2.putText(img, f"{m}m", (x - 10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out, img)
