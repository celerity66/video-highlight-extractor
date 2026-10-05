"""Processing one game: sync, scoreboard, camera choice, and rendering."""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import time

import cv2
import numpy as np

from . import clocks, director, game_time, motion, scoreboard, sync, timeline
from .motion import FPS

OUT_FPS = 30
WARMUP_AFTER_FACEOFF_S = 8.0     # warmup_camera: stay on it this long after the first faceoff
LABEL_MIN = 0.6      # how well the scoreboard label must be found (0.84 when right, <0.5 at other rinks)
SYNC_MIN = 1.3       # how clearly a camera must match to count as synced


def log(msg):
    print(msg, flush=True)


class Game:
    def __init__(self, game, config, rink_dir, cache_dir, trusted, force=False, check_only=False, action_only=False):
        self.g = game
        self.config = config
        self.rink_dir = rink_dir
        self.cache = cache_dir
        self.recs = game["recordings"]
        self.trusted = trusted
        self.grids = {c: motion.recording_motion(r, cache_dir) for c, r in self.recs.items()}
        self.lengths = {c: motion.file_lengths(r, cache_dir) for c, r in self.recs.items()}
        self.notes = []       # worth knowing
        self.problems = []    # serious: the game isn't rendered unless forced
        self.force = force
        self.check_only = check_only
        self.action_only = action_only

    # -- time ------------------------------------------------------------------------------
    def expected_lag(self, ref, cam):
        """Lag (samples) from the camera clocks when both are believable, else None."""
        r, c = self.recs[ref], self.recs[cam]
        if r.id in self.trusted and c.id in self.trusted:
            return (r.clock - c.clock).total_seconds() * FPS
        return None

    def game_coverage(self, cam):
        """How much of the game (seconds) a camera's recording covers, judged by the camera
        clocks that are believable; cameras with wrong clocks count as covering none."""
        r = self.recs[cam]
        good = [x for x in self.recs.values() if x.id in self.trusted]
        if r.id not in self.trusted or not good:
            return 0.0
        starts = sorted(x.clock.timestamp() for x in good)
        ends = sorted(x.clock.timestamp() + x.duration for x in good)
        g0, g1 = starts[len(starts) // 2], ends[len(ends) // 2]      # typical start and end
        c0 = r.clock.timestamp()
        return max(0.0, min(c0 + r.duration, g1) - max(c0, g0))

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
        sb_cfg = cfg.get("scoreboard")
        sb_cam = sb_cfg["camera"] if sb_cfg and sb_cfg["camera"] in self.recs else None
        report = {"game": self.g["game"], "name": self.g["name"], "start": self.g["start"]}

        # 1. scoreboard: running/stopped, periods, goals (on the scoreboard camera's time)
        mask = goals = periods = None
        if sb_cam:
            log(f"  reading the scoreboard (cam {sb_cam})")
            board = scoreboard.Scoreboard(cfg, self.rink_dir)
            a = motion.recording_crop(self.recs[sb_cam], cfg["scoreboard"]["crop"], self.cache)
            sh = board.shifts(a)
            if board.label_match < LABEL_MIN:
                self.problems.append(f"camera {sb_cam} doesn't show the scoreboard described in the rink settings "
                                     f"(match {board.label_match:.2f}): this game may need its own rink settings")
                sb_cam = None            # nothing to read; sync the ends without the scoreboards
        if sb_cam:
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
                self.problems.append("no game clock could be read from the scoreboard")
                mask = None
        elif self.problems:
            pass                         # the scoreboard problem is already reported

        # 2. sync every camera from all the evidence: clocks seen by any camera (the rink's
        # readable scoreboard, where there is one, is the most precise), motion at the same
        # end, opposite activity between the ends, and sync points set by hand
        log("  finding scoreboard clocks")
        runs, clock_info = {}, {}
        with cf.ThreadPoolExecutor(max_workers=len(self.recs)) as ex:
            found = dict(ex.map(lambda c: (c, clocks.clock_state(self.recs[c].files, self.lengths[c], self.cache)),
                                list(self.recs)))
        raw_states = {}
        for c, (st, info) in found.items():
            if st is not None:
                raw_states[c] = st
                runs[c] = scoreboard.fill_unknown(st).astype(np.float32)
            clock_info[c] = info
            log(f"    cam {c}: " + (f"clock found (ticking {info['ticking']:.0%} of the time)" if st is not None else info))
        if mask is not None:
            runs[sb_cam] = mask.astype(np.float32)      # read digit by digit: better than ticks alone
        report["clocks"] = {c: (i if isinstance(i, str) else "found") for c, i in clock_info.items()}
        log("  syncing cameras")
        links = timeline.pair_links(self, runs, manual=self.g.get("sync_points"))
        # start from the readable scoreboard's camera, else the camera that covers most of
        # the game (a camera that only caught part of it makes a poor starting point)
        reference = sb_cam if mask is not None else max(self.recs, key=self.game_coverage)
        offsets, used, sync_notes, conflicts = timeline.solve(list(self.recs), links, reference)
        for msg in conflicts:
            self.problems.append(f"the sync doesn't hold together: {msg}. Add a sync point in games.csv, "
                                 "or remove a camera that only caught part of the game")
        for a, b, strength, kind in used:
            log(f"    cam {b} linked to cam {a} by {kind} ({strength:.2f}x)")
        self.notes += sync_notes
        for c in self.recs:
            if c not in offsets:
                self.problems.append(f"camera {c} couldn't be linked to the other cameras: add a sync point for it in "
                                     "games.csv, or remove it from this game's recordings")
        report["offsets"] = offsets
        report["links"] = [list(u) for u in used]
        if len(offsets) < 2:
            self.problems.append("fewer than two cameras could be synced")

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

        # 5. game time for the action-only version: from the readable scoreboard if there
        # is one (on the reference camera's time, from 0), else from the clocks seen ticking
        mask_t0 = 0.0
        starts = self.g.get("period_starts") or []
        if mask is None and starts:
            missing = sorted({c for c, _ in starts if c not in offsets})
            if missing:
                self.problems.append(f"period starts given for camera(s) {', '.join(missing)}, which couldn't be synced")
            else:
                st = timeline.combined_running({c: v for c, v in raw_states.items() if c in offsets}, offsets, t0, n)
                idx = [int(round((sec - offsets[c] - t0) * FPS)) for c, sec in starts]
                mask, ps = game_time.running_from_starts(st, idx, cfg.get("period_minutes", 13) * 60)
                mask_t0, goals = t0, []
                report["clock_running_min"] = round(mask.sum() / FPS / 60, 1)
                report["periods"] = [[t0 + a / FPS, t0 + b / FPS] for a, b in ps]
                log(f"    game time from the period starts given: {mask.sum() / FPS / 60:.1f} min in {len(ps)} periods")
        if mask is None and raw_states and not starts:
            st = timeline.combined_running({c: v for c, v in raw_states.items() if c in offsets}, offsets, t0, n)
            gmask, ps, why = game_time.running_in_periods(st, cfg.get("period_minutes", 13) * 60)
            if why:
                self.notes.append(f"action-only version skipped: {why}")
            else:
                mask, mask_t0, goals = gmask, t0, []
                report["clock_running_min"] = round(mask.sum() / FPS / 60, 1)
                report["periods"] = [[t0 + a / FPS, t0 + b / FPS] for a, b in ps]
                log(f"    game time from the clocks: {mask.sum() / FPS / 60:.1f} min in {len(ps)} periods")
                self.notes.append("no readable scoreboard, so goals aren't detected (no extra time kept after goals)")
        elif mask is None:
            self.notes.append("no scoreboard clock seen by any camera: action-only version skipped")
        keep = None
        if mask is not None:
            k = cfg["keep"]
            keep = [[mask_t0 + a / FPS - k["pad_before_s"], mask_t0 + b / FPS + k["pad_after_s"]]
                    for a, b in scoreboard.segments(mask)]
            keep += [[mask_t0 + s / FPS, mask_t0 + s / FPS + k["goal_after_s"]] for s, _ in goals]
            keep = _merge(sorted(keep), k["join_gap_s"])
            report["action_min"] = round(sum(min(b, t1) - max(a, t0) for a, b in keep if b > t0 and a < t1) / 60, 1)

        # our team's warm-up: hold the camera at our end from the start until just after the
        # first faceoff, rather than following the play (which is often the other team)
        wc = self.g.get("warmup_camera")
        if wc:
            if wc not in offsets:
                self.problems.append(f"warmup_camera {wc} couldn't be synced")
            elif not report.get("periods"):
                self.notes.append(f"warmup_camera {wc} not used: the first faceoff isn't known "
                                  "(add period_starts in games.csv)")
            else:
                first = report["periods"][0][0]
                covers = (-offsets[wc], sum(self.lengths[wc]) - offsets[wc])
                shot_list = director.hold(shot_list, wc, t0, first + WARMUP_AFTER_FACEOFF_S, covers,
                                          cfg["director"]["min_shot_s"])
                if covers[0] > t0 + 1:
                    self.notes.append(f"warmup_camera {wc} started recording {covers[0] - t0:.0f} s into the video")
                log(f"    holding cam {wc} until {WARMUP_AFTER_FACEOFF_S:.0f} s after the first faceoff")

        # 6. check image, then render (unless something is clearly wrong)
        self.sync_image(offsets, t0, t1, os.path.join(out_dir, f"{stem}_sync_check.jpg"))
        self.sync_video(offsets, t0, t1, os.path.join(out_dir, f"{stem}_sync_check.mp4"))
        report["problems"] = self.problems
        if self.check_only or (self.problems and not self.force):
            report["notes"] = self.notes
            report["rendered"] = False
            # a check-only run doesn't replace the report of a game already rendered
            name = f"{stem}_check_report.json" if self.check_only else f"{stem}_report.json"
            json.dump(report, open(os.path.join(out_dir, name), "w"), indent=1)
            return report
        if not self.action_only:
            log("  rendering the full game")
            self.render(shot_list, None, os.path.join(out_dir, f"{stem}_multicam.mp4"), offsets)
        elif not keep:
            self.notes.append("--action-only, but no game time could be worked out, so nothing was rendered")
        if keep:
            log("  rendering the action-only version")
            self.render(shot_list, keep, os.path.join(out_dir, f"{stem}_multicam_action.mp4"), offsets)
        report["notes"] = self.notes
        report["rendered"] = True
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

    def sync_video(self, offsets, t0, t1, out, seconds=20):
        """20 s of every camera side by side from the same moment, to check the sync by eye."""
        t = t0 + (t1 - t0) / 3
        cams = sorted(offsets, key=lambda c: (self.config["cameras"].get(c, {}).get("end", ""), c))
        inputs, labels = [], []
        for c in cams:
            path, ft = self.locate(c, t + offsets[c])
            if path is None:
                continue
            inputs += ["-ss", f"{ft:.2f}", "-t", str(seconds), "-i", path]
            labels.append(c)
        if len(labels) < 2:
            return
        cols = 3 if len(labels) > 4 else 2
        while len(labels) % cols:
            inputs += ["-f", "lavfi", "-t", str(seconds), "-i", "color=black:s=640x360:r=30"]
            labels.append(None)
        parts = [f"[{i}:v]scale=640:360,setsar=1" + (f",drawtext=text='cam {c}':fontsize=28:fontcolor=yellow:"
                 "box=1:boxcolor=black@0.6:x=10:y=10" if c else "") + f"[v{i}]" for i, c in enumerate(labels)]
        layout = "|".join(f"{(i % cols) * 640}_{(i // cols) * 360}" for i in range(len(labels)))
        graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(labels))) + \
            f"xstack=inputs={len(labels)}:layout={layout}[out]"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", out], check=True)

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
