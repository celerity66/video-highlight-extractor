"""Working out which recordings belong to which game, and the review list (games.csv)."""
import csv
import os

import numpy as np

from . import motion, sync

MIN_RECORDING_MIN = 5       # ignore recordings shorter than this (accidental starts)
MATCH_STRENGTH = 1.5        # how clearly a lone wrong-clock recording must match a game by motion
SHIFT_TOLERANCE_MIN = 20    # cameras start recording up to this long before/after each other
MIN_SHARED_MIN = 10         # a recording also joins another game it overlaps by this much


def build_games(recordings, trusted, config, cache_dir, progress=print):
    """Games from recordings with believable clocks (overlapping in time), then each
    wrong-clock recording is matched to games by its motion pattern."""
    ends = {c: v["end"] for c, v in config["cameras"].items()}
    usable = {cam: [r for r in rs if r.duration >= MIN_RECORDING_MIN * 60] for cam, rs in recordings.items()}
    good = sorted((r for rs in usable.values() for r in rs if r.id in trusted), key=lambda r: r.clock)

    games = []
    for r in good:
        start = r.clock.timestamp()
        end = start + r.duration
        best = None
        for g in games:
            overlap = min(end, g["end"]) - max(start, g["start"])
            if overlap >= 0.5 * r.duration and (best is None or overlap > best[0]):
                best = (overlap, g)
        if best and r.camera not in best[1]["recordings"]:
            g = best[1]
            g["start"], g["end"] = min(g["start"], start), max(g["end"], end)
        else:
            g = {"start": start, "end": end, "recordings": {}, "notes": []}
            games.append(g)
        g["recordings"][r.camera] = r

    signals = {}

    def sig(r):
        if r.id not in signals:
            signals[r.id] = sync.sync_signal(motion.recording_motion(r, cache_dir))
        return signals[r.id]

    def motion_match(r, g):
        """Strength of the best motion match between r and the game's cameras
        (cameras at the same end first: they match most strongly)."""
        others = sorted(g["recordings"].values(), key=lambda o: ends.get(o.camera) != ends.get(r.camera))
        return max(((sync.best_lag(sig(o), sig(r))[1], o.camera) for o in others[:3]), default=(0, None))

    # A camera whose clock was reset keeps counting from the wrong time, so all its
    # recordings since the reset are off by the same amount. For each camera, find the
    # shift that puts the most of its wrong-clock recordings at the start of a game.
    for cam in usable:
        wrong = [r for r in usable[cam] if r.id not in trusted and r.clock]
        if not wrong:
            continue
        free = [g for g in games if cam not in g["recordings"]]
        best = None
        for r0 in wrong:
            for g0 in free:
                shift = g0["start"] - r0.clock.timestamp()
                hits = [(r, g) for r in wrong for g in free
                        if abs(r.clock.timestamp() + shift - g["start"]) <= SHIFT_TOLERANCE_MIN * 60]
                # one game per recording and one recording per game: keep the closest
                pairs, used_r, used_g = [], set(), set()
                for r, g in sorted(hits, key=lambda h: abs(h[0].clock.timestamp() + shift - h[1]["start"])):
                    if r.id not in used_r and id(g) not in used_g:
                        pairs.append((r, g))
                        used_r.add(r.id)
                        used_g.add(id(g))
                if best is None or len(pairs) > len(best[1]):
                    best = (shift, pairs)
        shift, pairs = best if best else (0, [])
        if len(pairs) == 1:
            # a single recording can't confirm a clock shift: it must match by motion too
            strength, other = motion_match(*pairs[0])
            if strength < MATCH_STRENGTH:
                pairs = []
        for r, g in pairs:
            strength, other = motion_match(r, g)       # before adding r, so it isn't compared with itself
            g["recordings"][cam] = r
            g["notes"].append(f"cam {cam} matched by its reset clock ({_describe(shift)}, "
                              f"motion check {strength:.1f}x vs cam {other})")
            progress(f"  cam {cam} {r.files[0].name} -> game {games.index(g) + 1} "
                     f"(clock {_describe(shift)}, motion {strength:.1f}x)")
        for r in wrong:
            if not any(r is pr for pr, _ in pairs):
                progress(f"  cam {cam} {r.files[0].name}: no match to any game")

    # a single-camera "game" is a camera left recording, not a game
    for g in games:
        if len(g["recordings"]) < 2:
            g["render"] = "no"
            g["notes"].append("only one camera: probably a camera left recording after a game")
    # a recording that runs on into a real game also covers that game's missing camera
    for r in good:
        for g in games:
            if r.camera in g["recordings"] or g.get("render") == "no":
                continue
            start = r.clock.timestamp()
            overlap = min(start + r.duration, g["end"]) - max(start, g["start"])
            if overlap >= MIN_SHARED_MIN * 60:
                g["recordings"][r.camera] = r
                g["notes"].append(f"cam {r.camera} only covers {overlap / 60:.0f} min of this game")
    return games


def _describe(shift_s):
    days = shift_s / 86400
    return f"{days:+.2f} days out" if abs(days) >= 1 else f"{shift_s / 3600:+.1f} h out"


FIELDS = ["game", "name", "render", "rink", "start", "length_min", "cameras", "recordings", "sync_points", "period_starts", "notes"]


def write_csv(games, path, rink):
    from datetime import datetime
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for i, g in enumerate(games, 1):
            recs = g["recordings"]
            w.writerow({
                "game": i,
                "name": "",
                "render": g.get("render", "yes"),
                "rink": rink,
                "start": datetime.fromtimestamp(g["start"]).strftime("%Y-%m-%d %H:%M"),
                "length_min": round((g["end"] - g["start"]) / 60),
                "cameras": " ".join(sorted(recs, key=lambda c: int(c) if c.isdigit() else c)),
                "recordings": "; ".join(f"{c}={recs[c].files[0].name}" for c in sorted(recs, key=lambda c: int(c) if c.isdigit() else c)),
                "sync_points": "",
                "period_starts": "",
                "notes": "; ".join(g["notes"]),
            })


def read_csv(path, recordings):
    """Games from the (possibly edited) review list. Each recording is named by its
    camera and first file, e.g. 5=2026_0905_112453_001.MP4."""
    by_id = {r.id: r for rs in recordings.values() for r in rs}
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            recs = {}
            for part in filter(None, (p.strip() for p in row["recordings"].split(";"))):
                cam, fname = (s.strip() for s in part.split("=", 1))
                rid = f"{cam}:{fname}"
                if rid not in by_id:
                    raise SystemExit(f"games.csv game {row['game']}: no recording starting with {fname} in camera folder {cam}")
                recs[cam] = by_id[rid]
            sync_points = parse_sync_points(row.get("sync_points") or "", recs, row["game"])
            period_starts = parse_moments(row.get("period_starts") or "", recs, row["game"], "period start")
            out.append({"game": row["game"], "name": row["name"].strip(), "render": row["render"].strip().lower(),
                        "rink": (row.get("rink") or "").strip(), "sync_points": sync_points,
                        "period_starts": period_starts,
                        "start": row["start"], "recordings": recs})
    return out


def parse_moments(text, recs, game_id, what):
    """Moments picked out by hand, e.g. "7=2026_0905_185418_005.MP4@05:40; 9=...@01:02:10"
    -> [(camera, seconds into that camera's recording)]. Times are minutes:seconds (or
    hours:minutes:seconds) into the named file, as shown by a video player."""
    out = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        try:
            cam, rest = (x.strip() for x in part.split("=", 1))
            fname, t = (x.strip() for x in rest.split("@", 1))
            secs = 0.0
            for piece in t.split(":"):
                secs = secs * 60 + float(piece)
        except ValueError:
            raise SystemExit(f"games.csv game {game_id}: can't read {what} '{part}' "
                             "(expected camera=file@minutes:seconds, e.g. 7=2026_0905_185418_005.MP4@05:40.0)")
        if cam not in recs:
            raise SystemExit(f"games.csv game {game_id}: {what} for camera {cam}, which isn't in this game")
        into = 0.0
        for f in recs[cam].files:
            if f.name == fname:
                out.append((cam, into + secs))
                break
            into += f.duration
        else:
            raise SystemExit(f"games.csv game {game_id}: {fname} isn't part of camera {cam}'s recording for this game")
    return out


def parse_sync_points(text, recs, game_id):
    """One moment picked out by hand in several cameras, e.g.
    "10=2026_0905_184631_005.MP4@03:12.4; 7=2026_0905_185418_005.MP4@05:40"
    -> {camera: seconds into that camera's recording}. Times are minutes:seconds
    (or hours:minutes:seconds) into the named file, as shown by a video player."""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        try:
            cam, rest = (x.strip() for x in part.split("=", 1))
            fname, t = (x.strip() for x in rest.split("@", 1))
            secs = 0.0
            for piece in t.split(":"):
                secs = secs * 60 + float(piece)
        except ValueError:
            raise SystemExit(f"games.csv game {game_id}: can't read sync point '{part}' "
                             "(expected camera=file@minutes:seconds, e.g. 7=2026_0905_185418_005.MP4@05:40.0)")
        if cam not in recs:
            raise SystemExit(f"games.csv game {game_id}: sync point for camera {cam}, which isn't in this game")
        into = 0.0
        for f in recs[cam].files:
            if f.name == fname:
                out[cam] = into + secs
                break
            into += f.duration
        else:
            raise SystemExit(f"games.csv game {game_id}: {fname} isn't part of camera {cam}'s recording for this game")
    if len(out) == 1:
        raise SystemExit(f"games.csv game {game_id}: a sync point needs the same moment in at least two cameras")
    return out


def output_name(game):
    """File-name stem: the name from the review list, or the date and time."""
    if game["name"]:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in game["name"]).strip("_")
        return f"{game['start'][:10]}_{safe}"
    return game["start"].replace(" ", "_").replace(":", "")
