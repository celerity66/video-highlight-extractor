"""Finding video files in the camera folders and joining them into recordings."""
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta

VIDEO_EXT = (".mp4", ".mov")

# Camera file names carry the camera's own clock time:
#   2026_0905_112156_001.MP4, tbhc2021_0101_104930_007.MP4 (letters added by hand)
#   20260905112617_00001.MP4
NAME_PATTERNS = [
    re.compile(r"^[A-Za-z]*(\d{4})_(\d{4})_(\d{6})_\d+\.(mp4|mov)$", re.I),
    re.compile(r"^(\d{4})(\d{4})(\d{6})_\d+\.(mp4|mov)$", re.I),
]
CHAIN_TOLERANCE_S = 10.0   # a file continues a recording if it starts within this of the previous end


@dataclass
class VideoFile:
    camera: str
    path: str
    clock: datetime | None   # start time by the camera's own clock (from the file name)
    duration: float          # seconds

    @property
    def name(self):
        return os.path.basename(self.path)


@dataclass
class Recording:
    camera: str
    files: list = field(default_factory=list)

    @property
    def clock(self):
        return self.files[0].clock

    @property
    def duration(self):
        return sum(f.duration for f in self.files)

    @property
    def id(self):
        return f"{self.camera}:{self.files[0].name}"


def name_clock(fname):
    for p in NAME_PATTERNS:
        m = p.match(fname)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2) + m.group(3), "%Y%m%d%H%M%S")
            except ValueError:
                return None
    return None


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def scan(footage_dir, cameras, cache_dir):
    """{camera: [Recording]} for each camera folder named in `cameras`.
    File lengths are cached, since probing files on a network or Windows drive is slow."""
    cache_path = os.path.join(cache_dir, "durations.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = {}
    for cam in cameras:
        folder = os.path.join(footage_dir, cam)
        if not os.path.isdir(folder):
            continue
        files = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(VIDEO_EXT):
                continue
            path = os.path.join(folder, fname)
            key = f"{path}|{os.path.getsize(path)}"
            if key not in cache:
                cache[key] = probe_duration(path)
            if cache[key]:
                files.append(VideoFile(cam, path, name_clock(fname), cache[key]))
        out[cam] = join_recordings(cam, files)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(cache, open(cache_path, "w"), indent=0)
    return out


def join_recordings(cam, files):
    """Cameras split long recordings into ~25-minute files. A file continues the previous
    one if it starts where that one ended by the camera's clock (even a wrong clock is
    consistent with itself). Files without a readable time stay on their own."""
    timed = sorted((f for f in files if f.clock), key=lambda f: (f.clock, f.name))
    recs = []
    for f in timed:
        if recs:
            prev = recs[-1].files[-1]
            gap = (f.clock - (prev.clock + timedelta(seconds=prev.duration))).total_seconds()
            if abs(gap) <= CHAIN_TOLERANCE_S:
                recs[-1].files.append(f)
                continue
        recs.append(Recording(cam, [f]))
    recs += [Recording(cam, [f]) for f in files if not f.clock]
    return recs


def trusted_clock(recordings):
    """Which recordings have a believable camera clock: within 30 days of the median time
    across all recordings (cameras reset to 2021 or 2024 after losing power)."""
    times = sorted(r.clock for rs in recordings.values() for r in rs if r.clock)
    if not times:
        return set()
    median = times[len(times) // 2]
    return {r.id for rs in recordings.values() for r in rs
            if r.clock and abs((r.clock - median).total_seconds()) < 30 * 86400}
