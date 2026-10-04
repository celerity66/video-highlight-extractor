#!/usr/bin/env python3
"""
Goalie Highlight Extractor
---------------------------
Scans hockey video (e.g. GoPro footage focused on the goalie) for motion
activity, and cuts out the "standing around" segments, keeping only the
active play. Outputs individual clip files and (optionally) merged
highlight videos.

HOW IT WORKS
------------
1. Reads the video frame-by-frame (at a reduced sample rate for speed).
2. Computes frame-to-frame difference to estimate motion level.
3. Marks frames as "active" when motion exceeds a threshold.
4. (Config mode, optional) Checks how much of the MOVING stuff in the
   crease area matches your goalie's jersey/pad colors, and drops action
   where his colors aren't present.
5. Groups active frames into segments, merging ones close together and
   padding the start/end so plays aren't cut off abruptly.
6. Uses ffmpeg to losslessly (or re-encoded) extract those segments.

REQUIREMENTS
------------
    pip install opencv-python --break-system-packages
    ffmpeg (includes ffprobe):  sudo apt install ffmpeg
    Python 3.11+ for config mode (uses the built-in tomllib)

THREE WAYS TO RUN
-----------------
1) Single file:
    python3 goalie_highlight_extractor.py input.mp4 --dry-run

2) Folder (one camera), each recording processed separately:
    python3 goalie_highlight_extractor.py /path/to/VIDEO/6 --batch --merge

3) Config file (multiple cameras, per-period camera choice, full-game merge):
    # See which recordings each camera has (copy these labels into the config)
    python3 goalie_highlight_extractor.py --config goalie_config.toml --list

    # Dry run every game in the config (or just one with --game NAME)
    python3 goalie_highlight_extractor.py --config goalie_config.toml --dry-run

    # Cut clips and build one full-game highlight video per game
    python3 goalie_highlight_extractor.py --config goalie_config.toml

Common options (CLI flags override values in the config's [settings]):
    --threshold 15        Motion sensitivity (lower = more sensitive, more clips)
    --min-gap 3.0         Merge active segments separated by less than this many seconds
    --min-duration 1.5    Discard active segments shorter than this (avoid single-frame noise)
    --pad-before 5.0      Seconds of padding added before each segment
    --pad-after 5.0       Seconds of padding added after each segment
    --sample-rate 5       Analyze every Nth frame (higher = faster, less precise)
    --roi x,y,w,h         Only look for motion in this box (pixels, 0,0 = top-left);
                          several boxes: "x,y,w,h; x,y,w,h"
    --dry-run             Just print detected segments, don't cut anything
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: opencv-python is not installed.\n"
        "Install it with:\n"
        "    pip install opencv-python --break-system-packages\n"
    )


# ---------------------------------------------------------------------------
# File naming patterns / chapter grouping
# ---------------------------------------------------------------------------

# GoPro chaptered filenames look like GH010042.MP4, GH020042.MP4, GH030042.MP4
# (HERO 5+) or GOPR0042.MP4, GP010042.MP4, GP020042.MP4 (older models).
# The trailing 4-digit "media ID" (0042 above) ties chapters together; the
# leading part encodes chapter number + codec. We group by media ID and sort
# by chapter so they get concatenated in the right order.
GOPRO_PATTERNS = [
    re.compile(r"^GH(\d{2})(\d{4})\.(MP4|mp4)$"),  # GH010042.MP4 -> chapter 01, id 0042
    re.compile(r"^GP(\d{2})(\d{4})\.(MP4|mp4)$"),  # GP010042.MP4
    re.compile(r"^GOPR(\d{4})\.(MP4|mp4)$"),       # GOPR0042.MP4 -> chapter 0 (first file), id 0042
]

# Renamed-via-PowerRenamer pattern: "zzz_xx_Game.mp4" where zzz is the
# video/recording number and xx is the chapter number, e.g. 001_02_Game.mp4
# Generalized to: <video_number>_<chapter_number>_<anything>.<ext>
RENAMED_PATTERN = re.compile(r"^(\d+)_(\d+)_.+\.(mp4|mov)$", re.IGNORECASE)

# Timestamped camera files: "YYYYMMDDHHMMSS_NNNNN.MP4", e.g. 20260905112617_00001.MP4
# The camera splits long recordings into fixed-length chunks (e.g. every 25 min),
# each with its own start timestamp and a running file number. There's no shared
# recording ID, so chunks are grouped by checking whether each file starts right
# where the previous one ended (using ffprobe to get the previous file's length).
DATED_PATTERN = re.compile(r"^(\d{14})_(\d+)\.(mp4|mov)$", re.IGNORECASE)
# Some cameras split the timestamp up: "YYYY_MMDD_HHMMSS_NNN.MP4", e.g. 2026_0905_112938_001.MP4
DATED_SPLIT_PATTERN = re.compile(r"^(\d{4})_(\d{4})_(\d{6})_(\d+)\.(mp4|mov)$", re.IGNORECASE)
DATED_CHAIN_TOLERANCE = 10.0  # seconds of slack allowed between chunk end and next start
DATED_FALLBACK_MAX_GAP = 30 * 60  # if ffprobe fails: max seconds between chunk start times


def get_video_duration(path):
    """Returns video duration in seconds using ffprobe, or None if it fails."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return None


def group_dated_chunks(dated_files):
    """
    dated_files: list of (datetime, seq_num, filepath).
    Chains files into recordings when a file starts within
    DATED_CHAIN_TOLERANCE seconds of the previous file's end.
    Returns a list of (first file's start datetime, [filepaths]).
    """
    dated_files.sort(key=lambda t: (t[0], t[1]))
    recordings = []
    prev_end = None
    prev_start = None
    warned = False

    for start, _seq, path in dated_files:
        duration = get_video_duration(path)
        if duration is None and not warned:
            print("  WARNING: ffprobe couldn't read video lengths (is ffmpeg installed?).\n"
                  "           Falling back to grouping by timestamp spacing only.")
            warned = True

        if not recordings:
            same_recording = False
        elif prev_end is not None:
            # Preferred: this file starts right where the previous one ended
            same_recording = abs((start - prev_end).total_seconds()) <= DATED_CHAIN_TOLERANCE
        else:
            # Fallback: starts within 30 min of the previous file's start
            same_recording = (start - prev_start).total_seconds() <= DATED_FALLBACK_MAX_GAP

        if same_recording:
            recordings[-1][1].append(path)
        else:
            recordings.append((start, [path]))
        prev_start = start
        prev_end = start + timedelta(seconds=duration) if duration else None

    return recordings


def group_gopro_chapters(folder):
    """
    Groups video files in a folder into recordings so multi-chapter recordings
    get stitched back together. Returns a dict of {label: [ordered filepaths]}.
    Files that don't match a known pattern are returned individually under
    their own filename as the label.
    """
    files = sorted(os.listdir(folder))
    groups = {}  # media_id -> list of (chapter_num, filepath)
    dated = []   # (start_datetime, seq_num, filepath) for timestamped camera files
    unmatched = []

    for fname in files:
        if not fname.lower().endswith((".mp4", ".mov")):
            continue
        full = os.path.join(folder, fname)
        matched = False

        md = DATED_PATTERN.match(fname)
        ms = DATED_SPLIT_PATTERN.match(fname)
        if md or ms:
            stamp, seq = (md.group(1), md.group(2)) if md else ("".join(ms.group(1, 2, 3)), ms.group(4))
            try:
                start = datetime.strptime(stamp, "%Y%m%d%H%M%S")
                dated.append((start, int(seq), full))
                continue
            except ValueError:
                pass  # not a real timestamp, fall through to other patterns

        # Try the renamed "videoNum_chapterNum_..." pattern first, since
        # that's what PowerRenamer / manually renamed footage will use.
        m0 = RENAMED_PATTERN.match(fname)
        if m0:
            video_num, chapter_num = m0.group(1), int(m0.group(2))
            groups.setdefault(f"vid{video_num}", []).append((chapter_num, full))
            matched = True

        if not matched:
            m = GOPRO_PATTERNS[0].match(fname) or GOPRO_PATTERNS[1].match(fname)
            if m:
                chapter, media_id = int(m.group(1)), m.group(2)
                groups.setdefault(media_id, []).append((chapter, full))
                matched = True

        if not matched:
            m2 = GOPRO_PATTERNS[2].match(fname)
            if m2:
                media_id = m2.group(1)
                groups.setdefault(media_id, []).append((0, full))
                matched = True

        if not matched:
            unmatched.append(full)

    ordered_groups = {}
    for media_id, chapter_list in groups.items():
        chapter_list.sort(key=lambda t: t[0])
        label = media_id if media_id.startswith("vid") else f"GoPro_{media_id}"
        ordered_groups[label] = [f for _, f in chapter_list]

    for start, chunk_files in group_dated_chunks(dated):
        ordered_groups[f"rec_{start:%Y%m%d%H%M%S}"] = chunk_files

    for f in unmatched:
        base = os.path.splitext(os.path.basename(f))[0]
        ordered_groups[base] = [f]

    return ordered_groups


def stitch_chapters(filepaths, work_dir):
    """
    If filepaths has more than one entry, losslessly concatenates them into
    a single file in work_dir using ffmpeg's concat demuxer. Returns the
    path to use for analysis (either the single original file, or the
    newly stitched one).
    """
    if len(filepaths) == 1:
        return filepaths[0]

    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "_chapters_concat_list.txt")
    with open(list_path, "w") as f:
        for p in filepaths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    out_name = os.path.splitext(os.path.basename(filepaths[0]))[0] + "_stitched.mp4"
    out_path = os.path.join(work_dir, out_name)

    print(f"  Stitching {len(filepaths)} chapters -> {out_name}")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("    stream copy failed, re-encoding to stitch instead...")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"ERROR stitching chapters for {filepaths[0]}:\n{result.stderr[-800:]}")

    os.remove(list_path)
    return out_path


# ---------------------------------------------------------------------------
# Jersey color matching
# ---------------------------------------------------------------------------

class ColorMatcher:
    """
    Matches pixels against a list of jersey/pad colors given as hex strings.
    Works in HSV so lighting changes (brightness) matter less than hue.

    Very light, low-saturation colors (white, light gray) are skipped because
    the ice itself is white and would match everywhere.
    """

    def __init__(self, hex_colors, hue_tolerance=12):
        self.ranges = []  # list of (lower_hsv, upper_hsv) numpy arrays
        self.names = []
        for hx in hex_colors:
            h, s, v = self._hex_to_hsv(hx)
            if s < 60:
                if v >= 170:
                    print(f"  NOTE: jersey color {hx} is white/very light; skipping it "
                          f"(the ice is white, so it would match everything).")
                    continue
                if v <= 70:
                    print(f"  NOTE: jersey color {hx} is black; it will also match shadows "
                          f"and dark gear, so treat it as a weak signal.")
                    self.ranges.append((np.array([0, 0, 0]), np.array([179, 255, 60])))
                    self.names.append(hx)
                    continue
                print(f"  NOTE: jersey color {hx} is gray; skipping it (too close to ice/boards).")
                continue

            min_s = max(60, int(s * 0.45))
            min_v = max(40, int(v * 0.35))
            lo_h, hi_h = h - hue_tolerance, h + hue_tolerance
            if lo_h < 0:  # hue wraps around (reds)
                self.ranges.append((np.array([0, min_s, min_v]), np.array([hi_h, 255, 255])))
                self.ranges.append((np.array([180 + lo_h, min_s, min_v]), np.array([179, 255, 255])))
            elif hi_h > 179:
                self.ranges.append((np.array([lo_h, min_s, min_v]), np.array([179, 255, 255])))
                self.ranges.append((np.array([0, min_s, min_v]), np.array([hi_h - 180, 255, 255])))
            else:
                self.ranges.append((np.array([lo_h, min_s, min_v]), np.array([hi_h, 255, 255])))
            self.names.append(hx)

        if not self.ranges:
            print("  WARNING: no usable jersey colors; color filtering is turned off.")

    @staticmethod
    def _hex_to_hsv(hx):
        hx = hx.strip().lstrip("#")
        if len(hx) != 6:
            sys.exit(f"ERROR: jersey color '{hx}' must be a 6-digit hex code like \"#1a3c8f\"")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        px = np.uint8([[[b, g, r]]])
        h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        return int(h), int(s), int(v)

    @property
    def usable(self):
        return bool(self.ranges)

    def mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.ranges:
            combined |= cv2.inRange(hsv, lo, hi)
        return combined


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------

def parse_roi(roi):
    """
    One or more boxes to look for motion in, each "x,y,w,h" in pixels with
    0,0 at the top-left of the frame. Several boxes are separated by ";"
    ("0,640,640,800; 1840,640,720,800"), or given as a list in the config.
    Returns a list of (x, y, w, h), or None for the whole frame.
    """
    if not roi:
        return None
    parts = roi if isinstance(roi, list) else str(roi).split(";")
    boxes = []
    for part in (str(p).strip() for p in parts):
        if not part:
            continue
        try:
            x, y, w, h = (int(v) for v in part.split(","))
        except ValueError:
            sys.exit(f"ERROR: roi '{part}' must be in the form x,y,w,h e.g. 0,0,960,1080")
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            sys.exit(f"ERROR: roi '{part}': x and y must be 0 or more, and w and h must be more than 0")
        boxes.append((x, y, w, h))
    return boxes or None


def check_roi_fits(boxes, frame_width, frame_height):
    """Exits with a clear error if a box doesn't lie fully inside the frame."""
    for x, y, w, h in boxes:
        if x + w > frame_width or y + h > frame_height:
            sys.exit(
                f"ERROR: roi {x},{y},{w},{h} goes outside the video frame, "
                f"which is {frame_width}x{frame_height} pixels.\n"
                "  x,y is the box's top-left corner, measured from the top-left of the frame.\n"
                f"  x + w must be at most {frame_width}, and y + h must be at most {frame_height}."
            )


def roi_area(boxes):
    """The bounding box of all the boxes, and a mask of the boxes inside it
    (None when there's just one box, which needs no mask)."""
    x0 = min(x for x, _, _, _ in boxes)
    y0 = min(y for _, y, _, _ in boxes)
    x1 = max(x + w for x, _, w, _ in boxes)
    y1 = max(y + h for _, y, _, h in boxes)
    if len(boxes) == 1:
        return (x0, y0, x1 - x0, y1 - y0), None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for x, y, w, h in boxes:
        mask[y - y0:y - y0 + h, x - x0:x - x0 + w] = 255
    return (x0, y0, x1 - x0, y1 - y0), mask


def open_video(video_path):
    if not os.path.isfile(video_path):
        sys.exit(
            f"ERROR: file not found: {video_path}\n"
            "  (In WSL, Windows drives are usually lowercase, e.g. /mnt/d/ not /mnt/D/)"
        )

    # Force the FFmpeg backend. Without this, if FFmpeg can't open the file
    # OpenCV falls back to its image-sequence reader, which chokes on the
    # digits in GoPro-style filenames and hides the real error.
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(
            f"ERROR: could not open video file: {video_path}\n"
            "  The file exists, so OpenCV's FFmpeg support is likely the problem.\n"
            "  Check with: python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep -A2 FFMPEG"
        )
    return cap


def detect_motion_segments(video_path, threshold, sample_rate, roi, min_duration,
                           start=0.0, end=None, color_matcher=None,
                           min_color_pct=0.0, color_filter=False):
    """
    Analyzes video_path between start and end (seconds).
    Returns (segments, duration, stats):
      segments: list of (start_seconds, end_seconds) of raw active segments
      duration: full video duration in seconds
      stats:    dict with color match info (empty when no color matcher)
    """
    cap = open_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    if end is None or end > duration:
        end = duration

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.2f}  Frames: {frame_count}  Duration: {duration/60:.1f} min")
    if start > 0 or end < duration:
        print(f"  Analyzing {fmt_time(start)} - {fmt_time(end)}")
    print(f"  Analyzing every {sample_rate} frame(s) for motion (threshold={threshold})...")

    start_frame = int(start * fps)
    end_frame = int(end * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    use_color = color_matcher is not None and color_matcher.usable
    area = mask = None
    prev_gray = None
    samples = []  # list of (timestamp_seconds, is_active, color_pct_or_None)
    frame_idx = start_frame
    span = max(1, end_frame - start_frame)

    while frame_idx < end_frame:
        if frame_idx % sample_rate != 0:
            # grab() skips decoding, much faster for frames we don't analyze
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        if roi:
            if area is None:
                # Check against the decoded frame rather than the file's metadata,
                # since OpenCV auto-rotates footage and can swap width and height.
                check_roi_fits(roi, frame.shape[1], frame.shape[0])
                area, mask = roi_area(roi)
            x, y, w, h = area
            frame_region = frame[y:y + h, x:x + w]
        else:
            frame_region = frame

        gray = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        timestamp = frame_idx / fps

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh_img = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            if mask is not None:
                # several boxes: only count motion inside them
                thresh_img &= mask
                motion_score = cv2.countNonZero(thresh_img) / cv2.countNonZero(mask) * 1000
            else:
                motion_score = (thresh_img.sum() / 255) / thresh_img.size * 1000
            is_active = motion_score > threshold

            color_pct = None
            if use_color and is_active:
                # Of the pixels that are MOVING, what % match his colors?
                # Only looking at moving pixels ignores static stuff like the
                # red/blue lines, boards, and ads.
                scale = min(1.0, 480 / frame_region.shape[1])
                small = cv2.resize(frame_region, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                motion_small = cv2.resize(thresh_img, (small.shape[1], small.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                moving = cv2.countNonZero(motion_small)
                if moving > 0:
                    matched = cv2.countNonZero(color_matcher.mask(small) & motion_small)
                    color_pct = matched / moving * 100

            samples.append((timestamp, is_active, color_pct))

        prev_gray = gray

        if (frame_idx - start_frame) % (sample_rate * 500) == 0:
            pct = (frame_idx - start_frame) / span * 100
            print(f"    ...{pct:.0f}% analyzed", end="\r")

        frame_idx += 1

    cap.release()
    print("    ...analysis complete       ")

    # Convert samples into raw segments, remembering each segment's color readings
    raw = []  # (start, end, [color_pct...])
    seg_start = None
    seg_colors = []
    last_ts = start
    for ts, is_active, color_pct in samples:
        last_ts = ts
        if is_active:
            if seg_start is None:
                seg_start = ts
                seg_colors = []
            if color_pct is not None:
                seg_colors.append(color_pct)
        elif seg_start is not None:
            raw.append((seg_start, ts, seg_colors))
            seg_start = None
    if seg_start is not None:
        raw.append((seg_start, last_ts, seg_colors))

    # Drop segments that are too short (likely noise)
    raw = [r for r in raw if (r[1] - r[0]) >= min_duration]

    stats = {}
    if use_color:
        all_colors = [c for _, _, cs in raw for c in cs]
        stats["avg_color_pct"] = sum(all_colors) / len(all_colors) if all_colors else 0.0
        if color_filter:
            kept = []
            for s, e, cs in raw:
                avg = sum(cs) / len(cs) if cs else 0.0
                if avg >= min_color_pct:
                    kept.append((s, e, cs))
            stats["dropped_by_color"] = len(raw) - len(kept)
            raw = kept

    segments = [(s, e) for s, e, _ in raw]
    return segments, duration, stats


def merge_close_segments(segments, min_gap):
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def apply_padding(segments, pad_before, pad_after, lower, upper):
    """Pads each segment, clamped to [lower, upper]."""
    padded = []
    for start, end in segments:
        s = max(lower, start - pad_before)
        e = min(upper, end + pad_after)
        padded.append((s, e))
    # Re-merge in case padding caused overlaps
    return merge_close_segments(padded, 0)


def fmt_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def parse_offset(value, field):
    """Parses "H:MM:SS", "MM:SS", or a number of seconds into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parts = [float(p) for p in str(value).strip().split(":")]
    except ValueError:
        sys.exit(f"ERROR: {field} = \"{value}\" isn't a valid time. Use \"H:MM:SS\" or \"MM:SS\".")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


# ---------------------------------------------------------------------------
# Cutting / merging
# ---------------------------------------------------------------------------

def cut_clips(video_path, segments, out_dir, prefix=None):
    os.makedirs(out_dir, exist_ok=True)
    base = prefix or os.path.splitext(os.path.basename(video_path))[0]
    clip_paths = []

    for i, (start, end) in enumerate(segments, 1):
        out_path = os.path.join(out_dir, f"{base}_clip{i:02d}.mp4")
        duration = end - start
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        print(f"  Cutting clip {i}: {fmt_time(start)} - {fmt_time(end)}  ({duration:.1f}s)")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fall back to re-encoding if stream copy fails (common with GoPro
            # footage when the cut doesn't land on a keyframe)
            print("    stream copy failed, re-encoding this clip instead...")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    ERROR cutting clip {i}:\n{result.stderr[-800:]}")
                continue

        clip_paths.append(out_path)

    return clip_paths


def merge_clips(clip_paths, out_dir, base_name):
    if not clip_paths:
        print("No clips to merge.")
        return

    list_path = os.path.join(out_dir, "_concat_list.txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    merged_path = os.path.join(out_dir, f"{base_name}_highlights_merged.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        merged_path,
    ]
    print(f"Merging {len(clip_paths)} clips into {merged_path} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  stream copy merge failed, re-encoding instead...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac",
            merged_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR merging clips:\n{result.stderr[-800:]}")
            return

    os.remove(list_path)
    print(f"Done: {merged_path}")


def process_one_video(video_path, args, out_dir, tag=""):
    """Runs detection + cutting for a single (already-stitched) video file."""
    roi = parse_roi(args.roi)

    raw_segments, duration, _ = detect_motion_segments(
        video_path, args.threshold, args.sample_rate, roi, args.min_duration
    )
    merged_segments = merge_close_segments(raw_segments, args.min_gap)
    final_segments = apply_padding(merged_segments, args.pad_before, args.pad_after, 0, duration)

    total_active = sum(e - s for s, e in final_segments)
    print(f"\n{tag}Found {len(final_segments)} active segment(s), "
          f"totaling {total_active/60:.1f} min out of {duration/60:.1f} min "
          f"({(total_active/duration*100 if duration else 0):.0f}% kept)\n")

    for i, (s, e) in enumerate(final_segments, 1):
        print(f"  Segment {i}: {fmt_time(s)} - {fmt_time(e)}")

    if args.dry_run:
        print(f"{tag}Dry run only, no clips were cut.")
        return []

    if not final_segments:
        print(f"{tag}No active segments found — try lowering --threshold.")
        return []

    clip_paths = cut_clips(video_path, final_segments, out_dir)

    if args.merge:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        merge_clips(clip_paths, out_dir, base_name)

    print(f"{tag}Done. {len(clip_paths)} clip(s) saved to ./{out_dir}/")
    return clip_paths


# ---------------------------------------------------------------------------
# Config mode
# ---------------------------------------------------------------------------

SETTING_KEYS = {
    "threshold": float, "min_gap": float, "min_duration": float,
    "pad_before": float, "pad_after": float, "sample_rate": int,
    "roi": lambda v: "; ".join(v) if isinstance(v, list) else str(v), "out_dir": str,
}


def load_config(path):
    try:
        import tomllib
    except ImportError:
        sys.exit("ERROR: config mode needs Python 3.11+ (for tomllib). Check with: python3 --version")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: config file not found: {path}")
    with open(path, "rb") as f:
        try:
            cfg = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            sys.exit(f"ERROR: couldn't read {path}: {e}")

    if not cfg.get("cameras"):
        sys.exit("ERROR: config needs at least one [cameras.NAME] section with a folder.")
    for name, cam in cfg["cameras"].items():
        if "folder" not in cam:
            sys.exit(f"ERROR: [cameras.{name}] is missing folder = \"...\"")
    for g in cfg.get("games", []):
        if "name" not in g:
            sys.exit("ERROR: every [[games]] entry needs a name.")
        for p in g.get("periods", []):
            for key in ("period", "camera", "start"):
                if key not in p:
                    sys.exit(f"ERROR: game '{g['name']}': each period needs '{key}'.")
            if p["camera"] not in cfg["cameras"]:
                sys.exit(f"ERROR: game '{g['name']}' period {p['period']}: "
                         f"camera '{p['camera']}' isn't defined under [cameras].")
            if p["camera"] not in g.get("recordings", {}):
                sys.exit(f"ERROR: game '{g['name']}': no recording listed for camera "
                         f"'{p['camera']}' under [games.recordings].")
    return cfg


def list_recordings(cfg, camera_groups):
    for cam, groups in camera_groups.items():
        print(f"\nCamera '{cam}'  ({cfg['cameras'][cam]['folder']})")
        if not groups:
            print("  (no video files found)")
        for label, files in groups.items():
            total = sum((get_video_duration(f) or 0) for f in files)
            print(f"  {label}   {len(files)} file(s), {total/60:.1f} min   "
                  f"[{os.path.basename(files[0])} ... {os.path.basename(files[-1])}]")
    print("\nCopy the labels above into [games.recordings] in your config.")


def run_config_mode(cfg, args):
    player = cfg.get("player", {})
    colors = player.get("jersey_colors", [])
    color_filter = bool(player.get("color_filter", True))
    min_color_pct = float(player.get("min_color_pct", 5.0))
    hue_tol = int(player.get("hue_tolerance", 12))

    if player:
        who = player.get("name", "goalie")
        num = player.get("number")
        print(f"Player: {who}" + (f"  #{num}" if num is not None else "")
              + (f"  colors: {', '.join(colors)}" if colors else ""))

    matcher = ColorMatcher(colors, hue_tol) if colors else None

    # Group each camera's folder into recordings once
    camera_groups = {}
    for cam, cam_cfg in cfg["cameras"].items():
        folder = cam_cfg["folder"]
        if not os.path.isdir(folder):
            sys.exit(f"ERROR: camera '{cam}' folder not found: {folder}\n"
                     "  (Is the drive mounted? In WSL try: sudo mount -t drvfs D: /mnt/d)")
        camera_groups[cam] = group_gopro_chapters(folder)

    if args.list:
        list_recordings(cfg, camera_groups)
        return

    games = cfg.get("games", [])
    if args.game:
        games = [g for g in games if g["name"] == args.game]
        if not games:
            sys.exit(f"ERROR: no game named '{args.game}' in the config.")
    if not games:
        sys.exit("ERROR: no [[games]] in the config. Run with --list to see recordings, then add a game.")

    for game in games:
        name = game["name"]
        game_dir = os.path.join(args.out_dir, name)
        stitch_dir = os.path.join(args.out_dir, "_stitched")
        stitched = {}  # (camera, label) -> video path
        game_clips = []
        summary = []

        print(f"\n{'=' * 60}\nGame: {name}\n{'=' * 60}")

        for p in sorted(game.get("periods", []), key=lambda p: p["period"]):
            pnum, cam = p["period"], p["camera"]
            label = game["recordings"][cam]
            groups = camera_groups[cam]
            if label not in groups:
                sys.exit(f"ERROR: game '{name}': recording '{label}' not found for camera '{cam}'.\n"
                         f"  Available: {', '.join(groups) or '(none)'}\n"
                         "  Run with --list to see them.")

            key = (cam, label)
            if key not in stitched:
                stitched[key] = stitch_chapters(groups[label], os.path.join(stitch_dir, cam))
            video_path = stitched[key]

            start = parse_offset(p["start"], f"period {pnum} start")
            end = parse_offset(p["end"], f"period {pnum} end") if "end" in p else None
            if end is not None and end <= start:
                sys.exit(f"ERROR: game '{name}' period {pnum}: end must be after start.")

            # a period's own roi (e.g. a camera that was set up differently at
            # that game) overrides the camera's, which overrides --roi
            roi = parse_roi(p.get("roi") or cfg["cameras"][cam].get("roi") or args.roi)

            print(f"\n--- Period {pnum}  (camera '{cam}', {label}) ---")
            raw, duration, stats = detect_motion_segments(
                video_path, args.threshold, args.sample_rate, roi, args.min_duration,
                start=start, end=end, color_matcher=matcher,
                min_color_pct=min_color_pct, color_filter=color_filter,
            )
            p_end = min(end, duration) if end is not None else duration
            segs = merge_close_segments(raw, args.min_gap)
            segs = apply_padding(segs, args.pad_before, args.pad_after, start, p_end)

            total_active = sum(e - s for s, e in segs)
            p_len = max(1e-6, p_end - start)
            print(f"\n[P{pnum}] {len(segs)} segment(s), {total_active/60:.1f} of "
                  f"{p_len/60:.1f} min ({total_active/p_len*100:.0f}% kept)")

            if "avg_color_pct" in stats:
                avg = stats["avg_color_pct"]
                note = f"  jersey color in moving pixels: {avg:.1f}% (min_color_pct = {min_color_pct})"
                if color_filter:
                    note += f", {stats.get('dropped_by_color', 0)} segment(s) dropped by color"
                print(note)
                if avg < min_color_pct:
                    print(f"  WARNING: his colors barely show up in period {pnum} on camera '{cam}'.\n"
                          "           Double-check the camera for this period, the ROI, or min_color_pct.")

            for i, (s, e) in enumerate(segs, 1):
                print(f"  Segment {i}: {fmt_time(s)} - {fmt_time(e)}")

            summary.append((pnum, cam, len(segs), total_active))

            if not args.dry_run and segs:
                clips = cut_clips(video_path, segs, game_dir, prefix=f"P{pnum}_{cam}")
                game_clips.extend(clips)

        print(f"\nGame '{name}' summary:")
        for pnum, cam, n, t in summary:
            print(f"  P{pnum}  camera {cam:<10} {n:>3} segment(s)  {t/60:5.1f} min")

        if args.dry_run:
            print("Dry run only, no clips were cut.")
        elif game_clips:
            merge_clips(game_clips, game_dir, f"{name}_full_game")
        else:
            print("No clips found for this game — try lowering threshold or min_color_pct.")

        if not args.keep_stitched:
            for (cam, label), path in stitched.items():
                if path.startswith(stitch_dir) and os.path.exists(path):
                    os.remove(path)
            for root, dirs, _ in os.walk(stitch_dir, topdown=False):
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except OSError:
                        pass
            try:
                os.rmdir(stitch_dir)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract goalie action clips from raw game footage.")
    parser.add_argument("video", nargs="?", help="Path to a single video file, OR a folder of footage (use with --batch). Not needed with --config.")
    parser.add_argument("--config", type=str, default=None, help="TOML config with player colors, cameras, and per-period camera/offsets")
    parser.add_argument("--list", action="store_true", help="Config mode: list each camera's recordings (labels for the config) and exit")
    parser.add_argument("--game", type=str, default=None, help="Config mode: only process the game with this name")
    parser.add_argument("--batch", action="store_true", help="Treat 'video' as a folder; auto-groups chapter files and processes each recording")
    parser.add_argument("--threshold", type=float, default=15.0, help="Motion sensitivity threshold (default 15)")
    parser.add_argument("--min-gap", type=float, default=3.0, help="Merge segments separated by less than this many seconds (default 3.0)")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Discard segments shorter than this (default 1.5s)")
    parser.add_argument("--pad-before", type=float, default=5.0, help="Seconds of padding before each segment (default 5.0)")
    parser.add_argument("--pad-after", type=float, default=5.0, help="Seconds of padding after each segment (default 5.0)")
    parser.add_argument("--sample-rate", type=int, default=5, help="Analyze every Nth frame (default 5)")
    parser.add_argument("--roi", type=str, default=None, help="Only detect motion inside this box: x,y,w,h in pixels, with 0,0 at the top-left of the frame. Several boxes: separate them with ;")
    parser.add_argument("--merge", action="store_true", help="Also produce one merged highlight video (per recording in batch mode)")
    parser.add_argument("--merge-all", action="store_true", help="Batch mode only: merge every recording's clips into ONE single highlight video")
    parser.add_argument("--dry-run", action="store_true", help="Only print detected segments, don't cut clips")
    parser.add_argument("--out-dir", type=str, default="highlights", help="Output directory (default: ./highlights)")
    parser.add_argument("--keep-stitched", action="store_true", help="Keep the intermediate stitched chapter files instead of deleting them")

    # Load config first so its [settings] become the defaults; CLI flags still win.
    pre_args, _ = parser.parse_known_args()
    cfg = None
    if pre_args.config:
        cfg = load_config(pre_args.config)
        settings = {}
        for k, v in cfg.get("settings", {}).items():
            if k not in SETTING_KEYS:
                print(f"  NOTE: ignoring unknown setting '{k}' in [settings]")
                continue
            settings[k] = SETTING_KEYS[k](v)
        parser.set_defaults(**settings)

    args = parser.parse_args()

    if cfg is not None:
        run_config_mode(cfg, args)
        return

    if not args.video:
        parser.error("give a video file/folder, or use --config")

    if not args.batch:
        process_one_video(args.video, args, args.out_dir)
        return

    # ---- Batch/folder mode ----
    if not os.path.isdir(args.video):
        sys.exit(f"ERROR: --batch was given but '{args.video}' is not a folder.")

    groups = group_gopro_chapters(args.video)
    if not groups:
        sys.exit(f"ERROR: no .mp4/.mov files found in {args.video}")

    print(f"Found {len(groups)} recording(s) in folder:")
    for name, files in groups.items():
        chapter_note = f" ({len(files)} chapters)" if len(files) > 1 else ""
        print(f"  {name}{chapter_note}: {[os.path.basename(f) for f in files]}")

    stitch_dir = os.path.join(args.out_dir, "_stitched")
    all_clip_paths = []

    for name, files in groups.items():
        print(f"\n=== Processing {name} ===")
        video_path = stitch_chapters(files, stitch_dir)
        clip_paths = process_one_video(video_path, args, args.out_dir, tag=f"[{name}] ")
        all_clip_paths.extend(clip_paths)

        if len(files) > 1 and not args.keep_stitched and video_path.startswith(stitch_dir):
            os.remove(video_path)

    if os.path.isdir(stitch_dir) and not os.listdir(stitch_dir):
        os.rmdir(stitch_dir)

    if args.merge_all and all_clip_paths:
        print(f"\n=== Merging all {len(all_clip_paths)} clips from every recording into one game highlight video ===")
        merge_clips(all_clip_paths, args.out_dir, "game")

    print(f"\nBatch complete. Processed {len(groups)} recording(s).")


if __name__ == "__main__":
    main()
