#!/usr/bin/env python3
"""
Goalie Highlight Extractor
---------------------------
Scans a hockey video (e.g. GoPro footage focused on the goalie) for motion
activity, and cuts out the "standing around" segments, keeping only the
active play. Outputs individual clip files and (optionally) one merged
highlight video.

HOW IT WORKS
------------
1. Reads the video frame-by-frame (at a reduced sample rate for speed).
2. Computes frame-to-frame difference to estimate motion level.
3. Marks frames as "active" when motion exceeds a threshold.
4. Groups active frames into segments, merging ones close together and
   padding the start/end so plays aren't cut off abruptly.
5. Uses ffmpeg to losslessly (or re-encoded) extract those segments.

REQUIREMENTS
------------
    pip install opencv-python --break-system-packages
    ffmpeg must be installed and on your PATH (https://ffmpeg.org/download.html)

USAGE
-----
    python3 goalie_highlight_extractor.py input.mp4

    Common options:
        --threshold 15        Motion sensitivity (lower = more sensitive, more clips)
        --min-gap 3.0         Merge active segments separated by less than this many seconds
        --min-duration 1.5    Discard active segments shorter than this (avoid single-frame noise)
        --pad-before 5.0      Seconds of padding added before each segment
        --pad-after 5.0       Seconds of padding added after each segment
        --sample-rate 5       Analyze every Nth frame (higher = faster, less precise)
        --merge               Also produce one merged highlight video (highlights_merged.mp4)
        --roi x,y,w,h         Only look for motion in this region of the frame
                               (e.g. crop to the goalie's crease area to ignore
                               crowd/bench movement elsewhere in frame)
        --dry-run             Just print detected segments, don't cut anything

EXAMPLE
-------
    # First do a dry run to see how many clips it finds and tune threshold
    python3 goalie_highlight_extractor.py period1.mp4 --dry-run

    # Once happy, cut clips and produce a merged highlight reel
    python3 goalie_highlight_extractor.py period1.mp4 --merge

    # If the goalie is only in the left half of frame, restrict detection
    # to that region to cut down on false positives from crowd/bench:
    python3 goalie_highlight_extractor.py period1.mp4 --roi 0,0,960,1080 --merge
"""

import argparse
import glob
import os
import re
import subprocess
import sys

try:
    import cv2
except ImportError:
    sys.exit(
        "ERROR: opencv-python is not installed.\n"
        "Install it with:\n"
        "    pip install opencv-python --break-system-packages\n"
    )


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
    Returns a list of lists of filepaths.
    """
    from datetime import timedelta

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
            recordings[-1].append(path)
        else:
            recordings.append([path])
        prev_start = start
        prev_end = start + timedelta(seconds=duration) if duration else None

    return recordings


def group_gopro_chapters(folder):
    """
    Groups files in a folder by GoPro media ID so multi-chapter recordings
    get stitched back together. Returns a dict of {media_id: [ordered filepaths]}.
    Files that don't match a known GoPro pattern are returned individually
    under their own filename as the key.
    """
    from datetime import datetime

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
        if md:
            try:
                start = datetime.strptime(md.group(1), "%Y%m%d%H%M%S")
                dated.append((start, int(md.group(2)), full))
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

    for chunk_files in group_dated_chunks(dated):
        first = os.path.splitext(os.path.basename(chunk_files[0]))[0]
        ordered_groups[f"rec_{first.split('_')[0]}"] = chunk_files

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


def parse_roi(roi_str):
    if not roi_str:
        return None
    try:
        x, y, w, h = (int(v) for v in roi_str.split(","))
        return (x, y, w, h)
    except Exception:
        sys.exit("ERROR: --roi must be in the form x,y,w,h e.g. 0,0,960,1080")


def detect_motion_segments(video_path, threshold, sample_rate, roi, min_duration):
    """Returns a list of (start_seconds, end_seconds) of raw active segments."""
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.2f}  Frames: {frame_count}  Duration: {duration/60:.1f} min")
    print(f"  Analyzing every {sample_rate} frame(s) for motion (threshold={threshold})...")

    prev_gray = None
    active_flags = []  # list of (timestamp_seconds, is_active)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            if roi:
                x, y, w, h = roi
                frame_region = frame[y:y + h, x:x + w]
            else:
                frame_region = frame

            gray = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            timestamp = frame_idx / fps

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh_img = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_score = (thresh_img.sum() / 255) / thresh_img.size * 1000
                is_active = motion_score > threshold
                active_flags.append((timestamp, is_active))

            prev_gray = gray

            if frame_idx % (sample_rate * 500) == 0:
                pct = (frame_idx / frame_count) * 100 if frame_count else 0
                print(f"    ...{pct:.0f}% analyzed", end="\r")

        frame_idx += 1

    cap.release()
    print("    ...analysis complete       ")

    # Convert flags into raw segments
    segments = []
    seg_start = None
    last_ts = 0
    for ts, is_active in active_flags:
        last_ts = ts
        if is_active and seg_start is None:
            seg_start = ts
        elif not is_active and seg_start is not None:
            segments.append((seg_start, ts))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, last_ts))

    # Drop segments that are too short (likely noise)
    segments = [(s, e) for s, e in segments if (e - s) >= min_duration]

    return segments, duration


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


def apply_padding(segments, pad_before, pad_after, duration):
    padded = []
    for start, end in segments:
        s = max(0, start - pad_before)
        e = min(duration, end + pad_after)
        padded.append((s, e))
    # Re-merge in case padding caused overlaps
    return merge_close_segments(padded, 0)


def fmt_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def cut_clips(video_path, segments, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
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

    raw_segments, duration = detect_motion_segments(
        video_path, args.threshold, args.sample_rate, roi, args.min_duration
    )
    merged_segments = merge_close_segments(raw_segments, args.min_gap)
    final_segments = apply_padding(merged_segments, args.pad_before, args.pad_after, duration)

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


def main():
    parser = argparse.ArgumentParser(description="Extract goalie action clips from raw game footage.")
    parser.add_argument("video", help="Path to a single video file, OR a folder of GoPro footage (use with --batch)")
    parser.add_argument("--batch", action="store_true", help="Treat 'video' as a folder; auto-groups GoPro chapter files (GH01xxxx/GH02xxxx etc.) and processes each recording")
    parser.add_argument("--threshold", type=float, default=15.0, help="Motion sensitivity threshold (default 15)")
    parser.add_argument("--min-gap", type=float, default=3.0, help="Merge segments separated by less than this many seconds (default 3.0)")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Discard segments shorter than this (default 1.5s)")
    parser.add_argument("--pad-before", type=float, default=5.0, help="Seconds of padding before each segment (default 5.0)")
    parser.add_argument("--pad-after", type=float, default=5.0, help="Seconds of padding after each segment (default 5.0)")
    parser.add_argument("--sample-rate", type=int, default=5, help="Analyze every Nth frame (default 5)")
    parser.add_argument("--roi", type=str, default=None, help="Region of interest x,y,w,h to restrict motion detection")
    parser.add_argument("--merge", action="store_true", help="Also produce one merged highlight video (per recording in batch mode)")
    parser.add_argument("--merge-all", action="store_true", help="Batch mode only: merge every recording's clips into ONE single highlight video for the whole game")
    parser.add_argument("--dry-run", action="store_true", help="Only print detected segments, don't cut clips")
    parser.add_argument("--out-dir", type=str, default="highlights", help="Output directory (default: ./highlights)")
    parser.add_argument("--keep-stitched", action="store_true", help="Batch mode only: keep the intermediate stitched chapter files instead of deleting them after processing")

    args = parser.parse_args()

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
