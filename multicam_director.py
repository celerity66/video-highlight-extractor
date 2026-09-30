#!/usr/bin/env python3
"""
Multi-camera game director
--------------------------
Turns footage from fixed cameras around a rink into one video per game that always
shows the camera closest to the action, plus an action-only version with warm-up,
intermissions and stoppages cut out (using the scoreboard clock).

    # 1. find the games (reads every video once: slow the first time, then cached)
    python3 multicam_director.py scan "/path/to/VIDEO"

    # 2. check game_videos/games.csv: name the games, set render to yes/no

    # 3. make the videos
    python3 multicam_director.py render "/path/to/VIDEO"

See MULTICAM.md for details.
"""
import argparse
import json
import os
import sys
import time

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
except ImportError:
    sys.exit("ERROR: opencv-python is not installed. Install it with: pip install opencv-python")

from multicam import footage, games, motion, pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rink(rink):
    rink_dir = rink if os.path.isdir(rink) else os.path.join(HERE, "rinks", rink)
    path = os.path.join(rink_dir, "rink.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no rink settings at {path}")
    return json.load(open(path)), rink_dir


def scan_footage(args, config):
    if not os.path.isdir(args.footage):
        sys.exit(f"ERROR: footage folder not found: {args.footage}\n"
                 "  (In WSL, Windows drives are under /mnt/ with a lowercase letter, e.g. /mnt/d/)")
    cache = os.path.join(args.out, ".cache")
    os.makedirs(cache, exist_ok=True)
    print("Looking for videos...")
    recs = footage.scan(args.footage, list(config["cameras"]), cache)
    if not any(recs.values()):
        sys.exit(f"ERROR: no camera folders ({', '.join(config['cameras'])}) with videos in {args.footage}")
    return recs, footage.trusted_clock(recs), cache


def cmd_scan(args):
    config, _ = load_rink(args.rink)
    recs, trusted, cache = scan_footage(args, config)
    usable = {c: [r for r in rs if r.duration >= games.MIN_RECORDING_MIN * 60] for c, rs in recs.items()}
    n = sum(len(r.files) for rs in usable.values() for r in rs)
    print(f"Found {sum(map(len, usable.values()))} recordings ({n} files). Reading motion from every file "
          "(cached, so only new files take time)...")
    t0 = time.time()
    motion.extract_all(usable, cache, jobs=args.jobs,
                       progress=lambda m: print(f"{(time.time() - t0) / 60:6.1f} min {m}", flush=True))
    print("Matching recordings to games...")
    found = games.build_games(recs, trusted, config, cache)
    path = os.path.join(args.out, "games.csv")
    if os.path.exists(path) and not args.force:
        path = os.path.join(args.out, "games_new.csv")
        print(f"games.csv already exists, so the new list is in {path}")
    games.write_csv(found, path, args.rink)
    print(f"\n{len(found)} game(s) written to {path}")
    print("Check it (name the games, set render to no to skip one), then run the render command.")


def cmd_render(args):
    config, _ = load_rink(args.rink)
    recs, trusted, cache = scan_footage(args, config)
    path = args.games or os.path.join(args.out, "games.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no games list at {path}. Run the scan command first.")
    todo = [g for g in games.read_csv(path, recs) if g["render"] in ("yes", "y", "1", "true")]
    if args.only:
        todo = [g for g in todo if args.only in (g["game"], g["name"])]
    if not todo:
        sys.exit("Nothing to render (check the render column in games.csv, or --only).")
    for g in todo:
        stem = games.output_name(g)
        done_name = f"{stem}_multicam_action.mp4" if args.action_only else f"{stem}_multicam.mp4"
        if os.path.exists(os.path.join(args.out, done_name)) and not (args.force or args.check_only):
            print(f"Game {g['game']} ({stem}): already rendered, skipping (use --force to redo)")
            continue
        print(f"\n=== Game {g['game']}: {stem} (cameras {' '.join(g['recordings'])}) ===")
        motion.extract_all({c: [r] for c, r in g["recordings"].items()}, cache, jobs=args.jobs, progress=lambda m: None)
        game_config, rink_dir = load_rink(g["rink"] or args.rink)
        report = pipeline.Game(g, game_config, rink_dir, cache, trusted, force=args.force,
                               check_only=args.check_only, action_only=args.action_only).run(args.out, stem)
        for note in report["notes"]:
            print(f"  NOTE: {note}")
        if args.check_only:
            for prob in report["problems"]:
                print(f"  PROBLEM: {prob}")
            print(f"  Sync check saved: {stem}_sync_check.jpg and {stem}_sync_check.mp4 (nothing rendered)")
        elif not report["rendered"]:
            print("  NOT RENDERED:")
            for prob in report["problems"]:
                print(f"    - {prob}")
            print(f"  See {stem}_sync_check.jpg. Fix the rink settings or games.csv, or use --force to render anyway.")
        else:
            for prob in report["problems"]:
                print(f"  PROBLEM (rendered anyway because of --force): {prob}")
            print(f"  done in {report['minutes_taken']} min")


def main():
    p = argparse.ArgumentParser(description="Make one video per game from several rink cameras.")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, text in (("scan", cmd_scan, "find the games and write games.csv"),
                           ("render", cmd_render, "make the videos for the games in games.csv")):
        s = sub.add_parser(name, help=text)
        s.add_argument("footage", help="folder containing one sub-folder per camera (named 5, 6, 7...)")
        s.add_argument("--rink", default="barnburner", help="rink settings: a name in rinks/ or a folder (default barnburner)")
        s.add_argument("--out", default="game_videos", help="where videos and games.csv go (default game_videos)")
        s.add_argument("--jobs", type=int, default=6, help="videos read at once (default 6)")
        s.add_argument("--force", action="store_true", help="overwrite games.csv; re-render existing videos, even if problems are found")
        s.set_defaults(fn=fn)
        if name == "render":
            s.add_argument("--games", help="games list to use (default <out>/games.csv)")
            s.add_argument("--only", help="render just this game (its number or name)")
            s.add_argument("--action-only", action="store_true",
                           help="render only the action-only version (e.g. after adding period starts)")
            s.add_argument("--check-only", action="store_true",
                           help="sync the cameras and make the check image and video, without rendering the game")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
