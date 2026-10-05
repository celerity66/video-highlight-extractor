# Goalie Highlight Extractor

Turn hours of raw game footage into a short highlight reel of the action.

If you film a goalie with a GoPro or another fixed camera, most of the video
is the goalie standing around while play is at the other end. This script
finds the parts with motion in front of the net, cuts them into separate
clips, and can join them into one highlight video. It doesn't upload
anything, and your original files are never changed.

```bash
python3 goalie_highlight_extractor.py period1.mp4 --merge
```

> **Tools in this repository**
> * `goalie_highlight_extractor.py`: highlights from one camera focused on a goalie (this page).
> * `multicam_director.py`: one video per game from several cameras around the rink,
>   always showing the camera closest to the action, plus an action-only cut.
>   See [MULTICAM.md](MULTICAM.md).

---

## Quick start

### 1. Install the requirements

You need **Python 3**, **OpenCV** and **ffmpeg**.

| System | Command |
|---|---|
| Ubuntu / Debian / WSL | `sudo apt install python3 python3-pip ffmpeg` then `pip install opencv-python` |
| macOS (Homebrew) | `brew install python ffmpeg` then `pip3 install opencv-python` |
| Windows | Easiest is to use [WSL](https://learn.microsoft.com/windows/wsl/install) and follow the Ubuntu line. Or install Python from [python.org](https://www.python.org/downloads/), ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) (add it to your PATH), then `pip install opencv-python` |

If pip refuses with an "externally managed environment" error, add
`--break-system-packages` to the pip command.

Check that everything works:

```bash
python3 -c "import cv2; print('OpenCV', cv2.__version__)"
ffmpeg -version
```

### 2. Do a dry run

A dry run lists the segments the script would keep, without cutting
anything. It's the quickest way to see whether the settings suit your
footage.

```bash
python3 goalie_highlight_extractor.py period1.mp4 --dry-run
```

```
Found 14 active segment(s), totaling 6.2 min out of 22.5 min (28% kept)

  Segment 1: 00:00:41.20 - 00:01:12.60
  Segment 2: 00:02:05.00 - 00:02:31.40
  ...
```

### 3. Cut the clips

```bash
python3 goalie_highlight_extractor.py period1.mp4 --merge
```

The clips are saved to a `highlights/` folder in the directory you ran the command from:

```
highlights/
  period1_clip01.mp4
  period1_clip02.mp4
  ...
  period1_highlights_merged.mp4   <- only with --merge
```

---

## Processing a whole game (batch mode)

Cameras split long recordings into several files. Point the script at a
folder and add `--batch`. It works out which files belong to the same
recording, joins them in order, and then processes each recording.

```bash
python3 goalie_highlight_extractor.py /path/to/game_folder --batch --merge-all
```

`--merge-all` produces one `game_highlights_merged.mp4` covering every
recording in the folder. Use `--merge` instead to get one merged video per
recording.

### Recognised file names

| Camera / naming style | Example files | Grouped by |
|---|---|---|
| GoPro HERO5 and newer | `GH010042.MP4`, `GH020042.MP4` | The last 4 digits (recording ID), ordered by chapter |
| GoPro (other models) | `GP010042.MP4`, `GOPR0042.MP4` | Same as above |
| Renamed files | `001_01_Game.mp4`, `001_02_Game.mp4` | First number = recording, second number = part |
| Timestamped cameras | `20260905112617_00001.MP4`, `2026_0905_112938_001.MP4` | Files that start where the previous one ended (within 10s) |
| Anything else | `saturday.mp4` | Processed on its own |

Only `.mp4` and `.mov` files are picked up. The joined files are written
temporarily to `highlights/_stitched/` and deleted afterwards. Use
`--keep-stitched` to keep them. Because the full video is copied while
joining, you need enough free disk space for the largest recording.

---

## Options

| Option | Default | What it does |
|---|---|---|
| `--threshold N` | `15` | How much motion counts as "action". **Lower = more sensitive** (more and longer clips). Higher = only big movement. |
| `--min-gap S` | `3.0` | Join two segments if they're less than this many seconds apart. |
| `--min-duration S` | `1.5` | Ignore bursts of motion shorter than this (filters out flickers and noise). |
| `--pad-before S` | `5.0` | Seconds added before each segment so you see the play build up. |
| `--pad-after S` | `5.0` | Seconds added after each segment. |
| `--sample-rate N` | `5` | Check every Nth frame. Higher is faster but less precise. |
| `--roi x,y,w,h` | whole frame | Only look for motion inside this rectangle, or several: `"x,y,w,h; x,y,w,h"` (see below). |
| `--merge` | off | Also create one merged video (per recording in batch mode). |
| `--merge-all` | off | Batch mode: create one merged video for the whole folder. |
| `--dry-run` | off | Print the segments only; don't cut anything. |
| `--out-dir DIR` | `highlights` | Where to save the clips. |
| `--batch` | off | Treat the input as a folder (see above). |
| `--title "A|B|C"` | none | Put a title card at the start of the merged video (see "Title card"). |
| `--title-seconds S` | `3` | How long the title card shows. |
| `--add-title VIDEO` | | Add the `--title` card to an existing video and stop. |
| `--keep-stitched` | off | Keep the temporary joined files. |
| `--config FILE` | none | Use a config file (see "Several cameras" below). |
| `--list` | off | Config mode: list each camera's recordings and stop. |
| `--game NAME` | all | Config mode: only do this game. |

Run `python3 goalie_highlight_extractor.py --help` for the full list.

---

## Tuning tips

Start with `--dry-run` and look at the "% kept" figure.

- **Too much kept** (crowd, bench, or the far end of the rink triggers clips):
  raise `--threshold` (try 20–30) or use `--roi`. For a camera behind the net,
  see "Several boxes" below.
- **Missing saves**: lower `--threshold` (try 8–12).
- **Lots of tiny clips**: raise `--min-gap` (e.g. `8`) so nearby action joins
  into one clip.
- **Clips start too late / end too early**: increase `--pad-before` /
  `--pad-after`.
- **Analysis is slow**: raise `--sample-rate` to `10`. This works well for
  footage at 60 fps or higher.

### Restricting detection to the crease (`--roi`)

If the camera sees more than the goal area, tell the script to only watch
a box inside the frame. The format is `x,y,w,h`, all in whole pixels (not
percentages):

- `x`, `y`: the box's **top-left** corner. `0,0` is the top-left corner of
  the video. `x` increases to the right and `y` increases **downwards**.
- `w`, `h`: the box's width and height.

The box must fit inside the frame (`x + w` no more than the video width,
`y + h` no more than the height). Otherwise the script stops and tells you
the frame size.

To find the numbers, save a single frame and open it in any image editor
that shows the cursor position (Paint, GIMP, Preview, etc.). Image editors
also put 0,0 at the top-left, so you can use their numbers directly:

```bash
ffmpeg -ss 60 -i period1.mp4 -frames:v 1 frame.png
```

Examples for a 1920×1080 video:

| Area to watch | `--roi` |
|---|---|
| Left half | `0,0,960,1080` |
| Bottom half | `0,540,1920,540` |
| Bottom-left quarter | `0,540,960,540` |
| Centre box, 800×600 | `560,240,800,600` |

```bash
python3 goalie_highlight_extractor.py period1.mp4 --roi 0,540,960,540 --dry-run
```

#### Several boxes

Give several boxes separated by `;` (in quotes), and only motion inside them
counts:

```bash
python3 goalie_highlight_extractor.py period1.mp4 --roi "0,640,640,800; 1840,640,720,800" --dry-run
```

This is the best setup for a camera **behind the net**, looking down the ice
over the goalie. A single box around the crease doesn't work well there: the
goalie is so close to the camera that his own shuffling and tracking of the
play counts as lots of motion, even when the play is at the far end. Instead,
put one box on each side of the net, below the far end of the rink. Motion
there means players are in his zone.

---

## Title card

A title card is a few seconds of text on a plain background at the start of the
merged video, such as the event, the opponent and the date:

```bash
python3 goalie_highlight_extractor.py period1.mp4 --merge --title "Spring Cup|Game 2 vs Hawks|Saturday, May 4, 2026|Goalie Highlights"
```

Each `|` starts a new line; the first line is the biggest. To add one to a video
you've already made:

```bash
python3 goalie_highlight_extractor.py --add-title period1_highlights_merged.mp4 --title "Spring Cup|Game 2 vs Hawks"
```

Only the card itself is encoded, in the same format as the video, so the video
keeps its original quality and this takes seconds. It works for H.264 and H.265
video (what GoPros and most cameras record). In config mode, give each game
`title = ["line 1", "line 2", ...]` instead.

---

## Several cameras: config mode

When a rink has a camera behind each net, the goalie is in front of one
camera in periods 1 and 3 and the other in period 2. A config file
describes your cameras and games once, then the script uses the right
camera and the right part of the recording for each period, and joins every
period into one video per game. Config mode needs **Python 3.11 or newer**.

1. Copy `goalie_config.example.toml` to `goalie_config.toml` and set each
   camera's folder.
2. List each camera's recordings:
   ```bash
   python3 goalie_highlight_extractor.py --config goalie_config.toml --list
   ```
   ```
   Camera 'cam10'  (/mnt/d/footage/VIDEO/10)
     rec_20260907100804   3 file(s), 72.9 min   [2026_0907_100804_014.MP4 ... 2026_0907_105805_016.MP4]
   ```
3. Add a `[[games]]` block per game: which recording from each camera, and for
   each period, the camera and where the period starts and ends in that
   camera's recording (`"H:MM:SS"`, as shown by a video player).
4. Dry run one game, then cut it:
   ```bash
   python3 goalie_highlight_extractor.py --config goalie_config.toml --game game1 --dry-run
   python3 goalie_highlight_extractor.py --config goalie_config.toml --game game1
   ```

The clips go to `highlights/<game>/` as `P1_cam10_clip01.mp4` and so on, with
`<game>_full_game_highlights_merged.mp4` covering every period.

What else the config can hold:

* `[settings]`: defaults for the options above (command-line options still win).
* `title = [...]` per game: a title card for that game's merged video. `[title]`
  sets `seconds` and `colour` (e.g. `"#0b2a6f"`) for all of them.
* `roi` per camera, and per period for a game where a camera was set up
  differently. A period's `roi` overrides its camera's.
* `[player]`: jersey colours. The script reports how much of the motion
  matches them, and with `color_filter = true` drops clips where they're
  missing. On real rink cameras colours often look much duller than the
  jersey, so check the reported percentage in a dry run before turning the
  filter on.

---

## How it works

1. Reads the video and compares every Nth frame with the previous one it
   checked, to measure how much of the picture changed.
2. Marks moments where the change is above `--threshold` as "active".
3. Throws away very short bursts, joins segments that are close together,
   and adds padding before and after each one.
4. Uses ffmpeg to cut each segment. It first tries a fast copy that keeps
   the original quality. If that fails, it re-encodes the clip.

Fast-copied clips start at the nearest keyframe, so a clip may begin a
moment earlier than the listed time. The padding makes this unnoticeable in
practice.

---

## Troubleshooting

**`ERROR: opencv-python is not installed`**
Run `pip install opencv-python` (add `--break-system-packages` if pip refuses).

**`ERROR: file not found`, but the file exists (WSL)**
Windows drives are lowercase under WSL: use `/mnt/d/...`, not `/mnt/D/...`.
Put paths with spaces in quotes.

**`ERROR: could not open video file`**
Your OpenCV was built without video support. Check with:
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -A2 FFMPEG
```
It should say `YES`. If not, reinstall with `pip install --force-reinstall opencv-python`.

**`WARNING: ffprobe couldn't read video lengths`**
ffmpeg isn't installed or isn't on your PATH. Timestamped files are then
grouped by a rougher rule (any file starting within 30 minutes of the
previous one).

**`No active segments found`**
Lower `--threshold`, or check that your `--roi` covers the goal area.

**Re-running the script**
Existing clips in the output folder with the same names are overwritten. Use
a different `--out-dir` if you want to keep an earlier run.

---

## License

MIT. See [LICENSE](LICENSE).
