# Multi-camera Game Director

Turns footage from several fixed cameras around a rink into **one video per game
that always shows the camera closest to the action**, like a TV broadcast. It also
makes an **action-only version** with the warm-up, intermissions and stoppages cut
out, keeping the celebrations after goals.

For each game you get:

| File | What it is |
|---|---|
| `<game>_multicam.mp4` | The whole game, switching between cameras |
| `<game>_multicam_action.mp4` | Only the time the game clock is running (plus a few seconds either side), and 20 s after each goal |
| `<game>_sync_check.jpg` | Every camera at the same moment, to check they're lined up |
| `<game>_sync_check.mp4` | 20 seconds of every camera side by side, to check the sync by eye |
| `<game>_camera_chart.png` | Which camera was used when, across the whole game |
| `<game>_report.json` | What was found: camera offsets, periods, goals, any warnings |
| `share/…` | Small copies for sharing, made with the `share` command (see "Sharing copies") |

Nothing is uploaded, and the original videos are never changed.

---

## How it works

1. **Finds the games.** Each camera splits its recording into ~25-minute files; these
   are joined back up. Recordings from different cameras that overlap in time become
   one game. Cameras whose clock is wrong (e.g. reset to 2021 or 2024 after losing
   power) are matched to games by comparing their motion with the other cameras.
2. **Lines the cameras up in time.** There's no sound to sync on, so it uses every
   kind of evidence it can find, most precise first:
   * **sync points you set by hand** (see below), if any;
   * **scoreboard clocks**: it looks for a clock in every camera's view by itself (the
     one thing that changes exactly once a second), and the clock starts and stops at
     the same moments for every camera that sees one (to within about half a second);
   * **motion**: cameras at the same end see the same play;
   * **opposite activity**: when play is in one zone that end is busy and the other
     quiet, and every rush flips it, so the two ends can be lined up even without a
     scoreboard (to within about 1.5 seconds).
   The strongest links connect all the cameras. A camera that can't be linked
   clearly stops the game (see "Checking the results") rather than being put in the
   wrong place.
3. **Reads the scoreboard clock** from the camera that sees it clearly: when the
   clock runs, when it stops, where each period starts and ends, and when the score
   changes. The scoreboard's lights flicker badly on camera, so the digits are
   recognised from pictures of this scoreboard's digits, and only readings that agree
   with each other are used.
4. **Chooses a camera** for every moment: the one with the most movement in the part
   of the ice it covers best, preferring the angle that looks across the ice at the
   play, and avoiding switches that are too quick.
5. **Renders** both versions at 1080p.

---

## Requirements

The same as the goalie script: **Python 3**, **OpenCV** and **ffmpeg**
(see [README.md](README.md#1-install-the-requirements)). A fast computer helps:
every video is decoded, which takes a while for hours of 1440p footage.

---

## Usage

The footage folder must contain **one sub-folder per camera**, named after the
camera (`5`, `6`, `7`, `8`, `9`, `10` for this rink). Each folder can hold several
games.

### The easy way: one script

From the Ubuntu (WSL) terminal:

```bash
cd ~/hockey/videos
./run_multicam.sh "/mnt/d/path/to/VIDEO" [name] [rink]
```

* `name`: the output folder inside `game_videos/` (default: the folder above
  `VIDEO`; `.` means `game_videos` itself).
* `rink`: the rink settings (default: `four_nets`).

It connects the drive if needed, finds the games (the first time only) and pauses
for you to check `games.csv`, offers a quick sync check, then renders every game
not already done. Run it again after fixing anything in `games.csv`: finished work
is kept. The steps below are what it runs, for doing them one at a time.

### 1. Find the games

```bash
python3 multicam_director.py scan "/mnt/d/path/to/VIDEO"
```

This reads every video once to measure its motion. **The first run is slow**:
about 1 hour 40 minutes for 45 hours of footage (117 files) on a 16-core machine.
The results are cached, so later runs only read new files and take seconds.

It writes `game_videos/games.csv`:

| game | name | render | rink | start | length_min | cameras | recordings | sync_points | period_starts | warmup_camera | event | event_game | goalie_video | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | yes | barnburner | 2026-09-05 11:20 | 73 | 5 6 7 8 9 10 | 5=2026_0905_112453_001.MP4; ... | | | | | | | |
| 2 | | yes | four_nets | 2026-09-05 18:21 | 79 | 5 6 7 8 9 10 | ... | | | | | | | cam 8 matched by its reset clock (...) |

### 2. Check the list

Open `games.csv` in a spreadsheet or text editor:

* **name**: optional, e.g. the opponent. Used in the file names
  (`2026-09-05_Warriors_multicam.mp4`). Left blank, the date and time are used.
* **render**: `yes` or `no`.
* **rink**: which rink settings to use (a folder in `rinks/`); blank means `--rink`.
* **recordings**: which recording each camera contributes, as
  `camera=first file name`. Fix it here if a camera was matched to the wrong game,
  or remove a camera that shouldn't be used (e.g. one knocked to face the ceiling).
* **sync_points**: optional; see "Syncing by hand" below.
* **period_starts**: optional; see "Period starts by hand" below.
* **warmup_camera**: optional; the camera at your team's end. See "Our warm-up" below.
* **event**, **event_game**, **goalie_video**: optional; for the sharing copies. See
  "Sharing copies" below.
* **notes**: how wrong-clock cameras were matched.

Running `scan` again never overwrites an edited `games.csv`; it writes
`games_new.csv` instead (or use `--force`).

### 3. Make the videos

```bash
python3 multicam_director.py render "/mnt/d/path/to/VIDEO"
```

Games already rendered are skipped (`--force` redoes them). To do one game:

```bash
python3 multicam_director.py render "/mnt/d/path/to/VIDEO" --only 3
```

Each game takes about 20 minutes: a few minutes reading the scoreboard, then
about 10 minutes for the full video and 7 for the action-only one. If a render is
interrupted, running it again reuses the parts already made.

### Syncing by hand

If the tool can't link a camera (or the check video shows it out of step), pick one
moment you can see in two or more cameras, such as a puck drop or a goal, and note
where it is in each camera's file, as shown by a video player. Put them in the
`sync_points` column as `camera=file@minutes:seconds`, separated by `;`:

```
10=2026_0905_184631_005.MP4@03:12.4; 7=2026_0905_185418_005.MP4@05:40.0
```

The file is whichever of that camera's ~25-minute files contains the moment. These
points are trusted over everything else, and they can include as many cameras as
you like. A camera not listed is linked to the others automatically.

### Period starts by hand

Where no camera can read the scoreboard, the action-only version relies on the
clocks it can see ticking. If they don't show a clear period structure, that version
is skipped (the report says why). You can then give the time of each period's
opening faceoff in the `period_starts` column, in the same form as sync points, one
per period, in any synced camera:

```
7=2026_0907_101548_014.MP4@04:12; 7=2026_0907_104049_015.MP4@01:30; 9=2026_0907_104851_00006.MP4@08:05
```

From each start the tool counts one period of running clock, cutting only where a
clock was clearly seen stopped. Then render just that version:

```bash
python3 multicam_director.py render "/mnt/d/path/to/VIDEO" --only 5 --action-only
```

### Our warm-up

Before the first faceoff each team warms up at its own end, and following the
action would often show the other team. Put the camera at your team's end in the
`warmup_camera` column (e.g. `10`, the high camera behind your net in period 1).
The video then stays on that camera from the start until 8 seconds after the first
faceoff, in both versions. This needs the first faceoff, so it works when the periods
are known: from a readable scoreboard, the clocks, or `period_starts`. If that
camera started recording late, the video follows the action until it starts.

### Sharing copies

```bash
python3 multicam_director.py share
```

makes a small copy of each rendered game's action-only video in `game_videos/share/`,
ready to upload to a photo server or share by link:

* 720p H.264, about a third of the size, and plays in any browser or phone;
* a 3-second title card first, from the `event`, `event_game` and `name` columns:
  "Barnburner / Game 3 vs FL Warriors / Sunday, Sep 6, 2026 / Game Action";
* a file name to match: `2026-09-06 Barnburner G3 vs FL Warriors (action).mp4`;
* the game's date and time stamped in, so a photo library (Immich, Google Photos,
  etc.) puts it on the right day.

`goalie_video` can name another video of the game, such as a goalie highlight video
from `goalie_highlight_extractor.py`; it gets the same treatment, as "(goalie)".

| Option | Default | What it does |
|---|---|---|
| `--full` | off | Also share the full-game videos |
| `--only GAME` | all | Just this game, by number or name |
| `--timezone TZ` | `America/New_York` | Where the games were played, for the dates |
| `--goalie-label TEXT` | `Goalie Highlights` | The goalie video's last title card line, e.g. `"Goalie Highlights  #1"` |
| `--no-goalie-card` | off | No title card on goalie videos, when they already have one from the goalie script's `--title` |
| `--card-colour 0xRRGGBB` | navy | The title card's colour |

Copies already made are skipped; delete one to make it again.

### Options

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `game_videos` | Where videos, `games.csv` and the cache go |
| `--rink NAME` | `barnburner` | Rink settings to use (a folder in `rinks/`) |
| `--jobs N` | `6` | How many videos to read at once |
| `--force` | off | Overwrite `games.csv`; re-render existing videos, even if problems were found |
| `--games FILE` | `<out>/games.csv` | (render) a different games list |
| `--only GAME` | all | (render) just this game, by number or name |
| `--action-only` | off | (render) make only the action-only version, e.g. after adding period starts |
| `--check-only` | off | (render) sync the cameras and make the check image and video only: a few minutes instead of 20 |

---

## Checking the results

* **`_sync_check.jpg` / `_sync_check.mp4`**: all cameras at the same moment. The play
  should match across cameras: the same players in the same places, the puck moving
  together, and the same time on any visible scoreboard.
* **When a game isn't rendered**: the tool stops and explains instead of making a
  video it can't trust, for example when a camera can't be synced, or when the
  scoreboard camera doesn't show the scoreboard described in the rink settings. Fix
  it in `games.csv` (add a sync point, or remove the camera) and run again, or use
  `--force` to render anyway.
* **`_report.json`**: what was found and how each camera was linked; `notes` lists
  anything the tool wasn't sure about.
* **In the videos**: if a cut jumps in time (the game clock on a visible scoreboard
  jumps between two shots), the cameras aren't lined up for that game.

---

## Rink settings

Everything specific to a rink and its camera positions is in `rinks/<rink>/`:

* `rink.json`: which part of each camera's view counts (weights per row of a 16 x 9
  grid, and any area to ignore, such as the goalie in front of a camera behind the
  net); camera preferences; optionally, where a readable scoreboard is in one
  camera's view; the period length; and the padding kept around each stoppage and
  after goals.
* `digits.npz`, `scoreboard_label.npy`: pictures of a readable scoreboard's digits
  and labels, learned from footage at that rink (only for rinks that have one).

Scoreboard clocks are otherwise found automatically in every game, so rink settings
don't need to know where cameras are aimed. Choose the settings per game with the
`rink` column in `games.csv`.

### `four_nets`: four cameras behind the nets (games 2-6 of the Barnburner)

| Camera | Position | Notes |
|---|---|---|
| 7 (low), 9 (high) | Behind one net | The goalie and net in the middle of the view are ignored (a bigger area for the low cameras) |
| 8 (low), 10 (high) | Behind the other net | Same |
| 5, 6 | Sides | |

The high cameras (9 and 10) see over the net and down the ice, so they also give the
long view used when play is far from every camera, such as in the neutral zone.

No readable scoreboard: game time comes from the clocks seen ticking, or from
period starts given in `games.csv`. Goals aren't detected.

### `barnburner`: the rink of game 1 of the Barnburner

| Camera | Position | Notes |
|---|---|---|
| 5 | Side, near the right end | Sees the main scoreboard: clock, periods and goals come from here |
| 6 | Side, near the left end | |
| 7, 9 | Right end, watching the net from opposite sides | 7 preferred; each is chosen when the play is across the ice from it |
| 8, 10 | Behind the left net | The goalie is ignored; 10 also gives the long view down the ice |

For both: periods are 13 minutes, and the action-only version keeps 3 s before each
faceoff, 6 s after each whistle and 20 s after each goal (where goals are detected).
