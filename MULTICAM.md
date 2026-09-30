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
| `<game>_camera_chart.png` | Which camera was used when, across the whole game |
| `<game>_report.json` | What was found: camera offsets, periods, goals, any warnings |

Nothing is uploaded, and the original videos are never changed.

---

## How it works

1. **Finds the games.** Each camera splits its recording into ~25-minute files; these
   are joined back up. Recordings from different cameras that overlap in time become
   one game. Cameras whose clock is wrong (e.g. reset to 2021 or 2024 after losing
   power) are matched to games by comparing their motion with the other cameras.
2. **Lines the cameras up in time.** There's no sound to sync on, so:
   * cameras at the same end of the rink are lined up by their motion patterns
     (whistles, line changes and faceoffs happen at the same moment for all of them);
   * the two ends are lined up with the scoreboards: one camera per end sees a
     scoreboard, and the game clock starts and stops at the same moments on both.
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

### 1. Find the games

```bash
python3 multicam_director.py scan "/mnt/d/path/to/VIDEO"
```

This reads every video once to measure its motion. **The first run is slow**:
about 1 hour 40 minutes for 45 hours of footage (117 files) on a 16-core machine.
The results are cached, so later runs only read new files and take seconds.

It writes `game_videos/games.csv`:

| game | name | render | start | length_min | cameras | recordings | notes |
|---|---|---|---|---|---|---|---|
| 1 | | yes | 2026-09-05 11:20 | 68 | 5 6 7 8 9 10 | 5=2026_0905_112453_001.MP4; ... | |
| 2 | | yes | 2026-09-05 18:21 | 71 | 5 6 7 8 9 10 | ... | cam 8 matched by motion (2.1x vs cam 10) |

### 2. Check the list

Open `games.csv` in a spreadsheet or text editor:

* **name**: optional, e.g. the opponent. Used in the file names
  (`2026-09-05_Warriors_multicam.mp4`). Left blank, the date and time are used.
* **render**: `yes` or `no`.
* **recordings**: which recording each camera contributes, as
  `camera=first file name`. Fix it here if a camera was matched to the wrong game.
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

### Options

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `game_videos` | Where videos, `games.csv` and the cache go |
| `--rink NAME` | `barnburner` | Rink settings to use (a folder in `rinks/`) |
| `--jobs N` | `6` | How many videos to read at once |
| `--force` | off | Overwrite `games.csv` / re-render existing videos |
| `--games FILE` | `<out>/games.csv` | (render) a different games list |
| `--only GAME` | all | (render) just this game, by number or name |

---

## Checking the results

* **`_sync_check.jpg`**: all cameras at the same moment. The play should match
  across cameras: the same players in the same places, and the same time on any
  visible scoreboard.
* **`_report.json`**: `notes` lists anything the tool wasn't sure about, e.g. a
  weak sync match or no scoreboard camera for a game (then only the full version
  is made).
* **In the videos**: if a cut jumps in time (the game clock on a visible scoreboard
  jumps between two shots), the cameras aren't lined up for that game.

---

## Rink settings

Everything specific to a rink and its camera positions is in `rinks/<rink>/`:

* `rink.json`: which end each camera is at; which part of each camera's view
  counts (weights per row of a 16 x 9 grid, and any area to ignore, such as the
  goalie in front of a camera behind the net); camera preferences; where the
  scoreboard is in the scoreboard camera's view; the period length; and the padding
  kept around each stoppage and after goals.
* `digits.npz`, `scoreboard_label.npy`, `far_board.npy`: pictures of the scoreboard's
  digits and labels, learned from footage at this rink.

If a camera is moved, its settings (and, for the scoreboard cameras, the pictures)
may need updating.

### The Barnburner settings

| Camera | Position | Notes |
|---|---|---|
| 5 | Side, near the right end | Sees the main scoreboard: clock, periods and goals come from here |
| 6 | Side, near the left end | |
| 7, 9 | Right end, watching the net from opposite sides | 7 preferred; each is chosen when the play is across the ice from it |
| 8, 10 | Behind the left net | The goalie is ignored; 10 also gives the long view down the ice and sees the far scoreboard |

Periods are 13 minutes. The action-only version keeps 3 s before each faceoff,
6 s after each whistle and 20 s after each goal.
