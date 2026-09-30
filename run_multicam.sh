#!/usr/bin/env bash
# Make multi-camera game videos from a folder of camera footage, step by step.
#
#   ./run_multicam.sh "/mnt/d/path/to/VIDEO" [name] [rink]
#
#   VIDEO  folder with one sub-folder per camera (5, 6, 7, 8, 9, 10)
#   name   output folder name inside game_videos/ (default: the folder above VIDEO);
#          "." means game_videos itself
#   rink   rink settings in rinks/ (default: four_nets)
#
# Run it again at any time: finished work is kept, so it carries on where it stopped.
# See MULTICAM.md for details.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$HERE/multicam_director.py"

if [ $# -lt 1 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
FOOTAGE="${1%/}"
# default: the folder above VIDEO, with anything but letters, numbers, . - _ turned into _
NAME="${2:-$(basename "$(dirname "$FOOTAGE")" | sed 's/[^A-Za-z0-9._-]\+/_/g')}"
RINK="${3:-four_nets}"
if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ || "$NAME" == ".." ]]; then
    echo "The name '$NAME' can't be used: use letters, numbers, - and _ only (or . for game_videos itself)."
    exit 1
fi
OUT="$(realpath -m "$HERE/game_videos/$NAME")"
GAMES="$OUT/games.csv"
SHOW="game_videos"; [ "$NAME" != "." ] && SHOW="game_videos/$NAME"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
pause() { read -r -p "$1 Press Enter to continue (or Ctrl+C to stop). " _; }
ask() { local a; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]; }

# 1. the footage must be reachable (Windows drives appear under /mnt/<letter>)
step "Checking the footage"
if [[ "$FOOTAGE" =~ ^/mnt/([a-z])/ ]] && [ -z "$(ls -A "/mnt/${BASH_REMATCH[1]}" 2>/dev/null)" ]; then
    DRIVE="${BASH_REMATCH[1]}"
    echo "Drive ${DRIVE^^}: isn't connected to Linux yet. Connecting it (this asks for your password)..."
    sudo mount -t drvfs "${DRIVE^^}:" "/mnt/$DRIVE"
fi
if [ ! -d "$FOOTAGE" ]; then
    echo "Can't find the footage folder: $FOOTAGE"
    echo "(Windows drives are under /mnt/ with a lowercase letter, e.g. /mnt/d/...; put the path in quotes.)"
    exit 1
fi
CAMS=$(find "$FOOTAGE" -mindepth 1 -maxdepth 1 -type d -printf '%f ' | tr -s ' ')
echo "Footage: $FOOTAGE"
echo "Camera folders: ${CAMS:-none found}"
echo "Output:  $SHOW   (rink settings: $RINK)"
if [ ! -f "$HERE/rinks/$RINK/rink.json" ]; then
    echo "No rink settings called '$RINK'. Available: $(ls "$HERE/rinks" | tr '\n' ' ')"
    exit 1
fi
mkdir -p "$OUT"

# 2. find the games (once), then let the user check the list
if [ ! -f "$GAMES" ]; then
    step "Finding the games (the first time takes about a minute per 25-minute video file)"
    python3 "$TOOL" scan "$FOOTAGE" --out "$OUT" --rink "$RINK"
    echo
    echo "Now check the games list:"
    echo "  $GAMES"
    echo "  (from Windows: \\\\wsl.localhost\\Ubuntu${GAMES//\//\\})"
    echo "  - name:   optional, e.g. the opponent (used in the file names)"
    echo "  - render: set to 'no' for anything that isn't a real game"
    echo "  Save it as CSV and keep the columns as they are."
    pause "When you've checked it,"
else
    step "Using the games list found at $SHOW/games.csv"
    echo "(To look for the games again, delete that file and run this again.)"
fi

# 3. optional quick sync check
if ask "Check that the cameras are in sync before rendering (a few minutes per game)?"; then
    step "Checking the sync"
    python3 "$TOOL" render "$FOOTAGE" --out "$OUT" --check-only
    echo
    echo "Watch each game's *_sync_check.mp4 in $SHOW: all cameras should move together."
    echo "If a game said PROBLEM, fix it in games.csv (see MULTICAM.md, 'Syncing by hand') first."
    pause "When you're happy,"
fi

# 4. render
step "Rendering (about 20-30 minutes per game; games already done are skipped)"
python3 "$TOOL" render "$FOOTAGE" --out "$OUT"

step "Finished"
ls -1 "$OUT" | grep -E '_multicam(_action)?\.mp4$' | sed 's/^/  /' || echo "  (no videos made)"
echo
echo "Videos are in $SHOW (from Windows: \\\\wsl.localhost\\Ubuntu${OUT//\//\\})"
echo "Any game marked NOT RENDERED above needs a fix in games.csv; then run this script again."
