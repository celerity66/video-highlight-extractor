"""Smaller copies of the videos for sharing (e.g. on a photo server): 720p H.264, a short
title card at the start, a clear file name, and the game's date and time stamped in, so
a photo or video library puts each video on the right day."""
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CARD_S = 3
# event name, game, date, what the video is: size, font and height on a 1280x720 card
CARD_LINES = [(54, BOLD, 230), (44, BOLD, 320), (34, FONT, 390), (38, FONT, 470)]


def jobs(game, out_dir, stem, full=False, tz="America/New_York", goalie_label="Goalie Highlights"):
    """(source video, file name, title card lines, date and time) for each sharing copy of a game."""
    when = datetime.strptime(game["start"], "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(tz))
    event, no, opp = game["event"], game["event_game"], game["name"]
    vs = f"vs {opp}" if opp else ""
    lines = [event or "Game", f"Game {no}  {vs}".strip() if no else vs, when.strftime("%A, %b %-d, %Y")]
    name = " ".join(x for x in (f"{when:%Y-%m-%d}", event, f"G{no}" if no else "", vs) if x)
    name = "".join(ch for ch in name if ch not in '/\\:*?"<>|')
    wanted = [("action", f"{stem}_multicam_action.mp4", "Game Action", 0)]
    if full:
        wanted.append(("full game", f"{stem}_multicam.mp4", "Full Game", 2))
    out = [(os.path.join(out_dir, src), f"{name} ({kind}).mp4", lines + [label], when + timedelta(minutes=m))
           for kind, src, label, m in wanted]
    if game.get("goalie_video"):
        # a minute after the game's action video, so it sorts just after it
        out.append((game["goalie_video"], f"{name} (goalie).mp4", lines + [goalie_label],
                    when + timedelta(minutes=1)))
    return out


def _esc(path):
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def make(src, out, lines, when, colour="0x0b2a6f", card=True):
    """Encode src to out (720p, title card first unless card=False, dated). Written to a
    temporary name first, so an interrupted copy is never mistaken for a finished one."""
    tmp = tempfile.mkdtemp(dir=os.path.dirname(out))
    try:
        draws = []
        for i, (text, (size, font, y)) in enumerate(zip(lines, CARD_LINES)):
            if not text:
                continue
            p = os.path.join(tmp, f"line{i}.txt")      # in a file: no escaping of names needed
            with open(p, "w") as fh:
                fh.write(text)
            draws.append(f"drawtext=fontfile={font}:textfile={_esc(p)}:fontsize={size}:fontcolor=white:"
                         f"x=(w-text_w)/2:y={y}")
        fade = ",".join(draws + [f"fade=t=out:st={CARD_S - 0.5}:d=0.5"])
        video = "[1:v]scale=1280:720,fps=30,format=yuv420p,setsar=1"
        graph = (f"[0:v]{fade},format=yuv420p,setsar=1[c];{video}[v];[c][v]concat=n=2:v=1:a=0[out]" if card
                 else f"{video}[out]")
        stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        part = os.path.join(tmp, "video.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=1280x720:r=30:d={CARD_S}",
                        "-i", src, "-filter_complex", graph, "-map", "[out]", "-an",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-g", "60", "-movflags", "+faststart",
                        "-metadata", f"creation_time={stamp}", "-metadata", f"title={os.path.basename(out)[:-4]}",
                        "-metadata", f"comment={' - '.join(x for x in lines if x)}", part], check=True)
        os.replace(part, out)
    finally:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
