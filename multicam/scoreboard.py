"""Reading the game state from the scoreboard: clock running or stopped, periods, goals.

The main scoreboard camera (camera 5 at this rink) sees the scoreboard large enough to
read the clock digits, but its LEDs flicker in that camera's view: the digits are dark
in a third of frames, sometimes for over a minute. So:
  * each frame is lined up on the always-lit HOME label, then each digit is recognised
    by comparing it with pictures of the digits learned from this scoreboard;
  * only readings that agree with their neighbours are trusted;
  * whether the clock ran between two trusted readings follows from how much clock time
    passed compared with real time, which also covers dark stretches;
  * the last minute of each period shows tenths, which aren't read: there, the clock is
    taken as running when the seconds digit changes on every frame.
A second camera at the other end (camera 10) sees another scoreboard too small to read,
but whether its clock is ticking lines the two ends of the rink up in time.
"""
import os

import cv2
import numpy as np

from .motion import FPS, _frames, _key

# judging whether the clock ticks, from the seconds digit's frame-to-frame similarity
USABLE = 15.0          # digit contrast needed to judge its shape
WINDOW_S = 4           # seconds per judgement
CLOCK_LIT = 60.0       # clock contrast needed to read digits
DIGIT_OK = 0.6         # every digit must match its picture at least this well...
DIGIT_MARGIN = 0.08    # ...and clearly better than the second-best digit
SCORE_LIT = 40.0
SCORE_BIN_S = 10
SCORE_DROP = 0.25      # a score digit must match its old picture this much worse than usual...
SCORE_HOLD = 6         # ...for this many 10 s blocks in a row, to count as a new score
JITTER = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def segments(mask):
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    return out


def judge_window(m):
    """m: WINDOW_S x FPS similarities (rows = seconds, columns = position in the second).
    1 running, 0 stopped, -1 unknown."""
    if np.isnan(m).mean() > 0.4:
        return -1
    v = m[~np.isnan(m)]
    if (v < 0.9).mean() > 0.8:
        return 1                      # every frame changes: tenths (last minute of a period)
    if np.isnan(m).all(axis=0).any():
        return -1                     # a position in the second never seen: can't judge
    cols = np.nanmean(m, axis=0)
    tick = int(np.nanargmin(cols))
    rest = np.delete(cols, tick)
    if np.nanmedian(rest) > 0.95:
        col = m[:, tick]
        dips = np.sum(col[~np.isnan(col)] < np.nanmedian(rest) - 0.02)
        if cols[tick] < 0.93 and dips >= WINDOW_S - 1:
            return 1                  # one change at the same point every second: ticking
        if np.nanmin(cols) > 0.95 and np.nanmin(v) > 0.9:
            return 0                  # nothing changes: stopped
    return -1


def tick_state(sims):
    """Per sample: 1 running, 0 stopped, -1 no evidence."""
    n = len(sims)
    w = WINDOW_S * FPS
    state = np.full(n, -1)
    for i in range(0, n - w, FPS):
        v = judge_window(sims[i:i + w].reshape(WINDOW_S, FPS))
        if v >= 0:
            mid = slice(i + FPS, i + w - FPS)
            state[mid] = np.where(state[mid] == -1, v, np.maximum(state[mid], v))
    return state


def fill_unknown(state):
    """Unknown samples take the nearest known value; True where running."""
    known = np.flatnonzero(state >= 0)
    if len(known) == 0:
        return np.zeros(len(state), bool)
    pos = np.arange(len(state))
    idx = np.searchsorted(known, pos)
    left = known[np.clip(idx - 1, 0, len(known) - 1)]
    right = known[np.clip(idx, 0, len(known) - 1)]
    nearest = np.where(np.abs(pos - left) <= np.abs(right - pos), left, right)
    return state[nearest] == 1


def _norm(im):
    im = im.astype(np.float32)
    return (im - im.mean()) / (im.std() + 1e-6)


class Scoreboard:
    """The readable scoreboard, as seen by one camera."""

    def __init__(self, config, rink_dir):
        sc = config["scoreboard"]
        self.camera = sc["camera"]
        self.crop = sc["crop"]
        self.label_at = sc["label_at"]
        self.search = sc["label_search"]
        self.cells = sc["cells"]
        self.period_s = sc["period_minutes"] * 60
        self.label = np.load(os.path.join(rink_dir, "scoreboard_label.npy"))
        d = np.load(os.path.join(rink_dir, "digits.npz"))
        self.digits = {"tens": {}, "units": {}}
        for k in d.files:
            pos, digit = k.split("_")
            self.digits[pos][int(digit)] = d[k]

    # -- lining frames up ------------------------------------------------------------
    def shifts(self, a, block=25):
        """(dx, dy) of the HOME label from where it was measured, per sample."""
        lx, ly = self.label_at[0] - 2, self.label_at[1] - 2
        h, w = self.label.shape
        s = self.search
        out = np.zeros((len(a), 2), int)
        for i in range(0, len(a), block):
            m = np.max(a[i:i + block], axis=0).astype(np.float32)
            y0, x0 = max(0, ly - s), max(0, lx - s)
            win = m[y0:ly + h + s, x0:lx + w + s]
            r = cv2.matchTemplate(win, self.label, cv2.TM_CCOEFF_NORMED)
            _, _, _, (bx, by) = cv2.minMaxLoc(r)
            out[i:i + block] = (bx + x0 - lx, by + y0 - ly)
        return out

    def patch(self, frame, shift, c, jitter=(0, 0)):
        x0, y0, x1, y1 = c
        ox = self.label_at[0] + shift[0] + jitter[0]
        oy = self.label_at[1] + shift[1] + jitter[1]
        return frame[oy + y0:oy + y1, ox + x0:ox + x1]

    def read_digit(self, frame, shift, cell_name, pos):
        """(digit, match, margin), trying the cell 1 pixel either way (edges fall between pixels)."""
        best = None
        for j in JITTER:
            im = _norm(self.patch(frame, shift, self.cells[cell_name], j))
            scores = sorted(((float((im * t).mean()), d) for d, t in self.digits[pos].items()), reverse=True)
            r = (scores[0][1], scores[0][0], scores[0][0] - scores[1][0])
            if best is None or r[1] > best[1]:
                best = r
        return best

    # -- the clock -----------------------------------------------------------------------
    def clock_values(self, a, sh):
        """Game clock in seconds (12:22 -> 742) per sample, -1 where not confidently read."""
        seconds_box = (self.cells["tens"][0] - 2, 6, self.cells["units"][2] + 4, 41)   # lit check
        clock_box = (self.cells["ten_minutes"][0] - 6, 6, self.cells["units"][2] + 4, 41)
        out = np.full(len(a), -1)
        for k in range(len(a)):
            p = self.patch(a[k], sh[k], seconds_box)
            if p.size == 0 or np.percentile(p, 95) - np.median(p) < CLOCK_LIT:
                continue
            m = self.read_digit(a[k], sh[k], "minutes", "units")
            t = self.read_digit(a[k], sh[k], "tens", "tens")
            u = self.read_digit(a[k], sh[k], "units", "units")
            if min(m[1], t[1], u[1]) < DIGIT_OK or min(m[2], t[2], u[2]) < DIGIT_MARGIN or t[0] > 5:
                continue
            # the "1" of 10:00-12:59: lit compared with the clock's dark background
            bg = np.median(self.patch(a[k], sh[k], clock_box))
            one = np.percentile(self.patch(a[k], sh[k], self.cells["ten_minutes"]), 95) - bg > 40
            out[k] = ((10 if one else 0) + m[0]) * 60 + t[0] * 10 + u[0]
        return out

    def seconds_similarity(self, a, sh):
        """Shape similarity of the seconds digit to the previous sample (1 = unchanged)."""
        sims = np.full(len(a), np.nan)
        prev = None
        for k in range(len(a)):
            p = self.patch(a[k], sh[k], self.cells["units"])
            cur = _norm(p) if p.size and np.percentile(p, 95) - np.median(p) > USABLE else None
            if cur is not None and prev is not None and cur.shape == prev.shape:
                sims[k] = (cur * prev).mean()
            prev = cur
        return sims

    # -- goals ----------------------------------------------------------------------------
    def score_bins(self, a, sh, team):
        c = self.cells[f"{team}_score"]
        b = SCORE_BIN_S * FPS
        panel = np.median(a.reshape(len(a), -1), axis=1)
        out = []
        for i in range(0, len(a), b):
            ims = []
            for k in range(i, min(i + b, len(a))):
                p = self.patch(a[k], sh[k], c)
                if p.size and np.percentile(p, 95) - panel[k] > SCORE_LIT:
                    ims.append(_norm(p))
            out.append(np.mean(ims, axis=0) if len(ims) >= 5 else None)
        return out


def trusted_points(vals):
    """(sample, clock seconds) for readings that agree with readings within +/- 5 s:
    the same value, or counting down no faster than 1 s per s."""
    idx = np.flatnonzero(vals >= 0)
    pts = []
    for n, k in enumerate(idx):
        near = idx[max(0, n - 12):n + 13]
        near = near[(np.abs(near - k) <= 5 * FPS) & (near != k)]
        if len(near) < 2:
            continue
        good = 0
        for j in near:
            dt, dv = (j - k) / FPS, vals[k] - vals[j]
            if dt > 0 and -0.5 <= dv <= dt + 1.2 or dt < 0 and -0.5 <= -dv <= -dt + 1.2:
                good += 1
        if good >= 0.75 * len(near):
            pts.append((int(k), int(vals[k])))
    return pts


def _place(mask, a, b, length, evidence):
    """Mark `length` running samples inside [a, b), where the tick evidence agrees best."""
    length = min(int(round(length)), b - a)
    if length <= 0:
        return
    sums = np.convolve(evidence[a:b].astype(np.float32), np.ones(length), mode="valid")
    start = a + (int(np.argmax(sums)) if sums.max() > 0 else (b - a - length) // 2)
    mask[start:start + length] = True


def play_mask(pts, tstate, period_s):
    """Per sample True while the game clock runs. Returns (mask, [(start, end) per period])."""
    n = len(tstate)
    ticks = fill_unknown(tstate)
    mask = np.zeros(n, bool)
    periods = []
    starts = [i for i, (k, v) in enumerate(pts) if v == period_s and i + 1 < len(pts) and pts[i + 1][1] < period_s]
    for p, si in enumerate(starts):
        ei = starts[p + 1] if p + 1 < len(starts) else len(pts)
        obs = []                      # one per displayed value: [value, first seen, last seen]
        for k, v in pts[si:ei]:
            if v > period_s:
                continue
            if obs and obs[-1][0] == v:
                obs[-1][2] = k
            elif not obs or v < obs[-1][0]:
                obs.append([v, k, k])
        running_obs = []
        for v, k0, k1 in obs:        # shown for about a tick: running; clearly longer: stopped
            running_obs.append((k1 - k0) / FPS < 1.3)
            if running_obs[-1]:
                mask[k0:k1 + 1] = True
        for i in range(len(obs) - 1):
            v1, _, k1 = obs[i]
            v2, k2, _ = obs[i + 1]
            gap, need = (k2 - k1) / FPS, max(0.0, v1 - v2 - 1)
            if need > gap + 1.5:
                continue              # impossible: a misreading
            if gap <= need + 1.2:
                mask[k1:k2] = True
                continue
            run = int(need * FPS) + FPS // 2
            before, after = running_obs[i], running_obs[i + 1]
            if before and not after:
                mask[k1:k1 + run] = True          # ran, then the whistle
            elif after and not before:
                mask[k2 - run:k2] = True          # faceoff, then ran
            elif run > FPS // 2:
                _place(mask, k1, k2, run, ticks)
            elif before and after:
                mask[k1:k2] = True
                stop = int((gap - need - 1) * FPS)
                if stop > 0:
                    sums = np.convolve(1 - ticks[k1:k2].astype(np.float32), np.ones(stop), mode="valid")
                    s0 = k1 + int(np.argmax(sums))
                    mask[s0:s0 + stop] = False
        # last minute, in tenths: running unless seen stopped, until the clock is used up
        v_last, _, k_last = obs[-1]
        limit = pts[ei][0] if ei < len(pts) else n
        used, k = 0, k_last
        while k < limit and used < v_last * FPS:
            if tstate[k] != 0:
                mask[k] = True
                used += 1
            k += 1
        periods.append((obs[0][1], k))
    return mask, periods


def score_change_bins(bins):
    lit = [j for j, b in enumerate(bins) if b is not None]
    if len(lit) < SCORE_HOLD + 3:
        return []
    cur = np.mean([bins[j] for j in lit[:6]], axis=0)
    base = np.median([(bins[j] * cur).mean() for j in lit[:6]])
    changes, n = [], 0
    while n < len(lit) - SCORE_HOLD:
        grp = [bins[j] for j in lit[n:n + SCORE_HOLD]]
        vs_cur = [(g * cur).mean() for g in grp]
        new = np.mean(grp, axis=0)
        vs_new = [(g * new).mean() for g in grp]
        if max(vs_cur) < base - SCORE_DROP and min(vs_new) > max(vs_cur) + 0.1:
            changes.append(lit[n])
            cur, base = new, np.median(vs_new)
            n += SCORE_HOLD
        else:
            n += 1
    return changes


def goal_stoppages(board, a, sh, mask):
    """[(sample where the clock stopped for the goal, team)]: the last stoppage before
    each score change appears. A change on both scores together is the camera, not a goal."""
    stops = [k for k in range(1, len(mask)) if mask[k - 1] and not mask[k]]
    changes = {t: [j * SCORE_BIN_S * FPS for j in score_change_bins(board.score_bins(a, sh, t))]
               for t in ("home", "away")}
    out = []
    for team, ks in changes.items():
        other = changes["away" if team == "home" else "home"]
        for kc in ks:
            if any(abs(kc - o) <= 30 * FPS for o in other):
                continue
            before = [k for k in stops if k <= kc]
            if before:
                out.append((before[-1], team))
    return sorted(out)


def far_board_similarity(path, config, rink_dir, cache_dir):
    """For the camera that sees the far scoreboard too small to read: similarity of its
    clock area to the previous sample, after lining each frame up on the board."""
    fb = config["far_scoreboard"]
    x, y, w, h = fb["crop"]
    out = os.path.join(cache_dir, _key(path, f"farclock{x}_{y}_{w}_{h}"))
    if os.path.exists(out):
        return np.load(out)
    board = np.load(os.path.join(rink_dir, "far_board.npy"))
    cx0, cy0, cx1, cy1 = fb["clock_in_board"]
    sims, prev = [], None
    for f in _frames(path, f"crop={w}:{h}:{x}:{y}", w, h):
        f = f.astype(np.float32)
        r = cv2.matchTemplate(f, board, cv2.TM_CCOEFF_NORMED)
        _, score, _, (bx, by) = cv2.minMaxLoc(r)
        c = f[by + cy0:by + cy1, bx + cx0:bx + cx1]
        cur = _norm(c) if score > 0.5 and c.shape == (cy1 - cy0, cx1 - cx0) else None
        sims.append((cur * prev).mean() if cur is not None and prev is not None else np.nan)
        prev = cur
    sims = np.array(sims, np.float32)
    np.save(out, sims)
    return sims
