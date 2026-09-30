"""Working out game time from clocks that can be seen ticking but not read.

Without reading the digits, a ticking clock could be game time, the warm-up timer or an
intermission timer. What tells them apart is the structure of a game: periods of a known
length of running clock, each starting at a faceoff after a long stop and ending when
the clock runs out, which is followed by another long stop (the intermission or the end).
So: try each place the first period could start (after a long stop), count running
clock from there until a period's worth is used, and prefer the start whose periods end
just before long stops. Anything ticking outside the periods is a timer and is cut.
Inside the periods, play is only cut where a clock was clearly seen stopped: stretches
no clock could see are kept, so play is never cut on a guess.
"""
import numpy as np

from .motion import FPS
from .scoreboard import segments

LONG_STOP_S = 60        # a period starts after a stop at least this long
PERIODS = 3


def firm_running(state):
    """Running unless clearly stopped: unknown stretches count as running, except ones
    with a stopped clock seen on both sides."""
    run = state != 0
    unknown = segments(state == -1)
    for a, b in unknown:
        before = state[a - 1] if a > 0 else -1
        after = state[b] if b < len(state) else -1
        if before == 0 and after == 0:
            run[a:b] = False
    return run


def periods(state, period_s):
    """[(start, end)] sample ranges of the periods, best guess. The last one may be cut
    short by the end of the footage."""
    ticking = state == 1
    stops = segments(state == 0)
    segs = segments(ticking)
    if not segs:
        return []
    cands = [segs[0][0]]
    for a, b in stops:
        if b - a >= LONG_STOP_S * FPS:
            nxt = [s for s, _ in segs if s >= b]
            if nxt:
                cands.append(nxt[0])
    cands = sorted(set(cands))
    # count game time as running unless a clock was clearly seen stopped: where clocks
    # can't be seen (dark or flickering boards), counting only visible ticks runs slow
    counted = firm_running(state)
    cum = np.concatenate([[0], np.cumsum(counted)])
    need = period_s * FPS

    long_stops = [(a, b) for a, b in stops if b - a >= END_STOP_S * FPS]
    best = None
    for s1 in cands:
        out, s, clean, off_by = [], s1, 0, 0.0
        for p in range(PERIODS):
            k = int(np.searchsorted(cum, cum[s] + need))
            if k > len(counted):              # footage ends before the period does
                out.append((s, len(counted)))
                break
            # the period really ends at the clock running out, which is followed by a long
            # stop; the count is only approximate, so snap to a long stop near it
            near = [(abs(a - k), a, b) for a, b in long_stops if abs(a - k) <= SNAP_S * FPS]
            if near:
                d, a, b = min(near)
                out.append((s, a))
                clean += 1
                off_by += d / FPS
                after = b
            else:
                out.append((s, k))
                after = k
            nxt = [c for c in cands if c >= after]
            if not nxt:
                break
            s = nxt[0]
        # prefer periods that end at long stops, then more periods, then closer snaps
        key = (clean, len(out), -off_by)
        if best is None or key > best[0]:
            best = (key, out)
    return best[1]


END_STOP_S = 30         # a period's end is followed by a stop at least this long
SNAP_S = 90             # how far a counted period end may be from the stop that really ends it


def ends_at_stop(stops, k):
    """Whether a stop of at least END_STOP_S starts right at sample k."""
    return any(abs(a - k) <= FPS and b - a >= END_STOP_S * FPS for a, b in stops)


def running_in_periods(state, period_s):
    """(per-sample True where game time is running, the periods, a reason if the
    periods can't be trusted). Trusted only if all three periods are found and each
    complete one ends just before a stop, as a period does when its clock runs out."""
    ps = periods(state, period_s)
    stops = segments(state == 0)
    problem = None
    if len(ps) < PERIODS:
        problem = f"only {len(ps)} period(s) could be found from the clocks"
    else:
        for i, (a, b) in enumerate(ps[:-1]):
            if not ends_at_stop(stops, b):
                problem = f"period {i + 1} doesn't end at a long stop, so the periods may be wrong"
                break
        # the game's end may be past the footage, or not show a clear stop: rather than
        # guess where the last period ends, keep everything to the end of the footage
        a, b = ps[-1]
        if not ends_at_stop(stops, b):
            ps[-1] = (a, len(state))
    run = firm_running(state)
    mask = np.zeros(len(state), bool)
    for a, b in ps:
        mask[a:b] = run[a:b]
    return mask, ps, problem


def running_from_starts(state, starts, period_s):
    """Game time from period starts given by hand (sample indices, one per period):
    from each start, count a period's worth of clock (running unless clearly seen
    stopped; everything, if no clock was seen), end at the long stop nearest that, and
    never run past the next period's start. (mask, periods)."""
    n = len(state)
    counted = firm_running(state)
    cum = np.concatenate([[0], np.cumsum(counted)])
    long_stops = [(a, b) for a, b in segments(state == 0) if b - a >= END_STOP_S * FPS]
    starts = sorted(max(0, min(n - 1, s)) for s in starts)
    out = []
    for i, s in enumerate(starts):
        limit = starts[i + 1] if i + 1 < len(starts) else n
        k = int(np.searchsorted(cum, cum[s] + period_s * FPS))
        near = [(abs(a - k), a) for a, b in long_stops if abs(a - k) <= SNAP_S * FPS]
        if near:
            end = min(near)[1]
        elif i + 1 == len(starts):
            end = n                     # no clear end to the last period: keep to the end of the footage
        else:
            end = k + SNAP_S * FPS      # no clear end: allow for the count being approximate
        out.append((s, min(end, limit)))
    mask = np.zeros(n, bool)
    for a, b in out:
        mask[a:b] = counted[a:b]
    return mask, out
