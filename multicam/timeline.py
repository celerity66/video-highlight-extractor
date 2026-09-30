"""Putting every camera of a game on one timeline, from all the evidence available.

For every pair of cameras there may be several kinds of evidence:
  * motion: cameras at the same end see the same play, so their motion patterns match;
  * opposite: cameras at opposite ends see the play come and go in turn: when it's in
    one zone, that end is busy and the other end quiet, and every rush flips it. So the
    ice near two cameras at opposite ends is busiest in turn, and the offset where
    their activity is most opposite is the true one (checked on game 1: within ~1 s);
  * clocks: cameras that can see a scoreboard clock see it start and stop at the same
    moments, wherever they are in the rink;
  * manual: the same moment picked out by hand in two cameras (games.csv sync_points).
The strongest links are used to connect all the cameras (a maximum spanning tree); a
camera that can't be linked clearly is left out rather than put in the wrong place.
"""
import numpy as np

from . import sync
from .motion import FPS

MANUAL = 99.0           # strength given to a sync point set by hand
CLOCK_SEARCH_S = 1200   # how far from the camera clocks' offset to search (motion, opposite)
CLOCK_SEARCH_CLOCKS_S = 600   # ...and for scoreboard-clock patterns, which repeat each period
# how precise each kind of link is, when combining them: scoreboard clocks and same-end
# motion are good to ~0.2 s; opposite activity has a ~1 s bias between the ends, so it
# only decides the gap between the ends when nothing better links them
PRECISION = {"manual": 10.0, "clocks": 3.0, "motion": 1.0, "opposite": 0.2}


def near_activity(grid):
    """How busy the ice near the camera is (bottom rows of the frame), smoothed over 2 s,
    with slow changes (over 2 minutes) removed."""
    m = np.asarray(grid)[:, 5:, :].reshape(len(grid), -1).astype(np.float32).mean(axis=1)
    m = np.log(m + 1e-3)
    m = np.convolve(m, np.ones(2 * FPS) / (2 * FPS), mode="same")
    k = 120 * FPS
    m = m - np.convolve(m, np.ones(k) / k, mode="same")
    return (m - m.mean()) / (m.std() + 1e-9)


def most_opposite(a, b, lags):
    """(lag, strength) where b's activity is most opposite to a's: the lowest correlation,
    compared with the lowest at least 10 s away."""
    lags, sc = sync.correlate(a, b, lags)
    sc = np.where(np.isfinite(sc), sc, np.inf)      # shifts with too little overlap don't count
    if not np.isfinite(sc).any():
        return 0, 0.0
    i = int(np.argmin(sc))
    far = (np.abs(lags - lags[i]) >= 10 * FPS) & np.isfinite(sc)
    other = sc[far].min() if far.any() else 0
    return int(lags[i]), float(sc[i] / other) if other < 0 else 0.0

SYNC_MIN = 1.6          # weakest link that counts (every correct link seen so far was 1.7x or more)
CONFLICT = 2.0          # a link this strong that disagrees with the result means something is wrong
DISAGREE_S = 2.0        # other strong links should agree with the chosen ones within this...
TOLERANCE_S = {"opposite": 3.5}   # ...or this, for the less precise kinds


def pair_links(game, clock_runs, manual=None):
    """{(a, b): [(lag_samples, strength, kind), ...]} strongest first: b's time = a's time + lag.
    manual: {camera: seconds into its recording} for one moment picked out by hand."""
    cams = list(game.recs)
    links = {}
    signals = {c: sync.sync_signal(game.grids[c]) for c in cams}
    near = {c: near_activity(game.grids[c]) for c in cams}
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            if manual and a in manual and b in manual:
                links[(a, b)] = [(int(round((manual[b] - manual[a]) * FPS)), MANUAL, "manual")]
                continue
            e = game.expected_lag(a, b)
            # search +/- 20 min around what the camera clocks say: even believable
            # clocks can be ~15 min apart (game 6), so a narrower search can miss the answer
            lags = sync.lags_around(e, CLOCK_SEARCH_S) if e is not None else None
            options = [sync.best_lag(signals[a], signals[b], lags) + ("motion",),
                       most_opposite(near[a], near[b], lags) + ("opposite",)]
            if a in clock_runs and b in clock_runs:
                # clock patterns repeat from period to period, so search them more narrowly
                clags = sync.lags_around(e, CLOCK_SEARCH_CLOCKS_S) if e is not None else None
                ra, rb = clock_runs[a], clock_runs[b]
                options.append(sync.best_lag(ra - ra.mean(), rb - rb.mean(), clags, separation_s=30) + ("clocks",))
            links[(a, b)] = sorted(options, key=lambda o: -o[1])
    return links


def solve(cams, links, reference):
    """Offsets (seconds into each camera's recording at reference time 0) for every
    camera that can be linked, strongest links first. Returns (offsets, used, notes,
    conflicts): conflicts are strong links that disagree with the result."""
    def link(a, b):
        if (a, b) in links:
            lag, s, kind = links[(a, b)][0]
            return lag, s, kind
        lag, s, kind = links[(b, a)][0]
        return -lag, s, kind

    offsets = {reference: 0.0}
    used, notes = [], []
    while True:
        best = None
        for a in offsets:
            for b in cams:
                if b in offsets:
                    continue
                lag, s, kind = link(a, b)
                if s >= SYNC_MIN and (best is None or s > best[2]):
                    best = (a, b, s, lag, kind)
        if best is None:
            break
        a, b, s, lag, kind = best
        offsets[b] = offsets[a] + lag / FPS
        used.append((a, b, round(s, 2), kind))
    # refine: every strong link that agrees with the tree, weighted by its strength
    # (a weighted least-squares fit), averages out the error of individual links
    cams_in = sorted(offsets, key=lambda c: c != reference)
    rows, rhs, weights, conflicts = [], [], [], []
    for (a, b), options in links.items():
        for lag, s, kind in options:
            if a not in offsets or b not in offsets or s < 1.5:
                continue
            implied = offsets[a] + lag / FPS
            if abs(implied - offsets[b]) <= TOLERANCE_S.get(kind, DISAGREE_S):
                r = np.zeros(len(cams_in))
                r[cams_in.index(b)] += 1
                r[cams_in.index(a)] -= 1
                rows.append(r)
                rhs.append(lag / FPS)
                weights.append(s * PRECISION[kind])
            elif options.index((lag, s, kind)) == 0 and s >= CONFLICT:
                conflicts.append(f"cameras {a} and {b} disagree by {implied - offsets[b]:+.1f} s "
                                 f"({kind}, {s:.2f}x)")
    if len(cams_in) > 1 and rows:
        A = np.array(rows)[:, 1:] * np.sqrt(weights)[:, None]      # reference fixed at 0
        y = np.array(rhs) * np.sqrt(weights)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        offsets = {reference: 0.0, **{c: float(v) for c, v in zip(cams_in[1:], sol)}}
    return offsets, used, notes, conflicts


def combined_running(clock_states, offsets, t0, n):
    """Clock running (True) per sample of the shared timeline, from every camera's clock:
    running if any clock ticks, stopped if a clock was seen stopped and none ticked."""
    tick = np.zeros(n, bool)
    stop = np.zeros(n, bool)
    for c, st in clock_states.items():
        if c not in offsets:
            continue
        first = int(round((-offsets[c] - t0) * FPS))
        src0, dst0 = max(0, -first), max(0, first)
        m = min(len(st) - src0, n - dst0)
        if m <= 0:
            continue
        s = st[src0:src0 + m]
        tick[dst0:dst0 + m] |= s == 1
        stop[dst0:dst0 + m] |= s == 0
    state = np.full(n, -1)
    state[stop] = 0
    state[tick] = 1
    return state
