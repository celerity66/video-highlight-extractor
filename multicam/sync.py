"""Lining cameras up in time.

Within each end of the rink, cameras see the same play, so their motion patterns match
strongly. Between the two ends they don't (each end sees different play), so the ends
are lined up using the scoreboards instead: the game clock starts and stops at the same
moments on both scoreboards (see scoreboard.py).
"""
import numpy as np

from .motion import FPS

MIN_OVERLAP_S = 60
SEPARATION_S = 5        # a "second best" match must be at least this far from the best


def sync_signal(grid):
    """Overall motion, smoothed over 1 s, with slow trends (lighting, view) removed."""
    m = np.asarray(grid).reshape(len(grid), -1).astype(np.float32).mean(axis=1)
    m = np.log(m + 1e-3)
    m = np.convolve(m, np.ones(FPS) / FPS, mode="same")
    k = 60 * FPS
    m = m - np.convolve(m, np.ones(k) / k, mode="same")
    return (m - m.mean()) / (m.std() + 1e-9)


def correlate(ref, x, lags=None):
    """Normalised correlation of x against ref at each lag, where lag L means
    x[i + L] lines up with ref[i]. Returns (lags, scores)."""
    n = len(ref) + len(x)
    nfft = 1 << (n - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    if lags is None:
        lags = np.arange(-len(ref) + MIN_OVERLAP_S * FPS, len(x) - MIN_OVERLAP_S * FPS)
    lo = np.maximum(0, -lags)
    hi = np.minimum(len(ref), len(x) - lags)
    overlap = hi - lo
    scores = np.where(overlap > MIN_OVERLAP_S * FPS, full[lags % nfft] / np.maximum(overlap, 1), -np.inf)
    return lags, scores


def best_lag(ref, x, lags=None, separation_s=SEPARATION_S):
    """(lag in samples, strength): strength = best score / best score at least
    `separation_s` away (use a longer separation for slowly changing signals)."""
    lags, sc = correlate(ref, x, lags)
    i = int(np.argmax(sc))
    far = np.abs(lags - lags[i]) >= separation_s * FPS
    second = sc[far].max() if far.any() else 0
    return int(lags[i]), float(sc[i] / second) if second > 0 else 0.0


def lags_around(expected, spread_s):
    return np.arange(int(expected - spread_s * FPS), int(expected + spread_s * FPS) + 1)
