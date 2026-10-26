"""Exhaustive search -- the baseline a referee will propose.

Cost is O(T n^2 2^(n^2)).  It is the only method that covers all six surfaces
for every T with no caveats, and it is the one the other two are validated
against; it is also hopeless past n = 6, which is exactly the point the paper
has to make quantitatively.

The search enumerates seeds by population m, ascending, and stops as soon as
n^2/m can no longer beat the incumbent -- valid because |X(T)| <= n^2.  That
alone removes most of the space, since the optimum is around 2-3.

`chunk` / `chunks` split the enumeration for an array job on the cluster, by
contiguous slices so that each board index is generated exactly once across the
whole job.

`batch` is deliberately small.  The inner loop touches ~290 temporaries per
batch, so throughput is governed by whether those stay in cache: measured on one
core, 32 K boards per batch (0.3 MB arrays) runs at 1.34e6 boards/s against
4.5e5 at 2 M boards per batch (16.8 MB) -- a factor of three for a parameter
that looks like it should not matter.
"""
import time

import numpy as np

from . import bounds as bd
from . import simulate as sim
from . import topology as tp

_POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def _popcount(a):
    """Vectorised popcount for a uint64 array."""
    a = a.astype(np.uint64, copy=False)
    out = np.zeros(a.shape, np.int16)
    for shift in range(0, 64, 8):
        out += _POP8[((a >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return out


def _step_batch(x, nbl, N):
    """One generation for a whole array of boards at once."""
    y = np.zeros_like(x)
    one = np.uint64(1)
    for c in range(N):
        s = np.zeros(x.shape, np.int8)
        for k in nbl[c]:
            s += ((x >> np.uint64(k)) & one).astype(np.int8)
        alive = (s == 3) | ((s == 2) & (((x >> np.uint64(c)) & one) == 1))
        y |= alive.astype(np.uint64) << np.uint64(c)
    return y


def search(n, T, cfg, corner_policy='simple', batch=1 << 15, chunk=0, chunks=1,
           progress=None):
    """Exact G_T(n) by enumeration.  Returns a result dict."""
    cfg = tp.parse(cfg)
    N = n * n
    if N > 63:
        raise ValueError('n=%d needs more than 63 bits per board' % n)
    nbl = sim.neighbour_lists(n, cfg, corner_policy)

    t0 = time.time()
    best_rate, best_seed, best_cells = 0.0, None, 0
    total = np.uint64(1) << np.uint64(N)

    # Each chunk takes a CONTIGUOUS slice of the board space.  Striding by
    # `x % chunks == chunk` instead would make every chunk walk the whole
    # range and discard 63/64 of it, so the scan would be repeated `chunks`
    # times over; with a contiguous slice each board index is generated once
    # across the whole job, and the batches stay large enough for numpy to be
    # efficient.
    span = total // np.uint64(chunks)
    lo = np.uint64(chunk) * span
    end = total if chunk == chunks - 1 else lo + span

    step = np.uint64(batch)
    scanned = 0
    while lo < end:
        hi = min(end, lo + step)
        xs = np.arange(lo, hi, dtype=np.uint64)
        lo = hi
        if xs.size == 0:
            continue

        m = _popcount(xs)
        # a seed below MIN_SEED produces nothing; a seed too big cannot win
        keep = m >= bd.MIN_SEED
        if best_rate > 0:
            keep &= (N / np.maximum(m, 1)) > best_rate
        xs, m = xs[keep], m[keep]
        scanned += int(xs.size)
        if xs.size == 0:
            continue

        y = xs
        for _ in range(T):
            y = _step_batch(y, nbl, N)
        rate = _popcount(y) / m
        k = int(rate.argmax())
        if rate[k] > best_rate:
            best_rate = float(rate[k])
            best_seed = int(xs[k])
            best_cells = int(_popcount(y[k:k + 1])[0])
        if progress:
            progress(int(lo - np.uint64(chunk) * span),
                     int(end - np.uint64(chunk) * span), best_rate)

    return {'n': n, 'T': T, 'topology': tp.LETTERS[cfg], 'method': 'brute',
            'rate': best_rate, 'final_cells': best_cells,
            'm': sim.popcount(best_seed) if best_seed is not None else None,
            'board': sim.encode(n, best_seed) if best_seed is not None else None,
            'boards_scanned': scanned, 'chunk': chunk, 'chunks': chunks,
            'seconds': round(time.time() - t0, 3), 'status': 'OPTIMAL'}
