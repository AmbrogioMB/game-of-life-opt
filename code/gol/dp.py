"""Transfer-matrix dynamic program for the one-step growth rate.

Why this exists
---------------
A referee will ask why the problem is not simply enumerated.  The answer is that
plain enumeration costs O(T n^2 2^(n^2)) and dies at n = 6, but there is a much
better elementary method, and the paper should present it rather than wait to be
shown it.  One Game of Life step is local, so after T steps a line of the board
depends only on the 2T+1 lines around it.  Sweeping the board line by line with
the last 2T lines as state gives an exact algorithm in

    O(n * 2^((2T+1) n))

time, against 2^(n^2) for enumeration.  For T = 1 that is O(n * 2^(3n)) and
reaches n = 12; the MILP's case is that the state space closes at T >= 2, where
the DP is hopeless and the MILP keeps going.

The ratio |X(1)|/|X(0)| is not additive, so it is handled by Dinkelbach's
method: for a trial value p/q the DP maximises the additive surrogate
q|X(1)| - p|X(0)| exactly, and the ratio of the maximiser becomes the next
trial value.  This terminates at the true optimum in a handful of iterations,
in exact integer arithmetic.

Applicability
-------------
The sweep direction must be free (unglued), and the gluing inside a line must
stay inside that line.  That holds for the square (sweep either way, nothing
glued) and the cylinder (sweep the rows, the columns wrap within each row).
It fails for the torus, Klein bottle, Moebius band and projective plane, where
either the sweep direction wraps or the seam maps a line onto a different line;
those need a cyclic DP costing an extra 2^(2n) and are left to the MILP and to
exhaustive search.
"""
import time

import numpy as np

from . import simulate as sim
from . import topology as tp

SUPPORTED = (tp.SQUARE, tp.CYLINDER)
NEG = np.int64(-(10 ** 15))


def supports(cfg, T=1):
    return tp.parse(cfg) in SUPPORTED and T == 1


def _tables(n, wrap):
    """Per-line neighbour-count tables.

    cnt3[u][j] = bits of line u at positions j-1, j, j+1
    cnt2[u][j] = bits of line u at positions j-1, j+1
    """
    C = 1 << n
    u = np.arange(C, dtype=np.int64)
    bits = ((u[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    cnt3 = np.zeros((C, n), np.int16)
    cnt2 = np.zeros((C, n), np.int16)
    for j in range(n):
        for d in (-1, 0, 1):
            k = j + d
            if wrap:
                k %= n
            elif not (0 <= k < n):
                continue
            cnt3[:, j] += bits[:, k]
            if d != 0:
                cnt2[:, j] += bits[:, k]
    return bits, cnt3, cnt2


def _maximise_surrogate(n, p, q, bits, cnt3, cnt2, popc):
    """max over X of  q|X(1)| - p|X(0)|, exactly, in integers.

    Returns (value, |X(1)|, |X(0)|) for a maximiser.
    """
    C = 1 << n
    val = np.full((C, C), NEG, np.int64)   # state (previous line, current line)
    num = np.zeros((C, C), np.int64)       # |X(1)| along the best path
    den = np.zeros((C, C), np.int64)       # |X(0)| along the best path
    val[0, :] = -p * popc                  # line 0, with a dead line above it
    den[0, :] = popc

    def produced(cc):
        """Live cells created in the MIDDLE line, for every (prev, cur)."""
        s = cnt3[:, None, :] + cnt2[None, :, :] + cnt3[cc][None, None, :]
        alive = (s == 3) | ((s == 2) & (bits[None, :, :] == 1))
        return alive.sum(-1).astype(np.int64)

    idx = np.arange(C)
    for line in range(n):
        last = (line == n - 1)
        if last:
            # flush: a dead line below the board, then read off the optimum
            v = produced(0)
            tot = val + q * v
            flat = int(tot.argmax())
            a, b = divmod(flat, C)
            return int(tot[a, b]), int(num[a, b] + v[a, b]), int(den[a, b])

        nval = np.full((C, C), NEG, np.int64)
        nnum = np.zeros((C, C), np.int64)
        nden = np.zeros((C, C), np.int64)
        for cc in range(C):
            v = produced(cc)
            tot = val + q * v - p * popc[cc]
            best_prev = tot.argmax(0)                  # best "previous line"
            cur = tot[best_prev, idx]
            upd = cur > nval[:, cc]
            nval[upd, cc] = cur[upd]
            nnum[upd, cc] = (num[best_prev, idx] + v[best_prev, idx])[upd]
            nden[upd, cc] = den[best_prev, idx][upd] + popc[cc]
        val, num, den = nval, nnum, nden


def growth(n, cfg=tp.SQUARE, T=1, max_iter=64, verbose=False):
    """Exact G_1(n) for the square or the cylinder.

    Returns a dict with the optimum as an exact fraction plus the seed.
    """
    cfg = tp.parse(cfg)
    if not supports(cfg, T):
        raise NotImplementedError(
            'the transfer-matrix DP needs a free sweep direction and a gluing '
            'that stays inside a line; %s with T=%d has neither, so its state '
            'would be 2^((2T+1)n) wide. Use the MILP or exhaustive search.'
            % (tp.NAMES[cfg], T))

    t0 = time.time()
    wrap = (cfg == tp.CYLINDER)
    bits, cnt3, cnt2 = _tables(n, wrap)
    popc = np.array([bin(u).count('1') for u in range(1 << n)], np.int64)

    p, q = 1, 1        # start from the stable 2x2 block: ratio 1
    iters = 0
    for iters in range(1, max_iter + 1):
        value, N, D = _maximise_surrogate(n, p, q, bits, cnt3, cnt2, popc)
        if D == 0:
            break
        if value == 0 or N * q == p * D:
            break
        p, q = N, D
        if verbose:
            print('    Dinkelbach %d: %d/%d = %.6f' % (iters, p, q, p / q))

    return {'n': n, 'T': T, 'topology': tp.LETTERS[cfg], 'method': 'dp',
            'rate': p / q, 'numerator': p, 'denominator': q,
            'iterations': iters, 'seconds': round(time.time() - t0, 3),
            'status': 'OPTIMAL'}


def cost_estimate(n, T=1):
    """Rough operation count, for the complexity table in the paper."""
    return n * 2 ** ((2 * T + 1) * n)
