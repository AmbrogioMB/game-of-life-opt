"""Valid bounds on the optimal growth rate.

Upper bounds
------------
(1)  one step:    |X(t+1)| <= 3 |X(t)|  for every configuration, hence
     |X(T)| <= 3^T |X(0)|.  (Lemma "upper" in the paper; it bounds an arbitrary
     configuration, so it iterates.)

(2)  light cone:  a cell is born only next to a live cell, so
     supp X(t+1) is contained in supp X(t) dilated by one step.  After T steps
     the support of a seed of size m is covered by m translates of the
     (2T+1)x(2T+1) box, giving  |X(T)| <= m (2T+1)^2, i.e. r <= (2T+1)^2.
     This is polynomial in T and beats 3^T from T >= 5 on.

(3)  board size:  |X(T)| <= n^2 and any seed that produces anything has at
     least 3 live cells, so r <= n^2/3.

Lower bound
-----------
A 2x2 block is stable on every one of the six surfaces for every n >= 4 (and on
the square, cylinder, torus and Klein bottle already for n = 3), so r >= 1.
This is verified, not assumed -- see `block_lower_bound`.
"""
from . import simulate as sim
from . import topology as tp

MIN_SEED = 3   # fewer than 3 live cells can never produce a live cell


def upper_bound(n, T, which='all'):
    """The tightest valid upper bound on G_T(n)."""
    cands = {'triple': 3.0 ** T,
             'lightcone': float((2 * T + 1) ** 2),
             'board': n * n / float(MIN_SEED)}
    if which == 'all':
        return min(cands.values())
    return cands[which]


def upper_bound_breakdown(n, T):
    return {'triple': 3.0 ** T,
            'lightcone': float((2 * T + 1) ** 2),
            'board': n * n / float(MIN_SEED),
            'best': upper_bound(n, T)}


def max_final_cells(n, T, m):
    """Upper bound on |X(T)| given |X(0)| = m.  Used to prune the m-loop."""
    return min(n * n, int(3 ** T * m), (2 * T + 1) ** 2 * m)


def block_lower_bound(n, cfg, corner_policy='simple'):
    """Return a stable 2x2 block as a bitmask, or None if none exists."""
    cfg = tp.parse(cfg)
    nbl = sim.neighbour_lists(n, cfg, corner_policy)
    for oi in range(n - 1):
        for oj in range(n - 1):
            x = sim.from_cells(n, [(oi, oj), (oi, oj + 1),
                                   (oi + 1, oj), (oi + 1, oj + 1)])
            if sim.step(x, nbl, n) == x:
                return x
    return None


def lower_bound(n, cfg, T, corner_policy='simple'):
    """A certified lower bound on G_T(n): 1.0 when a stable block exists."""
    return 1.0 if block_lower_bound(n, cfg, corner_policy) is not None else 0.0


def stripe_bound(n, cfg, T=1, corner_policy='simple'):
    """Best growth over all SINGLE full rows and columns.

    On the cylinder, torus and Klein bottle the vertical seam is direct, a
    single row closes up on itself, and this gives exactly 3.  On the Moebius
    band and the projective plane the seam reflects the row index, so a single
    row works only when it is fixed by that reflection, i.e. only for odd n;
    for even n the best single row gives 3 - 2/n.

    That parity dependence is a property of THIS construction, not of the
    optimum: see `row_set_bound`, which searches all reflection-invariant sets
    of rows and recovers 3 for every even n >= 6 via the pair {1, n-2}.
    """
    cfg = tp.parse(cfg)
    nbl = sim.neighbour_lists(n, cfg, corner_policy)
    best, arg = 0.0, None
    for k in range(n):
        for cells in ([(k, j) for j in range(n)], [(i, k) for i in range(n)]):
            x = sim.from_cells(n, cells)
            g = sim.growth(x, nbl, n, T)
            if g > best:
                best, arg = g, x
    return best, arg


def family_seed(k):
    """The block family from the draft's remark, on the (3k+2)x(3k+2) square.

        X^k = [[a a ... a b]
               [a a ... a b]
               [...        ]
               [a a ... a b]
               [c c ... c d]]

    with a 3x3, b 3x2, c 2x3, d 2x2 as in the draft.  Returned as a bitmask so
    that |X| and |X(1)| can be measured rather than taken from the formula.
    """
    if k < 1:
        raise ValueError('k must be at least 1')
    n = 3 * k + 2
    a = [(0, 0), (0, 1), (1, 0)]          # 3x3 block
    b = [(0, 0), (1, 0), (1, 1)]          # 3x2 block
    c = [(0, 0), (0, 1), (1, 1)]          # 2x3 block
    d = [(0, 0)]                          # 2x2 block
    cells = []
    for br in range(k):
        for bc in range(k):
            cells += [(3 * br + i, 3 * bc + j) for (i, j) in a]
        cells += [(3 * br + i, 3 * k + j) for (i, j) in b]
    for bc in range(k):
        cells += [(3 * k + i, 3 * bc + j) for (i, j) in c]
    cells += [(3 * k + i, 3 * k + j) for (i, j) in d]
    return n, sim.from_cells(n, cells)


def family_rate(k, cfg=tp.SQUARE):
    """(n, |X|, |X(1)|, rate) for the family, measured by simulation."""
    n, x = family_seed(k)
    nbl = sim.neighbour_lists(n, tp.parse(cfg))
    y = sim.step(x, nbl, n)
    m, f = sim.popcount(x), sim.popcount(y)
    return n, m, f, f / m


def family_formula(k):
    """The draft's closed form, for comparison with `family_rate`."""
    return ((3 * k + 2) ** 2 - 4) / (3 * (k + 1) ** 2 - 2)


def row_set_bound(n, cfg, T=1, corner_policy='simple', max_rows=None):
    """Best growth over all configurations that are a union of full rows/columns.

    This is what actually attains the bound 3 on the two surfaces with a
    flipped vertical seam.  A set R of live rows closes up under the gluing iff
    it is invariant under the seam reflection rho(i) = n-1-i, and it achieves
    3|R|n iff in addition the triples {r-1,r,r+1} are pairwise disjoint.  For
    odd n the singleton {(n-1)/2} works; for even n >= 6 the pair {1, n-2}
    does; for n = 4 neither exists, which is why that board is exceptional.

    Returns (best_rate, best_board_bitmask, best_row_set).
    """
    import itertools
    cfg = tp.parse(cfg)
    nbl = sim.neighbour_lists(n, cfg, corner_policy)
    best, arg, rows_arg = 0.0, None, None
    top = n if max_rows is None else min(n, max_rows)
    for k in range(1, top + 1):
        for rows in itertools.combinations(range(n), k):
            for horizontal in (True, False):
                cells = ([(r, j) for r in rows for j in range(n)] if horizontal
                         else [(j, r) for r in rows for j in range(n)])
                x = sim.from_cells(n, cells)
                g = sim.growth(x, nbl, n, T)
                if g > best:
                    best, arg, rows_arg = g, x, (rows, 'rows' if horizontal
                                                 else 'cols')
    return best, arg, rows_arg
