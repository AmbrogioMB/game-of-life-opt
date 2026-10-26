"""Validation suite.  Run:  python -m tests.test_gol   (from the project root)

Nothing here needs Gurobi except the MILP cross-check, which is skipped if
gurobipy is unavailable.
"""
import itertools
import sys

from gol import bounds as bd
from gol import brute
from gol import dp
from gol import simulate as sim
from gol import symmetry as sy
from gol import topology as tp

FAIL = []


def check(name, cond, detail=''):
    mark = 'ok  ' if cond else 'FAIL'
    if not cond:
        FAIL.append(name)
    print('  %s %s%s' % (mark, name, ('   ' + detail) if detail else ''))


# ------------------------------------------------------------ 1. topologies

def test_topology():
    print('\n1. topology invariants (symmetry, no loops, no repeats, degrees)')
    for n in range(3, 11):
        for cfg in tp.ALL:
            r = tp.check(n, cfg)
            check('n=%d %s' % (n, tp.NAMES[cfg]), not r['problems'],
                  '; '.join(r['problems']))


def test_projective_degeneracy():
    print('\n2. projective plane corner degeneracy is real, not a typo')
    for n in range(3, 9):
        raw = tp.neighbours(n, tp.PROJECTIVE, 0, 0, 'raw')
        check('n=%d corner folds onto itself' % n, (0, 0) in raw)
        check('n=%d corner has a double edge' % n,
              len(set(c for c in raw if c != (0, 0))) == 6)


# -------------------------------------------------------- 3. covering spaces

def _glide_cover_mobius(n, x):
    """Lift a Moebius board to its cylinder double cover (n rows, 2n columns)."""
    cells = sim.to_cells(n, x)
    out = set(cells)
    for (i, j) in cells:
        out.add((n - 1 - i, j + n))
    return out


def test_covering_spaces():
    """A Moebius step must agree with a cylinder step on the double cover."""
    print('\n3. covering-space consistency')
    import random
    rng = random.Random(20260922)
    for n in (4, 5, 6, 7):
        nb_m = sim.neighbour_lists(n, tp.MOBIUS)
        adj_c = tp.adjacency(2 * n, tp.CYLINDER)   # 2n x 2n, we use n rows only
        ok = True
        for _ in range(40):
            x = rng.getrandbits(n * n)
            lifted = _glide_cover_mobius(n, x)
            ymob = sim.to_cells(n, sim.step(x, nb_m, n))
            # step the lift by hand on the n x 2n cylinder
            ylift = set()
            for i in range(n):
                for j in range(2 * n):
                    s = 0
                    for (p, q) in adj_c[(i, j)]:
                        if p < n and (p, q % (2 * n)) in lifted:
                            s += 1
                    if s == 3 or (s == 2 and (i, j) in lifted):
                        ylift.add((i, j))
            if set((i, j) for (i, j) in ylift if j < n) != set(ymob):
                ok = False
                break
        check('n=%d Moebius agrees with its cylinder double cover' % n, ok)


# ------------------------------------------------------------- 4. known life

def test_known_patterns():
    print('\n4. textbook patterns on the plain square')
    n = 6
    nbl = sim.neighbour_lists(n, tp.SQUARE)
    block = sim.from_cells(n, [(1, 1), (1, 2), (2, 1), (2, 2)])
    check('block is still life', sim.step(block, nbl, n) == block)

    blinker = sim.from_cells(n, [(2, 1), (2, 2), (2, 3)])
    b1 = sim.step(blinker, nbl, n)
    check('blinker has period 2', sim.step(b1, nbl, n) == blinker
          and b1 != blinker)

    glider = sim.from_cells(n, [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)])
    g4 = sim.evolve(glider, nbl, n, 4)
    shifted = sim.from_cells(n, [(i + 1, j + 1) for (i, j)
                                 in sim.to_cells(n, glider)])
    check('glider translates by (1,1) in 4 steps', g4 == shifted)

    check('encode/decode round trip',
          all(sim.decode(n, sim.encode(n, v)) == v
              for v in (0, block, blinker, glider, (1 << (n * n)) - 1)))


# ------------------------------------------------------------ 5. the bounds

def test_bounds():
    print('\n5. bounds')
    check('3^T beats the light cone for T<=4, loses from T=5',
          all(bd.upper_bound(99, T, 'triple')
              <= bd.upper_bound(99, T, 'lightcone') for T in (1, 2, 3, 4))
          and bd.upper_bound(99, 5, 'triple')
          > bd.upper_bound(99, 5, 'lightcone'))

    ok = all(bd.block_lower_bound(n, cfg) is not None
             for n in range(4, 13) for cfg in tp.ALL)
    check('2x2 block stable on all six surfaces for n>=4', ok)

    odd = all(abs(bd.stripe_bound(n, tp.MOBIUS)[0] - 3.0) < 1e-12
              for n in (3, 5, 7, 9, 11, 13))
    even = all(abs(bd.stripe_bound(n, tp.MOBIUS)[0] - (3 - 2 / n)) < 1e-12
               for n in (4, 6, 8, 10, 12))
    check('single row on Moebius: 3 for odd n', odd)
    check('single row on Moebius: only 3-2/n for even n', even)

    # the parity is a property of the single-row construction, NOT of the
    # optimum: a reflection-invariant PAIR of rows recovers 3 for even n >= 6
    for cfg in (tp.MOBIUS, tp.PROJECTIVE):
        pair = all(abs(bd.row_set_bound(n, cfg, max_rows=2)[0] - 3.0) < 1e-12
                   for n in (6, 8, 10, 12))
        check('row pair {1,n-2} on %s: 3 for even n>=6' % tp.NAMES[cfg], pair)
        four = abs(bd.row_set_bound(4, cfg, max_rows=4)[0] - 2.5) < 1e-12
        check('no set of full rows reaches 3 on %s at n=4' % tp.NAMES[cfg],
              four)

    three = all(abs(bd.stripe_bound(n, cfg)[0] - 3.0) < 1e-12
                for n in range(3, 12)
                for cfg in (tp.CYLINDER, tp.TORUS, tp.KLEIN))
    check('stripe gives exactly 3 on cylinder, torus, Klein', three)


# ------------------------------------------------------------- 6. symmetry

def test_symmetry():
    print('\n6. symmetry breaking keeps at least one optimum')
    for n in (3, 4):        # full enumeration in pure Python, keep it small
        for cfg in tp.ALL:
            nbl = sim.neighbour_lists(n, cfg)
            ineqs = sy.breaking_inequalities(n, cfg)
            # the best rate over boards surviving the break must equal the
            # best rate over all boards
            best_all, best_kept = 0.0, 0.0
            for x in range(1 << (n * n)):
                if sim.popcount(x) < bd.MIN_SEED:
                    continue
                g = sim.growth(x, nbl, n, 1)
                if g > best_all:
                    best_all = g
                if g > best_kept:
                    cells = sim.to_cells(n, x)
                    if all(sum(v for c, v in coef.items() if c in set(cells))
                           >= -1e-9 for coef in ineqs):
                        best_kept = g
            check('n=%d %s optimum survives' % (n, tp.NAMES[cfg]),
                  abs(best_all - best_kept) < 1e-9,
                  'all=%.4f kept=%.4f' % (best_all, best_kept))


# ------------------------------------------- 7. methods agree with each other

def test_dp_vs_brute():
    print('\n7. transfer-matrix DP vs exhaustive search (T=1)')
    for cfg in (tp.SQUARE, tp.CYLINDER):
        for n in (3, 4, 5):
            b = brute.search(n, 1, cfg)['rate']
            d = dp.growth(n, cfg)['rate']
            check('n=%d %s  brute=%.6f dp=%.6f' % (n, tp.NAMES[cfg], b, d),
                  abs(b - d) < 1e-9)


def test_milp_vs_brute():
    print('\n8. MILP vs exhaustive search (all six surfaces)')
    try:
        from gol import milp
    except Exception as e:                                    # no licence, etc.
        print('  skipped (gurobipy unavailable: %s)' % e)
        return
    for n in (3, 4):
        for T in (1, 2):
            for cfg in tp.ALL:
                b = brute.search(n, T, cfg)['rate']
                for form in ('bigm', 'unary'):
                    best, _ = milp.solve(n, T, cfg, formulation=form)
                    r = best['rate'] if best else 0.0
                    check('n=%d T=%d %s [%s] brute=%.6f milp=%.6f'
                          % (n, T, tp.LETTERS[cfg], form, b, r),
                          abs(b - r) < 1e-6)


def main():
    test_topology()
    test_projective_degeneracy()
    test_covering_spaces()
    test_known_patterns()
    test_bounds()
    test_symmetry()
    test_dp_vs_brute()
    test_milp_vs_brute()
    print('\n%s' % ('ALL TESTS PASSED' if not FAIL
                    else '%d FAILURES: %s' % (len(FAIL), FAIL)))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
