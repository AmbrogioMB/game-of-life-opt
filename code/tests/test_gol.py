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

    for cfg in tp.ALL:
        x = bd.still_life(3, cfg)
        check('n=3 %s has an all-horizon lower bound of 1' % tp.NAMES[cfg],
              x is not None and x != 0
              and all(y == x for y in sim.trajectory(
                  x, sim.neighbour_lists(3, cfg), 3, 5))
              and bd.lower_bound(3, cfg, 5) == 1.0)

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
            r = dp.growth(n, cfg)
            d = r['rate']
            check('n=%d %s  brute=%.6f dp=%.6f' % (n, tp.NAMES[cfg], b, d),
                  abs(b - d) < 1e-9)
            x = sim.decode(n, r['board'])
            g = sim.growth(x, sim.neighbour_lists(n, cfg), n, 1)
            check('n=%d %s  DP seed attains %d/%d'
                  % (n, tp.NAMES[cfg], r['numerator'], r['denominator']),
                  abs(g - d) < 1e-12)

    limited = dp.growth(5, tp.SQUARE, max_iter=1)
    complete = dp.growth(5, tp.SQUARE)
    check('iteration-limited DP does not certify optimality',
          limited['status'] == 'FEASIBLE'
          and limited['rate'] <= complete['rate']
          and complete['status'] == 'OPTIMAL')


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


def test_chunk_coverage():
    print('\n9. a partial chunked enumeration is not reported as an optimum')
    import os
    import tempfile
    import run
    from gol import io_utils as io
    rows = [{'method': 'brute', 'n': 4, 'T': 1, 'topology': 's', 'rate': 1.5,
             'status': 'OPTIMAL', 'chunk': c, 'chunks': 4} for c in range(3)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'r.csv')
        f, w = io.open_writer(path)
        for r in rows:
            w.writerow(r)
        f.close()
        best, incomplete = run.certified_rates(path)
        check('3 of 4 chunks: left out and flagged',
              (4, 1, 's') not in best and incomplete == [((4, 1, 's'), 3, 4)])
        f, w = io.open_writer(path)
        w.writerow(dict(rows[0], chunk=3, rate=1.8))
        f.close()
        best, incomplete = run.certified_rates(path)
        check('4 of 4 chunks: maximum over the chunks reported',
              best.get((4, 1, 's')) == (1.8, 'brute') and not incomplete)


def test_sweep_certificate():
    print('\n10. a sweep is optimal only if every population is settled')
    import argparse
    import types
    import run

    def row(m, status, cells=None, bound=None):
        return {'n': 4, 'T': 1, 'topology': 's', 'm': m, 'status': status,
                'final_cells': cells, 'bound': bound,
                'rate': cells / m if cells is not None else None,
                'seconds': 0.0, 'nodes': 0}

    # n = 4, T = 1: the winner m = 4 (9/4) is optimal, but m = 5 stopped at a
    # bound of 14 cells, and 14/5 > 9/4 leaves it open
    rows = [row(3, 'OPTIMAL', 4), row(4, 'OPTIMAL', 9),
            row(5, 'FEASIBLE', 10, 14.0), row(6, 'INFEASIBLE'),
            row(7, 'OPTIMAL', 14)]
    status, upper, open_ = bd.certify_sweep(4, 1, rows)
    check('optimal winner, unresolved competitor: FEASIBLE',
          status == 'FEASIBLE' and open_ == [5] and float(upper) == 14 / 5)
    closed = rows[:2] + [row(5, 'FEASIBLE', 10, 11.0)] + rows[3:]
    status, _, open_ = bd.certify_sweep(4, 1, closed)
    check('competitor bounded below the winner: OPTIMAL',
          status == 'OPTIMAL' and not open_)
    for bound in (float('inf'), float('-inf'), float('nan')):
        status, upper, open_ = bd.certify_sweep(
            4, 1, rows[:2] + [row(5, 'NO_SOLUTION', bound=bound)] + rows[3:])
        check('non-finite bound %s leaves the population unresolved' % bound,
              status == 'FEASIBLE' and open_ == [5] and upper >= 14 / 5)

    # both summary branches of run.py, with the solvers mocked out
    best = rows[1]
    fake = types.SimpleNamespace(solve=lambda *a, **k: (dict(best), [dict(r) for r in rows]))
    saved = {k: sys.modules.get(k) for k in ('gol.milp', 'gol.cpsat')}
    import gol
    saved_attr = {k: getattr(gol, k, None) for k in ('milp', 'cpsat')}
    try:
        for name in ('milp', 'cpsat'):
            sys.modules['gol.' + name] = fake
            setattr(gol, name, fake)
        args = argparse.Namespace(threads=0, time_limit=0, no_cutoff=False)
        for method in ('milp', 'cpsat'):
            out = list(run.run_task({'method': method, 'n': '4', 'T': '1',
                                     'top': 's'}, args))
            summary = out[-1]
            check('%s summary with an open population is FEASIBLE' % method,
                  summary['method'] == method
                  and summary['status'] == 'FEASIBLE'
                  and abs(summary['bound'] - 14 / 5) < 1e-12)

        # with `emit`, population rows are handed over as they are solved and
        # only the summary is yielded
        def streaming(*a, on_result=None, **k):
            out = [dict(r) for r in rows]
            for r in out:
                on_result(r)
            return dict(best), out
        fake.solve = streaming
        for method in ('milp', 'cpsat'):
            seen = []
            out = list(run.run_task({'method': method, 'n': '4', 'T': '1',
                                     'top': 's'}, args, emit=seen.append))
            check('%s sweep streams its populations before the summary' % method,
                  [r['m'] for r in seen] == [r['m'] for r in rows]
                  and all(r['method'] == method + '-m' and r['task'] for r in seen)
                  and len(out) == 1 and out[0]['method'] == method
                  and out[0]['status'] == 'FEASIBLE')
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in saved_attr.items():
            if v is None:
                if hasattr(gol, k):
                    delattr(gol, k)
            else:
                setattr(gol, k, v)


def test_restart_and_merge():
    print('\n11. interrupted sweeps and retried results')
    import csv
    import tempfile
    from pathlib import Path
    from gol import io_utils as io
    with tempfile.TemporaryDirectory() as directory:
        part = str(Path(directory) / 'part.csv')
        merged = str(Path(directory) / 'merged.csv')
        task = 'milp|n=4|T=1|top=s'
        f, w = io.open_writer(part)
        io.write(f, w, dict(method='milp-m', task=task, m=4,
                             status='OPTIMAL', final_cells=7))
        check('a population checkpoint is not a completed sweep',
              task not in io.done_tasks(part))
        io.write(f, w, dict(method='milp', task=task, m=4,
                             status='FEASIBLE', rate=1.75))
        check('an unresolved summary can be explicitly retried',
              task in io.done_tasks(part)
              and task not in io.done_tasks(part, retry_feasible=True))
        io.write(f, w, dict(method='milp-m', task=task, m=4,
                             status='OPTIMAL', final_cells=8))
        io.write(f, w, dict(method='milp', task=task, m=4,
                             status='OPTIMAL', rate=2.0))
        f.close()
        check('a certified summary is retained on restart',
              task in io.done_tasks(part, retry_feasible=True))
        io.merge([part], merged)
        with open(merged, newline='') as source:
            rows = list(csv.DictReader(source))
        check('merge retains the latest population and summary',
              len(rows) == 2 and rows[0]['final_cells'] == '8'
              and rows[1]['status'] == 'OPTIMAL')
        before = Path(merged).read_bytes()
        try:
            io.merge([str(Path(directory))], merged)
        except OSError:
            pass
        check('a failed merge preserves the previous output',
              Path(merged).read_bytes() == before)


def main():
    test_topology()
    test_projective_degeneracy()
    test_covering_spaces()
    test_known_patterns()
    test_bounds()
    test_symmetry()
    test_dp_vs_brute()
    test_milp_vs_brute()
    test_chunk_coverage()
    test_sweep_certificate()
    test_restart_and_merge()
    print('\n%s' % ('ALL TESTS PASSED' if not FAIL
                    else '%d FAILURES: %s' % (len(FAIL), FAIL)))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
