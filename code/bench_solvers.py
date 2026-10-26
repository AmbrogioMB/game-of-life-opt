"""Gurobi vs CP-SAT on individual fixed-m subproblems.

Paired, bounded, and written incrementally so progress is visible.
"""
import csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gol import cpsat, milp                                   # noqa: E402

TL = 120                       # seconds per subproblem, per solver
CASES = [(6, 1, 's'), (6, 2, 's'), (6, 2, 't'), (6, 3, 's'),
         (7, 1, 's'), (7, 2, 's'), (8, 1, 's')]
MS = [4, 6, 8, 10, 12, 16, 20]

out = 'results_bench_solvers.csv'
new = not os.path.exists(out)
f = open(out, 'a', newline='', encoding='utf-8')
w = csv.writer(f)
if new:
    w.writerow(['n', 'T', 'top', 'm', 'solver', 'status', 'obj', 'bound',
                'seconds', 'nodes'])
    f.flush()

print('%-14s %4s | %-30s | %-30s | ratio'
      % ('n,T,top', 'm', 'Gurobi bigm (1 thread)', 'CP-SAT (1 worker)'),
      flush=True)
for (n, T, top) in CASES:
    for m in MS:
        if m > n * n:
            continue
        row = {}
        for name, fn in (('gurobi', lambda: milp.solve_fixed_m(
                              n, T, top, m, threads=1, time_limit=TL)),
                         ('cpsat', lambda: cpsat.solve_fixed_m(
                              n, T, top, m, threads=1, time_limit=TL))):
            t0 = time.time()
            try:
                r = fn()
            except Exception as e:
                r = {'status': 'ERROR:%s' % e, 'final_cells': None,
                     'bound': None, 'seconds': time.time() - t0, 'nodes': 0}
            row[name] = r
            w.writerow([n, T, top, m, name, r['status'], r['final_cells'],
                        r.get('bound'), r['seconds'], r.get('nodes', 0)])
            f.flush()
        g, c = row['gurobi'], row['cpsat']
        fmt = lambda r: '%8.1f s %-9s obj=%-5s' % (
            r['seconds'], r['status'][:9],
            r['final_cells'] if r['final_cells'] is not None else '--')
        ratio = ('x%.1f' % (g['seconds'] / c['seconds'])
                 if c['seconds'] > 0.001 else '-')
        print('%-14s %4d | %-30s | %-30s | %s'
              % (str((n, T, top)), m, fmt(g), fmt(c), ratio), flush=True)
f.close()
print('\nwritten to', out)
