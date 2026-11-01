"""CP-SAT model for the maximum growth rate.

Why a second solver
-------------------
The LP bound of the integer program is weak: on the 190 subproblems of the
paper's formulation comparison it exceeds the integer optimum by 46% at the
median, 91% on average and 733% at worst, and neither the valid cuts nor the
unary encoding of the neighbour count change that materially.  CP-SAT combines
clause learning with propagation and its own linear relaxations, so it attacks
the same model in a different way; the paper compares the two empirically
rather than claiming a reason why either should win.

The model is the same one, stated natively:

    x[i,j,t+1]  <->  (s == 3)  or  ((s == 2) and x[i,j,t])

with s the live-neighbour count, expressed with reified equalities instead of
big-M constants.
"""
import time

from ortools.sat.python import cp_model

from . import bounds as bd
from . import simulate as sim
from . import symmetry as sy
from . import topology as tp


def _build(n, T, cfg, corner_policy, symmetry):
    cfg = tp.parse(cfg)
    adj = tp.adjacency(n, cfg, corner_policy)
    m = cp_model.CpModel()

    x = {(i, j, t): m.NewBoolVar('x_%d_%d_%d' % (i, j, t))
         for i in range(n) for j in range(n) for t in range(T + 1)}

    for t in range(T):
        for (i, j), nb in adj.items():
            s = sum(x[k, l, t] for (k, l) in nb)
            cur, nxt = x[i, j, t], x[i, j, t + 1]

            eq3 = m.NewBoolVar('')
            m.Add(s == 3).OnlyEnforceIf(eq3)
            m.Add(s != 3).OnlyEnforceIf(eq3.Not())

            eq2 = m.NewBoolVar('')
            m.Add(s == 2).OnlyEnforceIf(eq2)
            m.Add(s != 2).OnlyEnforceIf(eq2.Not())

            surv = m.NewBoolVar('')                    # surv <-> eq2 and cur
            m.AddBoolAnd([eq2, cur]).OnlyEnforceIf(surv)
            m.AddBoolOr([eq2.Not(), cur.Not()]).OnlyEnforceIf(surv.Not())

            m.AddImplication(eq3, nxt)                 # nxt <-> eq3 or surv
            m.AddImplication(surv, nxt)
            m.AddBoolOr([eq3, surv, nxt.Not()])

    if symmetry:
        for coef in sy.breaking_inequalities(n, cfg, corner_policy):
            m.Add(sum(int(round(v)) * x[c[0], c[1], 0]
                      for c, v in coef.items()) >= 0)
    return m, x


def solve_fixed_m(n, T, cfg, m_cells, symmetry=True, corner_policy='simple',
                  threads=0, time_limit=0, log=False):
    """max |X(T)| subject to |X(0)| = m_cells."""
    t0 = time.time()
    model, x = _build(n, T, cfg, corner_policy, symmetry)
    board = [(i, j) for i in range(n) for j in range(n)]

    model.Add(sum(x[i, j, 0] for (i, j) in board) == m_cells)
    final = sum(x[i, j, T] for (i, j) in board)
    model.Add(final <= bd.max_final_cells(n, T, m_cells))
    model.Maximize(final)

    solver = cp_model.CpSolver()
    if threads:
        solver.parameters.num_search_workers = threads
    if time_limit:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.log_search_progress = bool(log)
    st = solver.Solve(model)

    res = {'n': n, 'T': T, 'topology': tp.LETTERS[tp.parse(cfg)], 'm': m_cells,
           'formulation': 'cpsat', 'symmetry': int(bool(symmetry)),
           'seconds': round(time.time() - t0, 3),
           'nodes': int(solver.NumBranches()),
           'final_cells': None, 'rate': None, 'bound': None, 'gap': None,
           'board': None, 'status': None}

    if st == cp_model.INFEASIBLE:
        res['status'] = 'INFEASIBLE'
        return res
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res['status'] = 'NO_SOLUTION'
        return res

    cells = int(round(solver.ObjectiveValue()))
    seed = sim.from_cells(n, [(i, j) for (i, j) in board
                              if solver.Value(x[i, j, 0])])
    res.update(status='OPTIMAL' if st == cp_model.OPTIMAL else 'FEASIBLE',
               final_cells=cells, rate=cells / m_cells,
               bound=float(solver.BestObjectiveBound()),
               board=sim.encode(n, seed))
    return res


def solve(n, T, cfg, symmetry=True, corner_policy='simple', threads=0,
          time_limit=0, log=False, on_result=None):
    """G_T(n) by the same pruned sweep over m as the integer program."""
    cfg = tp.parse(cfg)
    N = n * n
    best, rows = None, []
    for mc in range(bd.MIN_SEED, N + 1):
        if best is not None and N / mc <= best['rate'] + 1e-12:
            break
        r = solve_fixed_m(n, T, cfg, mc, symmetry, corner_policy,
                          threads, time_limit, log)
        rows.append(r)
        if on_result:
            on_result(r)
        if r['rate'] is not None and (best is None or r['rate'] > best['rate']):
            best = r
    return best, rows
