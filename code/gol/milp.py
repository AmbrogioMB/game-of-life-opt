"""Gurobi models for the maximum growth rate.

Two ways of handling the ratio |X(T)|/|X(0)|:

  'fixed-m'  pin |X(0)| = m and maximise |X(T)| -- a pure integer objective, no
             linearisation, no big-M on the ratio, no tolerance artefacts.
             G_T(n) = max_m f(m)/m, and the m-loop is pruned by f(m) <= n^2.
             This is the default and it is what the reported numbers use.

  'ratio'    the McCormick linearisation of r*x_{ij,0} used in the draft, but
             with the envelope built on the tightened bounds from `bounds`
             (the min of 3^T, (2T+1)^2 and n^2/3) instead of 3^T.  Kept so the
             two can be benchmarked against each other.

Two ways of encoding one Game of Life step:

  'bigm'     the draft's encoding: two indicator binaries y,z per cell and time
             with big-M constants of the size of the neighbourhood, plus two
             valid cuts.

  'unary'    an exact encoding of the neighbour count.  With
             sum_k sigma_k = 1 and sum_k k*sigma_k = s the vector sigma is the
             indicator of s, and the rule becomes four inequalities

                 x' >= sigma_3
                 x' >= sigma_2 + x - 1
                 x' <= sigma_3 + sigma_2
                 x' <= sigma_3 + x

             More binaries, no big-M constants.  Measured on small instances
             its LP bound equals that of 'bigm' WITH the two cuts below, so it
             is not tighter in practice -- see the remark in the paper.
"""
import time

import gurobipy as gp
from gurobipy import GRB

from . import bounds as bd
from . import simulate as sim
from . import symmetry as sy
from . import topology as tp


def _add_dynamics(model, x, n, T, adj, formulation, cuts=True):
    """Constrain x[i,j,t+1] to be the Game of Life successor of x[.,.,t]."""
    for t in range(T):
        for (i, j), nb in adj.items():
            s = gp.quicksum(x[k, l, t] for (k, l) in nb)
            M = len(nb)
            cur, nxt = x[i, j, t], x[i, j, t + 1]

            if formulation == 'bigm':
                y = model.addVar(vtype=GRB.BINARY, name='y_%d_%d_%d' % (i, j, t))
                z = model.addVar(vtype=GRB.BINARY, name='z_%d_%d_%d' % (i, j, t))
                # alive next  =>  2 <= s <= 3, with s = 2 only if alive now
                model.addConstr(s + cur >= 3 * nxt, 'A_%d_%d_%d' % (i, j, t))
                model.addConstr(s <= 3 + (M - 3) * (1 - nxt),
                                'B_%d_%d_%d' % (i, j, t))
                # dead next  =>  overcrowded (y) or starved (z)
                model.addConstr(s >= 4 * y, 'C_%d_%d_%d' % (i, j, t))
                model.addConstr(s + cur <= 2 + (M - 1) * (1 - z),
                                'D_%d_%d_%d' % (i, j, t))
                model.addConstr(1 - nxt <= y + z, 'E_%d_%d_%d' % (i, j, t))
                if cuts:
                    # valid cut: cannot be overcrowded and starved at once
                    model.addConstr(y + z <= 1, 'F_%d_%d_%d' % (i, j, t))
                    # valid cut: implied integrally by A and B, occasionally tightens the LP
                    model.addConstr(s >= 2 * nxt, 'G_%d_%d_%d' % (i, j, t))

            elif formulation == 'unary':
                sig = model.addVars(range(M + 1), vtype=GRB.BINARY,
                                    name='sig_%d_%d_%d' % (i, j, t))
                model.addConstr(gp.quicksum(sig[k] for k in range(M + 1)) == 1)
                model.addConstr(gp.quicksum(k * sig[k]
                                            for k in range(M + 1)) == s)
                model.addConstr(nxt >= sig[3])
                model.addConstr(nxt >= sig[2] + cur - 1)
                model.addConstr(nxt <= sig[3] + sig[2])
                model.addConstr(nxt <= sig[3] + cur)
            else:
                raise ValueError('unknown formulation %r' % (formulation,))


def _base_model(n, T, cfg, formulation, symmetry, corner_policy,
                threads, time_limit, mipgap, log, cuts=True):
    cfg = tp.parse(cfg)
    adj = tp.adjacency(n, cfg, corner_policy)

    model = gp.Model()
    model.Params.OutputFlag = 1 if log else 0
    if threads:
        model.Params.Threads = threads
    if time_limit:
        model.Params.TimeLimit = time_limit
    if mipgap is not None:
        model.Params.MIPGap = mipgap

    idx = [(i, j, t) for i in range(n) for j in range(n) for t in range(T + 1)]
    x = model.addVars(idx, vtype=GRB.BINARY, name='x')
    _add_dynamics(model, x, n, T, adj, formulation, cuts)

    if symmetry:
        ineqs = sy.breaking_inequalities(n, cfg, corner_policy)
        for q, coef in enumerate(ineqs):
            model.addConstr(
                gp.quicksum(v * x[i, j, 0] for (i, j), v in coef.items()) >= 0,
                'sym_%d' % q)
    return model, x, adj


def _blank(n, T, cfg, m, formulation, symmetry, cuts=True):
    return {'n': n, 'T': T, 'topology': tp.LETTERS[tp.parse(cfg)], 'm': m,
            'formulation': formulation, 'symmetry': int(bool(symmetry)),
            'cuts': int(bool(cuts)),
            'status': None, 'seconds': None, 'nodes': 0, 'final_cells': None,
            'rate': None, 'bound': None, 'gap': None, 'board': None}


# ------------------------------------------------------------------- fixed m

def solve_fixed_m(n, T, cfg, m, formulation='bigm', symmetry=True,
                  corner_policy='simple', threads=0, time_limit=0,
                  mipgap=None, cutoff_cells=None, log=False, cuts=True,
                  relax=False):
    """max |X(T)| subject to |X(0)| = m.  Returns a result dict.

    With relax=True the LP relaxation is solved instead, which is the root
    bound the formulation comparison reports.
    """
    t0 = time.time()
    model, x, _ = _base_model(n, T, cfg, formulation, symmetry, corner_policy,
                              threads, time_limit, mipgap, log, cuts)
    model.addConstr(gp.quicksum(x[i, j, 0]
                                for i in range(n) for j in range(n)) == m)
    final = gp.quicksum(x[i, j, T] for i in range(n) for j in range(n))
    model.setObjective(final, GRB.MAXIMIZE)
    model.addConstr(final <= bd.max_final_cells(n, T, m))
    if cutoff_cells is not None:
        model.addConstr(final >= cutoff_cells)

    if relax:
        model.update()          # relax() copies the model as last updated
        model = model.relax()
    model.optimize()
    res = _blank(n, T, cfg, m,
                 formulation + ('+lp' if relax else ''), symmetry, cuts)
    res['seconds'] = round(time.time() - t0, 3)
    res['nodes'] = int(getattr(model, 'NodeCount', 0))

    if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        res['status'] = 'INFEASIBLE'
        return res
    if model.SolCount == 0:
        res['status'] = 'NO_SOLUTION'
        res['bound'] = float(getattr(model, 'ObjBound', float('nan')))
        return res

    if relax:
        res.update(status='OPTIMAL', final_cells=None,
                   rate=float(model.ObjVal) / m, bound=float(model.ObjVal),
                   gap=None)
        return res

    cells = int(round(model.ObjVal))
    seed = sim.from_cells(n, [(i, j) for i in range(n) for j in range(n)
                              if round(x[i, j, 0].X) == 1])
    res.update(status='OPTIMAL' if model.Status == GRB.OPTIMAL else 'FEASIBLE',
               final_cells=cells, rate=cells / m,
               bound=float(model.ObjBound),
               gap=float(getattr(model, 'MIPGap', 0.0)),
               board=sim.encode(n, seed))
    return res


def solve(n, T, cfg, formulation='bigm', symmetry=True, corner_policy='simple',
          threads=0, time_limit=0, mipgap=None, use_cutoff=True, log=False,
          on_result=None, cuts=True):
    """G_T(n) by sweeping m, with pruning.  Returns (best, [all results]).

    The sweep runs m = 3, 4, ... and stops as soon as n^2/m can no longer beat
    the incumbent, since |X(T)| <= n^2 always.  With `use_cutoff` each
    subproblem is additionally asked only for a strict improvement, which turns
    most of them into quick infeasibility proofs.
    """
    cfg = tp.parse(cfg)
    N = n * n
    best, rows = None, []
    for m in range(bd.MIN_SEED, N + 1):
        if best is not None and N / m <= best['rate'] + 1e-12:
            break
        cutoff = None
        if use_cutoff and best is not None:
            need = int(best['rate'] * m) + 1
            if need > bd.max_final_cells(n, T, m):
                continue
            cutoff = need
        r = solve_fixed_m(n, T, cfg, m, formulation, symmetry, corner_policy,
                          threads, time_limit, mipgap, cutoff, log, cuts)
        rows.append(r)
        if on_result:
            on_result(r)
        if r['rate'] is not None and (best is None or r['rate'] > best['rate']):
            best = r
    return best, rows


# --------------------------------------------------------------------- ratio

def solve_ratio(n, T, cfg, formulation='bigm', symmetry=True,
                corner_policy='simple', threads=0, time_limit=0, mipgap=None,
                log=False, relax=False, cuts=True):
    """The draft's ratio model, with the McCormick envelope on tight bounds.

    With relax=True it returns the LP relaxation value, which is what the
    formulation-strength comparison in the paper needs.
    """
    t0 = time.time()
    cfg = tp.parse(cfg)
    lb = bd.lower_bound(n, cfg, T, corner_policy)
    ub = bd.upper_bound(n, T)
    model, x, _ = _base_model(n, T, cfg, formulation, symmetry, corner_policy,
                              threads, time_limit, mipgap, log, cuts)

    board = [(i, j) for i in range(n) for j in range(n)]
    r = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name='r')
    g = model.addVars(board, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='g')

    # g_ij = r * x_ij0, McCormick over r in [lb, ub] and x in {0,1}
    for (i, j) in board:
        model.addConstr(g[i, j] >= lb * x[i, j, 0])
        model.addConstr(g[i, j] >= ub * x[i, j, 0] + r - ub)
        model.addConstr(g[i, j] <= lb * x[i, j, 0] + r - lb)
        model.addConstr(g[i, j] <= ub * x[i, j, 0])
    model.addConstr(gp.quicksum(g[i, j] for (i, j) in board)
                    == gp.quicksum(x[i, j, T] for (i, j) in board))
    model.addConstr(gp.quicksum(x[i, j, 0] for (i, j) in board) >= bd.MIN_SEED)
    model.setObjective(r, GRB.MAXIMIZE)

    if relax:
        model.update()          # relax() copies the model as last updated
        model = model.relax()
    model.optimize()

    out = _blank(n, T, cfg, None, formulation + ('+lp' if relax else ''),
                 symmetry, cuts)
    out['seconds'] = round(time.time() - t0, 3)
    out['nodes'] = int(getattr(model, 'NodeCount', 0))
    out['lb_used'] = lb
    out['ub_used'] = ub
    if model.SolCount == 0 and model.Status != GRB.OPTIMAL:
        out['status'] = 'NO_SOLUTION'
        return out
    out.update(status='OPTIMAL' if model.Status == GRB.OPTIMAL else 'FEASIBLE',
               rate=float(model.ObjVal),
               bound=float(getattr(model, 'ObjBound', model.ObjVal)),
               gap=float(getattr(model, 'MIPGap', 0.0)))
    if not relax:
        seed = sim.from_cells(n, [(i, j) for (i, j) in board
                                  if round(x[i, j, 0].X) == 1])
        out['board'] = sim.encode(n, seed)
        out['final_cells'] = int(round(sum(x[i, j, T].X for (i, j) in board)))
    return out
