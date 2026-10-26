#!/usr/bin/env python3
"""Command line front end.

    python run.py check                      structural self-check, no solver
    python run.py tasks   --out FILE ...     write a task list
    python run.py worker  --tasks FILE --shard K --of N --out FILE
    python run.py solve   --method M ...     run one thing directly
    python run.py merge   --out FILE parts   concatenate part files
    python run.py table   --csv FILE         LaTeX table for the paper
    python run.py analyse --csv FILE         convergence, family optimality
    python run.py duel    --csv FILE         Gurobi against CP-SAT

A task is one line of the task file, e.g.

    milp  n=7 T=2 top=s form=bigm sym=1
    dp    n=11 T=1 top=s
    brute n=5 T=3 top=p chunk=0/8

Workers take every N-th task, append results as they go, and skip tasks already
present in their output file, so a job killed by the wall clock can simply be
resubmitted.
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gol import bounds as bd            # noqa: E402
from gol import brute                   # noqa: E402
from gol import dp                      # noqa: E402
from gol import io_utils as io          # noqa: E402
from gol import simulate as sim         # noqa: E402
from gol import topology as tp          # noqa: E402


# Methods whose 'rate' column is a certified optimal growth rate.  Everything
# else -- per-m subproblems, LP relaxations, duel entries, time-limited runs --
# reports something that is not G_T(n) and must be excluded from the tables.
PROVEN_METHODS = {'dp', 'brute', 'milp', 'cpsat'}


# ------------------------------------------------------------------- tasks

def parse_task(line):
    parts = line.split()
    spec = {'method': parts[0]}
    for kv in parts[1:]:
        k, _, v = kv.partition('=')
        spec[k] = v
    return spec


def task_id(spec):
    keys = ('n', 'T', 'top', 'form', 'sym', 'cuts', 'lp', 'chunk', 'm', 'ms')
    return '%s|%s' % (spec['method'],
                      '|'.join('%s=%s' % (k, spec[k])
                               for k in keys if k in spec))


def cmd_tasks(args):
    lines = []
    tops = args.topologies
    for n in args.n:
        for T in args.T:
            for top in tops:
                if args.method == 'cpsat':
                    lines.append('cpsat n=%d T=%d top=%s sym=%d'
                                 % (n, T, top, int(args.symmetry)))
                elif args.method == 'duel':
                    for mm in args.ms:
                        if mm <= n * n:
                            lines.append('duel n=%d T=%d top=%s m=%d'
                                         % (n, T, top, mm))
                elif args.method in ('milp', 'root'):
                    for form in args.formulations:
                        for cuts in args.cuts:
                            lines.append(
                                '%s n=%d T=%d top=%s form=%s sym=%d cuts=%d%s'
                                % (args.method, n, T, top, form,
                                   int(args.symmetry), cuts,
                                   (' ms=' + '.'.join(str(v) for v in args.ms))
                                   if args.method == 'root' else ''))
                elif args.method == 'dp':
                    if dp.supports(top, T):
                        lines.append('dp n=%d T=%d top=%s' % (n, T, top))
                elif args.method == 'brute':
                    for c in range(args.chunks):
                        lines.append('brute n=%d T=%d top=%s chunk=%d/%d'
                                     % (n, T, top, c, args.chunks))
    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('%d tasks -> %s' % (len(lines), args.out))


# ------------------------------------------------------------------ running

def run_task(spec, args):
    """Yield result rows for one task."""
    method = spec['method']
    n = int(spec['n'])
    T = int(spec['T'])
    top = spec.get('top', 's')
    tid = task_id(spec)

    if method == 'dp':
        r = dp.growth(n, top, T)
        r['task'] = tid
        yield r

    elif method == 'brute':
        chunk, chunks = 0, 1
        if 'chunk' in spec:
            a, _, b = spec['chunk'].partition('/')
            chunk, chunks = int(a), int(b or 1)

        # a long task otherwise prints nothing until it finishes, which makes
        # "is it stuck or just slow?" impossible to answer from the log
        import time as _t
        state = {'last': 0.0}

        def _progress(done, total, best):
            now = _t.time()
            if now - state['last'] < 120:      # at most one line every 2 min
                return
            state['last'] = now
            print('      %s  %5.1f%%  best so far %.4f'
                  % (tid, 100.0 * done / total, best), flush=True)

        r = brute.search(n, T, top, chunk=chunk, chunks=chunks,
                         progress=_progress)
        r['task'] = tid
        yield r

    elif method == 'milp':
        from gol import milp
        form = spec.get('form', 'bigm')
        symm = bool(int(spec.get('sym', 1)))
        cuts = bool(int(spec.get('cuts', 1)))
        best, rows = milp.solve(n, T, top, formulation=form, symmetry=symm,
                                threads=args.threads,
                                time_limit=args.time_limit,
                                use_cutoff=not args.no_cutoff, cuts=cuts)
        for r in rows:
            r['method'] = 'milp-m'
            r['task'] = tid
            yield r
        summary = dict(best) if best else {'n': n, 'T': T,
                                           'topology': tp.LETTERS[tp.parse(top)],
                                           'status': 'NO_SOLUTION'}
        summary['method'] = 'milp'
        summary['m'] = best['m'] if best else None
        # the summary must carry the cost of the WHOLE m sweep, not of the one
        # subproblem that happened to be optimal
        summary['seconds'] = round(sum(r['seconds'] or 0 for r in rows), 3)
        summary['nodes'] = sum(r['nodes'] or 0 for r in rows)
        summary['task'] = tid
        yield summary

    elif method == 'cpsat':
        from gol import cpsat
        symm = bool(int(spec.get('sym', 1)))
        best, rows = cpsat.solve(n, T, top, symmetry=symm,
                                 threads=args.threads,
                                 time_limit=args.time_limit)
        for r in rows:
            r['method'] = 'cpsat-m'
            r['task'] = tid
            yield r
        summary = dict(best) if best else {
            'n': n, 'T': T, 'topology': tp.LETTERS[tp.parse(top)],
            'status': 'NO_SOLUTION'}
        summary['method'] = 'cpsat'
        summary['m'] = best['m'] if best else None
        summary['seconds'] = round(sum(r['seconds'] or 0 for r in rows), 3)
        summary['nodes'] = sum(r['nodes'] or 0 for r in rows)
        summary['task'] = tid
        yield summary

    elif method == 'duel':
        # one fixed-m subproblem, both solvers, for the paired benchmark
        from gol import cpsat, milp
        mm = int(spec['m'])
        for name, r in (('gurobi', milp.solve_fixed_m(
                            n, T, top, mm, threads=args.threads,
                            time_limit=args.time_limit)),
                        ('cpsat', cpsat.solve_fixed_m(
                            n, T, top, mm, threads=args.threads,
                            time_limit=args.time_limit))):
            r['method'] = 'duel-' + name
            r['task'] = tid
            yield r

    elif method == 'ratio':
        from gol import milp
        r = milp.solve_ratio(n, T, top, formulation=spec.get('form', 'bigm'),
                             symmetry=bool(int(spec.get('sym', 1))),
                             cuts=bool(int(spec.get('cuts', 1))),
                             threads=args.threads, time_limit=args.time_limit,
                             relax=bool(int(spec.get('lp', 0))))
        r['method'] = 'ratio'
        r['task'] = tid
        yield r

    elif method == 'root':
        # root LP bound and MIP effort of the fixed-m model, per m.
        # This is the formulation-strength experiment.
        from gol import milp
        form = spec.get('form', 'bigm')
        symm = bool(int(spec.get('sym', 1)))
        cuts = bool(int(spec.get('cuts', 1)))
        # A formulation comparison needs the SAME instances for every variant
        # and a spread of difficulty, not every population: sweeping all of
        # them costs hours per task and adds nothing.  `ms` fixes the set.
        ms = spec.get('ms')
        pops = ([int(v) for v in ms.split('.')] if ms
                else [m for m in (4, 6, 8, 10, 12, 16, 20, 24) if m <= n * n])
        for m in pops:
            for relax in (True, False):
                r = milp.solve_fixed_m(n, T, top, m, formulation=form,
                                       symmetry=symm, cuts=cuts,
                                       threads=args.threads,
                                       time_limit=args.time_limit,
                                       relax=relax)
                r['method'] = 'root-lp' if relax else 'root-mip'
                r['task'] = tid
                yield r
    else:
        raise ValueError('unknown method %r' % (method,))


def cmd_worker(args):
    with open(args.tasks, encoding='utf-8') as f:
        tasks = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    mine = [t for i, t in enumerate(tasks) if i % args.of == args.shard]
    done = io.done_tasks(args.out)
    f, w = io.open_writer(args.out)
    print('shard %d/%d: %d of %d tasks, %d already done'
          % (args.shard, args.of, len(mine), len(tasks), len(done)), flush=True)

    errors = 0
    for line in mine:
        spec = parse_task(line)
        tid = task_id(spec)
        if tid in done:
            print('skip  %s' % line, flush=True)
            continue
        print('start %s' % line, flush=True)
        try:
            for row in run_task(spec, args):
                io.write(f, w, row)
            print('done  %s' % line, flush=True)
        except Exception:
            traceback.print_exc()
            errors += 1
            io.write(f, w, {'method': spec['method'], 'task': tid,
                            'n': spec.get('n'), 'T': spec.get('T'),
                            'topology': spec.get('top'), 'status': 'ERROR'})
    f.close()
    # keep going past a failed task, but let the caller know it happened
    return 1 if errors else 0


def cmd_solve(args):
    spec = {'method': args.method, 'n': str(args.n[0]), 'T': str(args.T[0]),
            'top': args.topologies[0]}
    if args.formulations:
        spec['form'] = args.formulations[0]
    spec['sym'] = str(int(args.symmetry))
    spec['cuts'] = str(int(args.cuts[0]))
    if args.method == 'root':
        spec['ms'] = '.'.join(str(v) for v in args.ms)
    f = w = None
    if args.out:
        f, w = io.open_writer(args.out)
    for row in run_task(spec, args):
        keep = {k: v for k, v in row.items()
                if k in ('method', 'topology', 'n', 'T', 'm', 'rate',
                         'final_cells', 'status', 'seconds', 'formulation')}
        print(keep, flush=True)
        if w:
            io.write(f, w, row)
    if f:
        f.close()


# ------------------------------------------------------------------- output

def cmd_check(args):
    print('structural check of all six surfaces')
    bad = 0
    for n in range(3, 13):
        for cfg in tp.ALL:
            r = tp.check(n, cfg)
            if r['problems']:
                bad += 1
                print('  n=%d %-11s %s' % (n, tp.NAMES[cfg], r['problems']))
    print('  %s' % ('all clear' if not bad else '%d problems' % bad))

    print('\ndegree profiles (n=6)')
    for cfg in tp.ALL:
        print('  %-11s %s' % (tp.NAMES[cfg], tp.degree_profile(6, cfg)))

    print('\nupper bounds  min(3^T, (2T+1)^2, n^2/3)')
    print('  %4s %10s %10s %10s %10s' % ('n/T', '3^T', '(2T+1)^2', 'n^2/3',
                                         'best'))
    for n in (6, 10, 20):
        for T in (1, 2, 3, 5, 8):
            b = bd.upper_bound_breakdown(n, T)
            print('  %2d/%-2d %10.1f %10.1f %10.2f %10.2f'
                  % (n, T, b['triple'], b['lightcone'], b['board'], b['best']))

    print('\nlower bound: stable 2x2 block')
    for n in range(3, 9):
        miss = [tp.LETTERS[c] for c in tp.ALL
                if bd.block_lower_bound(n, c) is None]
        print('  n=%d  %s' % (n, 'all six' if not miss
                              else 'missing on ' + ','.join(miss)))


def cmd_analyse(args):
    """Evidence for the two asymptotic conjectures about the square."""
    import csv as _csv
    g = {}
    with open(args.csv, newline='', encoding='utf-8') as f:
        for row in _csv.DictReader(f):
            if row['method'] not in PROVEN_METHODS or row['topology'] != 's':
                continue
            if row.get('T') != '1' or not row.get('rate'):
                continue
            if row.get('status') not in ('OPTIMAL', None, ''):
                continue
            n = int(row['n'])
            g[n] = max(g.get(n, 0.0), float(row['rate']))

    print('rate of convergence of G_1^s(n) to 3')
    print('  %4s %12s %12s %12s' % ('n', 'G_1^s(n)', '3 - G', 'n(3 - G)'))
    for n in sorted(g):
        gap = 3.0 - g[n]
        print('  %4d %12.6f %12.6f %12.4f' % (n, g[n], gap, n * gap))
    print('  a flat last column means the gap is Theta(1/n), ruling out O(1/n^2)')

    print('\nis the block family optimal where it applies (n = 3k+2)?')
    print('  %4s %4s %12s %12s %8s' % ('k', 'n', 'family', 'optimum', 'equal'))
    for k in range(1, 9):
        n, m, fin, rate = bd.family_rate(k)
        opt = g.get(n)
        print('  %4d %4d %12.6f %12s %8s'
              % (k, n, rate, '%.6f' % opt if opt else '--',
                 ('yes' if abs(rate - opt) < 1e-9 else 'NO') if opt else '--'))


def cmd_duel(args):
    """Gurobi against CP-SAT on the paired fixed-m subproblems.

    The ratio is CP-SAT time over Gurobi time on the same subproblem, taken
    only over pairs that both solvers proved optimal; pairs where one or both
    hit the time limit are counted separately, since they have no ratio.
    """
    import csv as _csv
    import statistics as _st
    runs = {}
    with open(args.csv, newline='', encoding='utf-8') as f:
        for row in _csv.DictReader(f):
            if row['method'] in ('duel-gurobi', 'duel-cpsat'):
                runs.setdefault(row['task'], {})[row['method'][5:]] = row

    pairs, censored = [], {}
    for d in runs.values():
        if len(d) < 2:
            continue
        g, c = d['gurobi'], d['cpsat']
        n, T = int(g['n']), int(g['T'])
        og, oc = g['status'] == 'OPTIMAL', c['status'] == 'OPTIMAL'
        if og and oc:
            ratio = float(c['seconds']) / max(float(g['seconds']), 1e-3)
            pairs.append((n, T, ratio))
        else:
            who = 'gurobi only' if og else 'cpsat only' if oc else 'neither'
            censored[(T, who)] = censored.get((T, who), 0) + 1

    Ts = sorted({T for _, T, _ in pairs})
    ns = sorted({n for n, _, _ in pairs})
    print('%d pairs, %d proved optimal by both'
          % (len(pairs) + sum(censored.values()), len(pairs)))
    print('\nratio CP-SAT / Gurobi, over pairs both proved optimal')
    print('  %4s %6s %8s %18s %14s'
          % ('T', 'pairs', 'median', 'interquartile', 'CP-SAT faster'))
    for T in Ts:
        xs = [r for _, t, r in pairs if t == T]
        q1, _, q3 = _st.quantiles(xs, n=4, method='inclusive')
        print('  %4d %6d %8.2f %8.2f--%-8.2f %13.0f%%'
              % (T, len(xs), _st.median(xs), q1, q3,
                 100.0 * sum(r < 1 for r in xs) / len(xs)))

    print('\nmedian ratio by horizon and board size (pairs)')
    print('  %4s ' % 'T' + ''.join('%16s' % ('n=%d' % n) for n in ns))
    for T in Ts:
        cells = []
        for n in ns:
            xs = [r for m, t, r in pairs if t == T and m == n]
            cells.append('%16s' % ('%.2f (%d)' % (_st.median(xs), len(xs))
                                   if xs else '--'))
        print('  %4d ' % T + ''.join(cells))

    print('\npairs with no ratio: proved optimal by')
    for T in sorted({T for T, _ in censored}):
        print('  T=%d  ' % T + ', '.join(
            '%s %d' % (who, censored.get((T, who), 0))
            for who in ('gurobi only', 'cpsat only', 'neither')))


def cmd_merge(args):
    k = io.merge(args.parts, args.out)
    print('%d rows -> %s' % (k, args.out))


def cmd_table(args):
    import csv as _csv
    best = {}
    with open(args.csv, newline='', encoding='utf-8') as f:
        for row in _csv.DictReader(f):
            if row['method'] not in PROVEN_METHODS or not row.get('rate'):
                continue
            if row.get('status') not in ('OPTIMAL', None, ''):
                continue
            key = (int(row['n']), int(row['T']), row['topology'])
            r = float(row['rate'])
            if key not in best or r > best[key][0]:
                best[key] = (r, row['method'])
    ns = sorted({k[0] for k in best})
    Ts = sorted({k[1] for k in best})
    order = ['s', 'c', 't', 'm', 'k', 'p']
    print(r'\begin{tabular}{rr' + 'r' * len(order) + '}')
    print(r'\toprule')
    print('$n$ & $T$ & ' + ' & '.join(
        r'$\mathcal{G}^%s_T(n)$' % c for c in order) + r'\\')
    print(r'\midrule')
    for n in ns:
        for T in Ts:
            cells = []
            for c in order:
                v = best.get((n, T, c))
                cells.append('%.3f' % v[0] if v else '--')
            head = '%d' % n if T == Ts[0] else ''
            print('%s & %d & %s \\\\' % (head, T, ' & '.join(cells)))
    print(r'\bottomrule')
    print(r'\end{tabular}')


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--n', type=int, nargs='+', default=[5])
        p.add_argument('--T', type=int, nargs='+', default=[1])
        p.add_argument('--topologies', nargs='+', default=list('sctmkp'))
        p.add_argument('--formulations', nargs='+', default=['bigm'])
        p.add_argument('--symmetry', type=int, default=1)
        p.add_argument('--cuts', type=int, nargs='+', default=[1])
        p.add_argument('--ms', type=int, nargs='+',
                       default=[4, 6, 8, 10, 12, 16, 20])
        p.add_argument('--threads', type=int, default=0)
        p.add_argument('--time-limit', type=float, default=0)
        p.add_argument('--no-cutoff', action='store_true')
        p.add_argument('--chunks', type=int, default=1)

    p = sub.add_parser('check');    p.set_defaults(fn=cmd_check)

    p = sub.add_parser('tasks');    common(p)
    p.add_argument('--method', default='milp',
                   choices=['milp', 'dp', 'brute', 'root', 'cpsat', 'duel'])
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser('worker')
    p.add_argument('--tasks', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--of', type=int, default=1)
    p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--time-limit', type=float, default=0)
    p.add_argument('--no-cutoff', action='store_true')
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser('solve');    common(p)
    p.add_argument('--method', default='milp',
                   choices=['milp', 'dp', 'brute', 'ratio', 'root',
                            'cpsat', 'duel'])
    p.add_argument('--out')
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser('merge')
    p.add_argument('--out', required=True)
    p.add_argument('parts', nargs='+')
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser('table')
    p.add_argument('--csv', required=True)
    p.set_defaults(fn=cmd_table)

    p = sub.add_parser('analyse')
    p.add_argument('--csv', required=True)
    p.set_defaults(fn=cmd_analyse)

    p = sub.add_parser('duel')
    p.add_argument('--csv', required=True)
    p.set_defaults(fn=cmd_duel)

    args = ap.parse_args()
    return args.fn(args) or 0


if __name__ == '__main__':
    sys.exit(main())
