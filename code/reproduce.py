"""Run a task list across several local processes, then merge.

The experiments were produced by sharding a task list across worker processes.
`tasks/*.txt` in this repository are those lists verbatim -- they are the exact
specification of what was run, down to every board size, horizon, surface and
initial population. This replaces the shell orchestration with something
portable.

    python reproduce.py ../tasks/dp_big.txt --workers 4
    python reproduce.py ../tasks/duel.txt --workers 24

Results go to ../rerun/<name>/part_<k>.csv and are merged into
../rerun/<name>.csv, so the shipped data in ../data is never overwritten.
Re-running skips tasks already recorded, so an interrupted
run resumes by repeating the command.

Note on worker counts: the exhaustive search streams large arrays and is
limited by memory bandwidth rather than cores, so more than about 24 workers
buys nothing and can cost. The solver tasks are compute-bound; give those
fewer workers with more threads each (--threads).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tasks', help='a task list, e.g. ../tasks/duel.txt')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1,
                    help='solver threads per worker')
    ap.add_argument('--time-limit', type=float, default=0,
                    help='seconds per solver subproblem; 0 = none')
    ap.add_argument('--out', default=None,
                    help='output directory (default ../rerun/<task name>)')
    a = ap.parse_args()

    tasks = os.path.abspath(a.tasks)
    if not os.path.exists(tasks):
        sys.exit('no such task list: ' + tasks)
    name = os.path.splitext(os.path.basename(tasks))[0]
    out = a.out or os.path.join(HERE, '..', 'rerun', name)
    os.makedirs(out, exist_ok=True)

    n = sum(1 for l in open(tasks, encoding='utf-8')
            if l.strip() and not l.startswith('#'))
    print('%s: %d tasks, %d workers x %d thread(s)'
          % (name, n, a.workers, a.threads), flush=True)

    env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[v] = str(a.threads)

    procs = []
    for k in range(a.workers):
        log = open(os.path.join(out, 'worker_%d.log' % k), 'w',
                   encoding='utf-8')
        procs.append((subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'run.py'), 'worker',
             '--tasks', tasks, '--shard', str(k), '--of', str(a.workers),
             '--threads', str(a.threads),
             '--time-limit', str(a.time_limit),
             '--out', os.path.join(out, 'part_%d.csv' % k)],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE), log))

    failed = 0
    for p, log in procs:
        if p.wait() != 0:
            failed += 1
        log.close()
    if failed:
        print('%d worker(s) exited non-zero; see %s/worker_*.log'
              % (failed, out), flush=True)

    parts = [os.path.join(out, f) for f in sorted(os.listdir(out))
             if f.startswith('part_') and f.endswith('.csv')]
    if not parts:
        print('no results to merge', flush=True)
        return 1
    if subprocess.call([sys.executable, os.path.join(HERE, 'run.py'),
                        'merge', '--out', out + '.csv'] + parts, cwd=HERE):
        print('merge failed', flush=True)
        return 1
    print('done -> %s.csv' % out, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
