"""Audit the released small-board certificates and rerun the m=8 search.

Compile exact_m8.cpp, then pass its executable with --executable. The solver
status of the time-limited square n=6, T=3 sweep in data/all.csv is FEASIBLE;
the combined certificate that closes it is written to a separate JSON report.
A later sweep of the same task without a time limit, in
data/eos_milp_n6_t3_square.csv, settles every population and is rechecked too.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import zipfile

from gol import bounds, simulate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
TASK = 'milp|n=6|T=3|top=s|form=bigm|sym=1|cuts=1'

# Task groups of the serial batch, as named in data/raw/batch.zip, with their
# number of tasks. Each has data/<group>.csv and tasks/<group>.txt.
BATCH = dict(brute_small=54, milp_xval=52, milp_n6=7, dp_small=14, dp_large=4)
# Files listed in the batch manifest but not released: they identify a private
# notification channel and have no bearing on the results.
WITHHELD = {'NTFY_TOPIC.txt', 'README.md'}
# Every experiment CSV that enters data/all.csv, in merge order.
SOURCES = ('brute_small', 'brute_n6', 'dp_small', 'dp_large', 'milp_xval',
           'milp_n6', 'milp_n6_long', 'milp_n6_t3', 'milp_n78',
           'cpsat_table', 'duel', 'formulation')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    with (DATA/name).open(newline='') as f:
        return list(csv.DictReader(f))


def rows_by_key(rows):
    return {(r['method'], r['task'], r['m']): r for r in rows}


def task_set(name):
    return {'|'.join(line.split()) for line in (ROOT/'tasks'/name).read_text().splitlines()
            if line.strip() and not line.startswith('#')}


def key(row):
    return int(row['n']), int(row['T']), row['topology']


def witness(row):
    n, steps, top = key(row)
    seed = simulate.decode(n, row['board'])
    require(0 < seed < 1 << (n*n), 'invalid seed')
    pop = seed.bit_count()
    final = simulate.evolve(seed, simulate.neighbour_lists(n, top), n, steps).bit_count()
    require(math.isclose(final/pop, float(row['rate']), abs_tol=1e-12),
            'witness rate mismatch: '+row.get('task', ''))
    if row.get('final_cells') not in (None, ''):
        require(final == int(row['final_cells']) and pop == int(row['m']),
                'witness population mismatch')
    return Fraction(final, pop)


def audit_batch(exact):
    """Check the batch CSVs against the raw archive, then recheck every proof."""
    tasks = witnesses = 0
    with zipfile.ZipFile(DATA/'raw/batch.zip') as archive:
        require(archive.testzip() is None, 'corrupt batch archive')
        names = set(archive.namelist())
        manifest = json.loads(archive.read('manifest.json'))
        for name, expected in manifest.items():
            if name in WITHHELD:
                require(name not in names, 'withheld file present: '+name)
                continue
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    'batch source hash mismatch: '+name)
        state = json.loads(archive.read('run.json'))
        require(state['status'] == 'COMPLETED', 'batch did not complete')
        environment = json.loads(archive.read('results/environment.json'))
        require(environment['manifest_sha256']
                == hashlib.sha256(archive.read('manifest.json')).hexdigest(),
                'environment refers to another manifest')
        for source, released in [('run.json', 'run.json'),
                                 ('results/environment.json', 'environment.json'),
                                 ('results/audit.json', 'run_audit.json'),
                                 ('results/exact_m8.json', 'exact_m8.json')]:
            require(archive.read(source) == (DATA/released).read_bytes(),
                    'metadata differs from archive: '+released)
        audit = json.loads(archive.read('results/audit.json'))
        require(audit['status'] == 'VERIFIED' and not audit['missing']
                and not audit['unresolved'], 'batch audit incomplete')
        for name, expected in audit['hashes'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    'batch result hash mismatch: '+name)
        shipped = json.loads(archive.read('results/exact_m8.json'))
        require(all(exact[k] == shipped[k] for k in exact if k != 'seconds'),
                'fresh enumeration differs from data/exact_m8.json')
        require(shipped['source_sha256']
                == hashlib.sha256(archive.read('exact_m8.cpp')).hexdigest(),
                'C++ source hash mismatch')
        require(archive.read('exact_m8.cpp') == (ROOT/'code/exact_m8.cpp').read_bytes(),
                'released C++ source differs from the one that was run')
        for group, count in BATCH.items():
            rows = read(group+'.csv')
            raw = {}
            for name in sorted(names):
                if name.startswith('results/'+group+'/part_') and name.endswith('.csv'):
                    raw.update(rows_by_key(csv.DictReader(io.StringIO(archive.read(name).decode()))))
            require(len(raw) == len(rows) and raw == rows_by_key(rows),
                    'CSV differs from raw task files: '+group)
            require((ROOT/'tasks'/(group+'.txt')).read_bytes() == archive.read('tasks/'+group+'.txt'),
                    'task list differs from archive: '+group)
            expected = task_set(group+'.txt')
            summaries = [r for r in rows if r['method'] in ('dp', 'brute', 'milp')]
            require(len(summaries) == count == len(expected)
                    and {r['task'] for r in summaries} == expected,
                    'task coverage mismatch: '+group)
            for row in rows:
                if row.get('board'):
                    witness(row)
                    witnesses += 1
            for row in summaries:
                require(row['status'] == 'OPTIMAL' and row.get('board'), 'unfinished task')
                if row['method'] == 'milp':
                    populations = [r for r in rows
                                   if r['task'] == row['task'] and r['method'] == 'milp-m']
                    status, upper, unresolved = bounds.certify_sweep(
                        int(row['n']), int(row['T']), populations)
                    require(status == 'OPTIMAL' and not unresolved, 'MILP sweep not certified')
                    require(math.isclose(float(upper), float(row['bound']), abs_tol=1e-12),
                            'MILP upper bound mismatch')
                elif row['method'] == 'brute':
                    require(row['chunk'] == '0' and row['chunks'] == '1',
                            'incomplete enumeration')
                else:
                    require(witness(row) == Fraction(int(row['numerator']), int(row['denominator'])),
                            'DP counts disagree with witness')
                tasks += 1
    return tasks, witnesses


def audit_long(exact_row):
    """Check the long n=6 sweeps and close the square T=3 sweep with m=8."""
    rows = read('milp_n6_long.csv')
    with zipfile.ZipFile(DATA/'raw/milp_n6_long.zip') as archive:
        require(archive.testzip() is None, 'corrupt long-sweep archive')
        raw = {}
        for name in sorted(archive.namelist()):
            if name.startswith('results/milp_n6_rest/part_'):
                raw.update(rows_by_key(csv.DictReader(io.StringIO(archive.read(name).decode()))))
        require(raw == rows_by_key(rows), 'milp_n6_long.csv differs from the raw archive')
        require(archive.read('run.json') == (DATA/'milp_n6_long_run.json').read_bytes(),
                'run state differs from archive')
    summaries = [r for r in rows if r['method'] == 'milp']
    require(len(summaries) == 7 and {r['task'] for r in summaries} == task_set('milp_n6_long.txt'),
            'long-sweep task coverage mismatch')
    require(all(r.get('board') for r in summaries), 'missing long-sweep witness')
    witnesses = 0
    for row in rows:
        if row.get('board'):
            witness(row)
            witnesses += 1
    combined = None
    for row in summaries:
        populations = [r for r in rows if r['task'] == row['task'] and r['method'] == 'milp-m']
        status, upper, unresolved = bounds.certify_sweep(6, int(row['T']), populations)
        require(status == row['status'], 'MILP summary mismatch')
        require(math.isclose(float(upper), float(row['bound']), abs_tol=1e-12),
                'MILP summary upper bound mismatch')
        if row['task'] == TASK:
            require(status == 'FEASIBLE' and unresolved == [8], 'unexpected open population')
            populations = [r for r in populations if int(r['m']) != 8] + [exact_row]
            status, upper, unresolved = bounds.certify_sweep(6, 3, populations)
            require(status == 'OPTIMAL' and not unresolved and upper == Fraction(20, 7),
                    'combined proof did not close')
            combined = dict(n=6, T=3, topology='s', status=status, rate=str(upper),
                            method='MILP + exhaustive fixed-population search',
                            milp_status=row['status'], milp_seconds=float(row['seconds']),
                            exhaustive_result={k: v for k, v in exact_row.items()
                                               if k not in ('board', 'rate', 'bound')})
        else:
            require(status == 'OPTIMAL' and not unresolved, 'unexpected open sweep')
    require(combined is not None, 'missing square n=6, T=3 sweep')
    return len(summaries), witnesses, combined


def audit_cluster():
    """Recheck the later sweeps run without a time limit on a cluster.

    eos_milp_n6_t3_square.csv is the square n=6, T=3 sweep, which settles
    every population; eos_milp_n8_t1.csv holds the six n=8, T=1 sweeps. They
    are kept out of data/all.csv, whose time-limited sweep of the same task
    keeps its status. Returns the certified square rate and the counts.
    """
    witnesses = sweeps = 0
    rates = {}
    for name, wanted in (('eos_milp_n6_t3_square.csv', {(6, 3, 's')}),
                         ('eos_milp_n8_t1.csv', {(8, 1, s) for s in 'sctmkp'})):
        rows = read(name)
        summaries = [r for r in rows if r['method'] == 'milp']
        require({key(r) for r in summaries} == wanted and len(summaries) == len(wanted),
                'cluster sweep coverage mismatch: '+name)
        require(all(r['method'] in ('milp', 'milp-m') for r in rows),
                'unexpected rows: '+name)
        for row in rows:
            if row.get('board'):
                witness(row)
                witnesses += 1
        for row in summaries:
            populations = [r for r in rows
                           if r['task'] == row['task'] and r['method'] == 'milp-m']
            status, upper, unresolved = bounds.certify_sweep(
                int(row['n']), int(row['T']), populations)
            require(status == row['status'] == 'OPTIMAL' and not unresolved
                    and row.get('board'), 'cluster sweep not certified: '+row['task'])
            require(math.isclose(float(upper), float(row['bound']), abs_tol=1e-12)
                    and math.isclose(float(upper), float(row['rate']), abs_tol=1e-12),
                    'cluster sweep bound mismatch: '+row['task'])
            rates[key(row)] = upper
            sweeps += 1
    require(rates[(6, 3, 's')] == Fraction(20, 7), 'unexpected square n=6, T=3 optimum')
    # the dynamic program gives the same one-step optima at n = 8
    for row in read('dp_small.csv'):
        if key(row) in rates:
            require(row['status'] == 'OPTIMAL'
                    and Fraction(int(row['numerator']), int(row['denominator'])) == rates[key(row)],
                    'cluster sweep disagrees with the dynamic program')
    return rates[(6, 3, 's')], sweeps, witnesses


def audit_table(combined, square_rate):
    """Coverage and agreement of all 72 small-board values in data/all.csv."""
    all_rows = read('all.csv')
    merged = {}
    for name in SOURCES:
        merged.update(rows_by_key(read(name+'.csv')))
    require(merged == rows_by_key(all_rows) and len(merged) == len(all_rows),
            'all.csv differs from the experiment sources')
    require(task_set('milp_n6_t3.txt') == {r['task'] for r in read('milp_n6_t3.csv')},
            'milp_n6_t3 task coverage mismatch')
    witnesses = sweeps = 0
    for row in all_rows:
        if row.get('board'):
            witness(row)
            witnesses += 1
        if row['method'] == 'milp':
            populations = [r for r in all_rows
                           if r['task'] == row['task'] and r['method'] == 'milp-m']
            status, upper, unresolved = bounds.certify_sweep(
                int(row['n']), int(row['T']), populations)
            require(status == row['status'], 'shipped MILP summary mismatch: '+row['task'])
            if row.get('bound'):
                require(upper is not None and math.isclose(float(upper), float(row['bound']), abs_tol=1e-12),
                        'shipped MILP upper bound mismatch: '+row['task'])
            sweeps += 1
    wanted = {(n, t, s) for n in range(3, 7) for t in range(1, 4) for s in 'sctmkp'}
    brute, second, chunks = {}, {}, {}
    for row in all_rows:
        k = key(row)
        if k not in wanted or row['status'] != 'OPTIMAL' or not row['rate']:
            continue
        rate = float(row['rate'])
        if row['method'] == 'brute':
            group = chunks.setdefault((k, int(row['chunks'] or 1)), {})
            group[int(row['chunk'] or 0)] = rate
        elif row['method'] in ('dp', 'milp', 'cpsat'):
            if k in second:
                require(math.isclose(second[k], rate, abs_tol=1e-12), 'second methods disagree')
            second[k] = rate
    for (k, count), group in chunks.items():
        if set(group) == set(range(count)):
            rate = max(group.values())
            if k in brute:
                require(math.isclose(brute[k], rate, abs_tol=1e-12), 'enumerations disagree')
            brute[k] = rate
    require(set(brute) == wanted, 'incomplete enumeration certificates')
    require(wanted-set(second) == {(6, 3, 's')}, 'unexpected second-method coverage')
    for k, rate in second.items():
        require(math.isclose(rate, brute[k], abs_tol=1e-12), 'methods disagree')
    require(math.isclose(brute[(6, 3, 's')], float(Fraction(combined['rate'])), abs_tol=1e-12),
            'combined proof disagrees with enumeration')
    # the last value has its second-method certificate in the cluster sweep
    require(math.isclose(brute[(6, 3, 's')], float(square_rate), abs_tol=1e-12),
            'cluster sweep disagrees with enumeration')
    return witnesses, sweeps


def audit(executable):
    # Run a fresh complete search; never infer optimality from a witness alone.
    exact = json.loads(subprocess.check_output([str(executable)], text=True))
    require(all(exact.get(k) == v for k, v in
                {'n': 6, 'T': 3, 'topology': 's', 'm': 8, 'status': 'OPTIMAL',
                 'method': 'exhaustive-fixed-m', 'boards_scanned': math.comb(36, 8)}.items()),
            'wrong exhaustive-search specification')
    rng = random.Random(20261031)
    boards = [0, (1 << 36)-1] + [1 << i for i in range(36)]
    boards += [rng.getrandbits(36) for _ in range(64)]
    boards += [sum(1 << i for i in rng.sample(range(36), 8)) for _ in range(64)]
    nbl = simulate.neighbour_lists(6, 's')
    comparisons = 0
    for seed in boards:
        for _ in range(3):
            expected = simulate.step(seed, nbl, 6)
            actual = int(subprocess.check_output([str(executable), '--step', str(seed)], text=True))
            require(actual == expected, 'C++ / Python step mismatch')
            comparisons += 1
            seed = expected
    exact_row = dict(exact, board=simulate.encode(6, exact['seed']),
                     rate=exact['final_cells']/8, bound=exact['final_cells'])
    witness(exact_row)

    batch_tasks, batch_witnesses = audit_batch(exact)
    long_tasks, long_witnesses, combined = audit_long(exact_row)
    square_rate, cluster_sweeps, cluster_witnesses = audit_cluster()
    all_witnesses, milp_sweeps = audit_table(combined, square_rate)

    files = ['data/'+s+'.csv' for s in SOURCES] + ['tasks/'+s+'.txt' for s in SOURCES]
    files += ['data/'+n for n in ('all.csv', 'exact_m8.json', 'environment.json', 'run.json',
                                  'run_audit.json', 'milp_n6_long_run.json',
                                  'raw/batch.zip', 'raw/milp_n6_long.zip',
                                  'eos_milp_n6_t3_square.csv', 'eos_milp_n8_t1.csv',
                                  'eos_environment.json')]
    files += ['code/'+n for n in ('exact_m8.cpp', 'verify_certificates.py',
                                  'run.py', 'reproduce.py', 'tests/test_gol.py',
                                  'gol/simulate.py', 'gol/topology.py', 'gol/bounds.py',
                                  'gol/dp.py', 'gol/brute.py', 'gol/milp.py',
                                  'gol/cpsat.py', 'gol/symmetry.py', 'gol/io_utils.py')]
    return dict(status='CERTIFIED', enumeration_values=72,
                second_method_values=72, second_method_values_in_all_csv=71,
                combined_proof_values=1,
                cluster_sweeps_checked=cluster_sweeps,
                cluster_witnesses_checked=cluster_witnesses,
                batch_tasks=batch_tasks, batch_witnesses_checked=batch_witnesses,
                long_sweep_tasks=long_tasks, long_sweep_witnesses_checked=long_witnesses,
                all_witnesses_checked=all_witnesses, milp_sweeps_checked=milp_sweeps,
                cpp_python_step_comparisons=comparisons, combined_proof=combined,
                sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.executable.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: 72 enumeration certificates; 72 second-method certificates '
          '(71 in all.csv, 1 in the cluster sweep); 1 combined proof.')
    print('PASS: %d cluster sweeps without a time limit (%d witnesses).'
          % (report['cluster_sweeps_checked'], report['cluster_witnesses_checked']))
    print('PASS: %d batch tasks (%d witnesses); %d long n=6 sweeps (%d witnesses).'
          % (report['batch_tasks'], report['batch_witnesses_checked'],
             report['long_sweep_tasks'], report['long_sweep_witnesses_checked']))
    print('PASS: %d C++/Python step comparisons.' % report['cpp_python_step_comparisons'])
    print('PASS: %d shipped witnesses; %d MILP sweep summaries.'
          % (report['all_witnesses_checked'], report['milp_sweeps_checked']))
    print('Certificate:', args.out)


if __name__ == '__main__':
    main()
