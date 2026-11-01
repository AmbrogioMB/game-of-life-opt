"""CSV output with resume, so an interrupted cluster job can be restarted."""
import csv
import os

FIELDS = ['method', 'topology', 'n', 'T', 'formulation', 'symmetry', 'cuts', 'm',
          'rate', 'numerator', 'denominator', 'final_cells', 'bound', 'gap',
          'status', 'seconds', 'nodes', 'iterations', 'boards_scanned',
          'chunk', 'chunks', 'lb_used', 'ub_used', 'board', 'task']


def open_writer(path):
    """Append to `path`, writing the header only if the file is new."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    f = open(path, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    if new:
        w.writeheader()
        f.flush()
    return f, w


def write(f, w, row):
    w.writerow(row)
    f.flush()          # survive a wall-clock kill


def done_tasks(path, retry_feasible=False):
    """Tasks with a terminal summary or explicit completion marker.

    Per-population rows are checkpoints, not completed sweeps. On a restart
    an unfinished sweep is rerun; merge() retains its latest records.
    """
    if not os.path.exists(path):
        return set()
    latest = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            task = row.get('task')
            if not task:
                continue
            method, status = row.get('method'), row.get('status')
            if method == task.split('|', 1)[0] or method == 'task-complete':
                latest[task] = status
    terminal = {'OPTIMAL', 'FEASIBLE', 'NO_SOLUTION', 'INFEASIBLE', 'COMPLETED'}
    if retry_feasible:
        terminal -= {'FEASIBLE', 'NO_SOLUTION'}
    return {task for task, status in latest.items() if status in terminal}


def merge(paths, out):
    """Merge atomically, retaining the latest record when a task was retried."""
    import tempfile
    rows = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='', encoding='utf-8') as g:
            for row in csv.DictReader(g):
                key = (row.get('method'), row.get('task'), row.get('m'))
                rows[key] = row
    directory = os.path.dirname(os.path.abspath(out))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.merge-', suffix='.csv', dir=directory)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows.values())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return len(rows)
