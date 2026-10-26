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


def done_tasks(path):
    """Task ids already present in `path` with a usable result."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('task') and row.get('status') not in (None, '', 'ERROR'):
                out.add(row['task'])
    return out


def merge(paths, out):
    """Concatenate part files into one CSV, replacing `out` if it exists."""
    if os.path.exists(out):
        os.remove(out)
    f, w = open_writer(out)
    seen = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='', encoding='utf-8') as g:
            for row in csv.DictReader(g):
                key = (row.get('method'), row.get('task'), row.get('m'))
                if key in seen:
                    continue
                seen.add(key)
                w.writerow(row)
    f.close()
    return len(seen)
