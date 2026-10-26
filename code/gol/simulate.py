"""Reference Game of Life dynamics on any of the six surfaces.

A configuration is an int bitmask: cell (i,j) is bit i*n+j.  For n <= 8 the whole
board fits in 64 bits, which is what the exhaustive search relies on.
"""
import base64

from . import topology as tp


def neighbour_lists(n, cfg, corner_policy='simple'):
    """[[flat neighbour indices] for each flat cell index]."""
    cfg = tp.parse(cfg)
    adj = tp.adjacency(n, cfg, corner_policy)
    return [[p * n + q for (p, q) in adj[(i, j)]]
            for i in range(n) for j in range(n)]


def step(x, nbl, n):
    """One generation.  `nbl` comes from neighbour_lists."""
    y = 0
    for c in range(n * n):
        s = 0
        for k in nbl[c]:
            s += (x >> k) & 1
        if s == 3 or (s == 2 and (x >> c) & 1):
            y |= 1 << c
    return y


def evolve(x, nbl, n, T):
    for _ in range(T):
        x = step(x, nbl, n)
    return x


def trajectory(x, nbl, n, T):
    out = [x]
    for _ in range(T):
        x = step(x, nbl, n)
        out.append(x)
    return out


def popcount(x):
    return bin(x).count('1')


def growth(x, nbl, n, T):
    """|X(T)| / |X(0)|, or 0.0 for the empty configuration."""
    m = popcount(x)
    if m == 0:
        return 0.0
    return popcount(evolve(x, nbl, n, T)) / m


# ---------------------------------------------------------------- conversions

def from_cells(n, cells):
    x = 0
    for (i, j) in cells:
        x |= 1 << (i * n + j)
    return x


def to_cells(n, x):
    return [(c // n, c % n) for c in range(n * n) if (x >> c) & 1]


def to_string(n, x, alive='#', dead='.'):
    return '\n'.join(''.join(alive if (x >> (i * n + j)) & 1 else dead
                             for j in range(n)) for i in range(n))


def from_string(s):
    rows = [r.strip() for r in s.strip().splitlines() if r.strip()]
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError('not a square board')
    return from_cells(n, [(i, j) for i in range(n)
                          for j in range(n) if rows[i][j] not in '.0 '])


def encode(n, x):
    """Compact, stable base64 encoding (big-endian, row-major, n^2 bits)."""
    nbytes = (n * n + 7) // 8
    return base64.b64encode(x.to_bytes(nbytes, 'big')).decode()


def decode(n, s):
    return int.from_bytes(base64.b64decode(s), 'big')


def to_tikz(n, x, cfg):
    r"""A \drawboard{...} call for the LaTeX source."""
    cfg = tp.parse(cfg)
    cells = ','.join('%d/%d' % (i, j) for (i, j) in to_cells(n, x))
    return r'\drawboard{%d}{%d}{%s}{%d}' % (n, n, cells, cfg)
