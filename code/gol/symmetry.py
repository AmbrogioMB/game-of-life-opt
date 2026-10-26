"""Automorphism groups of the six discrete surfaces, and a valid symmetry break.

`automorphisms` searches a family of candidate maps (a dihedral operation on the
square followed by a translation, folded by the gluing) and keeps exactly those
that preserve the adjacency structure.  Nothing is assumed: every returned
permutation is verified against the adjacency.  The result is therefore a
verified SUBGROUP of Aut, not necessarily all of it -- which is all symmetry
breaking needs, since any subgroup yields valid inequalities.

On the projective plane the four corner cells have degree 6, so translations are
*not* automorphisms and the group found is small; a symmetry break that fixed
one designated cell alive would be invalid there.  The weight inequality below
stays valid for every surface.

The symmetry break is the standard "maximum-weight orbit representative": fix a
weight w on cells and keep only the image of each configuration that maximises
sum_c w_c x_c.  That representative satisfies

    sum_d (w_d - w_{g(d)}) x_d >= 0     for every g in the group,

which is one linear inequality per group element, on the t=0 variables only, and
is valid for every topology and every group -- transitive or not.
"""
from . import topology as tp

_DIHEDRAL = (
    lambda n, i, j: (i, j),
    lambda n, i, j: (j, n - 1 - i),
    lambda n, i, j: (n - 1 - i, n - 1 - j),
    lambda n, i, j: (n - 1 - j, i),
    lambda n, i, j: (i, n - 1 - j),
    lambda n, i, j: (n - 1 - i, j),
    lambda n, i, j: (j, i),
    lambda n, i, j: (n - 1 - j, n - 1 - i),
)


def _candidates(n, cfg):
    for d in _DIHEDRAL:
        for di in range(n):
            for dj in range(n):
                perm = {}
                ok = True
                for i in range(n):
                    for j in range(n):
                        a, b = d(n, i, j)
                        c = tp.fold(n, cfg, a + di, b + dj)
                        if c is None:
                            ok = False
                            break
                        perm[(i, j)] = c
                    if not ok:
                        break
                if ok and len(set(perm.values())) == n * n:
                    yield perm


def automorphisms(n, cfg, corner_policy='simple'):
    """All verified adjacency-preserving permutations found by the search."""
    cfg = tp.parse(cfg)
    adj = {c: frozenset(v) for c, v in
           tp.adjacency(n, cfg, corner_policy).items()}
    found, seen = [], set()
    for perm in _candidates(n, cfg):
        key = tuple(perm[(i, j)] for i in range(n) for j in range(n))
        if key in seen:
            continue
        if all(frozenset(perm[k] for k in adj[c]) == adj[perm[c]] for c in adj):
            seen.add(key)
            found.append(perm)
    return found


def is_transitive(n, cfg, corner_policy='simple'):
    g = automorphisms(n, cfg, corner_policy)
    orbit = {p[(0, 0)] for p in g}
    return len(orbit) == n * n


def orbits(n, cfg, corner_policy='simple'):
    g = automorphisms(n, cfg, corner_policy)
    seen, out = set(), []
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            o = {p[(i, j)] for p in g}
            seen |= o
            out.append(sorted(o))
    return out


def breaking_inequalities(n, cfg, corner_policy='simple', weights=None):
    """[{cell: coefficient}] -- each dict is one valid inequality  >= 0.

    Default weights decrease along the row-major order, so the retained
    representative is the one pushed towards the top-left corner.
    """
    if weights is None:
        weights = {(i, j): float(n * n - (i * n + j))
                   for i in range(n) for j in range(n)}
    rows = []
    for g in automorphisms(n, cfg, corner_policy):
        coef = {}
        for c in weights:
            v = weights[c] - weights[g[c]]
            if abs(v) > 1e-12:
                coef[c] = v
        if coef:
            rows.append(coef)
    # drop duplicates
    uniq, seen = [], set()
    for coef in rows:
        key = tuple(sorted((c, round(v, 9)) for c, v in coef.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(coef)
    return uniq
