"""Grid topologies as quotients of the n x n square.

Each surface is obtained by gluing opposite edges of the fundamental square.
A neighbour offset (di,dj) that leaves the square is folded back by the gluing
map; if the relevant pair of edges is not glued, the neighbour simply does not
exist.

    surface            left~right          top~bottom
    ---------------------------------------------------------------
    square       (s)   --                  --
    cylinder     (c)   direct              --
    torus        (t)   direct              direct
    Moebius      (m)   flipped             --
    Klein        (k)   direct              flipped
    projective   (p)   flipped             flipped

"flipped" means orientation-reversing: crossing the left/right seam reflects the
row index, crossing the top/bottom seam reflects the column index.

Degeneracy on the projective plane
----------------------------------
On RP^2 the four corner cells have a degenerate neighbourhood: one of the eight
offsets folds back onto the cell itself, and two further offsets fold onto the
same opposite corner.  The cell-adjacency object is therefore a multigraph with
a loop and a double edge at each corner, and it is *not* 8-regular.  This is
intrinsic to the square model of RP^2, not an artefact of the implementation.
`corner_policy` declares how the Game of Life rule reads that neighbourhood:

    'simple'      distinct cells, self excluded  -> corner degree 6  (default)
    'multiplicity' self excluded, double edge counted twice -> 7
    'raw'          fold multiset taken verbatim, self included -> 8

'simple' is the default because the GoL rule counts *live neighbouring cells*,
and a cell is neither its own neighbour nor two of its own neighbours.
"""

SQUARE, CYLINDER, TORUS, MOBIUS, KLEIN, PROJECTIVE = range(6)

NAMES = {SQUARE: 'square', CYLINDER: 'cylinder', TORUS: 'torus',
         MOBIUS: 'moebius', KLEIN: 'klein', PROJECTIVE: 'projective'}
LETTERS = {SQUARE: 's', CYLINDER: 'c', TORUS: 't',
           MOBIUS: 'm', KLEIN: 'k', PROJECTIVE: 'p'}
BY_LETTER = {v: k for k, v in LETTERS.items()}
BY_NAME = {v: k for k, v in NAMES.items()}
ALL = (SQUARE, CYLINDER, TORUS, MOBIUS, KLEIN, PROJECTIVE)

# (glue_horizontally, flip_on_horizontal, glue_vertically, flip_on_vertical)
_GLUING = {
    SQUARE:     (False, False, False, False),
    CYLINDER:   (True,  False, False, False),
    TORUS:      (True,  False, True,  False),
    MOBIUS:     (True,  True,  False, False),
    KLEIN:      (True,  False, True,  True),
    PROJECTIVE: (True,  True,  True,  True),
}


def parse(spec):
    """Accept an int, a letter ('s'), or a name ('square')."""
    if isinstance(spec, int):
        if spec not in NAMES:
            raise ValueError('unknown topology code %r' % (spec,))
        return spec
    s = str(spec).strip().lower()
    if s in BY_LETTER:
        return BY_LETTER[s]
    if s in BY_NAME:
        return BY_NAME[s]
    if s.isdigit():
        return parse(int(s))
    raise ValueError('unknown topology %r' % (spec,))


def fold(n, cfg, p, q):
    """Fold a (possibly out-of-range) cell back into the fundamental square.

    Returns (p, q) or None if the cell falls off a free (unglued) edge.
    """
    gh, fh, gv, fv = _GLUING[cfg]
    if not (0 <= q < n):
        if not gh:
            return None
        q %= n
        if fh:
            p = n - 1 - p
    if not (0 <= p < n):
        if not gv:
            return None
        p %= n
        if fv:
            q = n - 1 - q
    if 0 <= p < n and 0 <= q < n:
        return (p, q)
    return None


def neighbours(n, cfg, i, j, corner_policy='simple'):
    """The neighbours of cell (i,j), as a list (order irrelevant).

    Under 'simple' the list has no repeats and never contains (i,j).
    """
    raw = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            c = fold(n, cfg, i + di, j + dj)
            if c is not None:
                raw.append(c)
    if corner_policy == 'raw':
        return raw
    raw = [c for c in raw if c != (i, j)]
    if corner_policy == 'multiplicity':
        return raw
    if corner_policy != 'simple':
        raise ValueError('unknown corner_policy %r' % (corner_policy,))
    out, seen = [], set()
    for c in raw:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def adjacency(n, cfg, corner_policy='simple'):
    """{(i,j): [neighbours]} for the whole board."""
    cfg = parse(cfg)
    return {(i, j): neighbours(n, cfg, i, j, corner_policy)
            for i in range(n) for j in range(n)}


def degree_profile(n, cfg, corner_policy='simple'):
    """{degree: how many cells have it} -- used in the paper's topology table."""
    adj = adjacency(n, cfg, corner_policy)
    prof = {}
    for v in adj.values():
        prof[len(v)] = prof.get(len(v), 0) + 1
    return dict(sorted(prof.items()))


def check(n, cfg, corner_policy='simple'):
    """Structural invariants every surface must satisfy.

    Returns a dict of findings; an empty 'problems' list means all good.
    """
    cfg = parse(cfg)
    adj = adjacency(n, cfg, corner_policy)
    cells = list(adj)
    problems = []

    loops = [c for c in cells if c in adj[c]]
    if loops and corner_policy != 'raw':
        problems.append('%d self-loops' % len(loops))

    dups = sum(len(v) - len(set(v)) for v in adj.values())
    if dups and corner_policy == 'simple':
        problems.append('%d repeated neighbours' % dups)

    asym = [(c, k) for c in cells for k in set(adj[c]) if c not in set(adj[k])]
    if asym:
        problems.append('%d asymmetric pairs (e.g. %r)' % (len(asym), asym[0]))

    # closed surfaces are 8-regular away from the RP^2 corners
    prof = degree_profile(n, cfg, corner_policy)
    expected = {
        SQUARE:     {3: 4, 5: 4 * (n - 2), 8: (n - 2) ** 2},
        CYLINDER:   {5: 2 * n, 8: n * (n - 2)},
        MOBIUS:     {5: 2 * n, 8: n * (n - 2)},
        TORUS:      {8: n * n},
        KLEIN:      {8: n * n},
        PROJECTIVE: {6: 4, 8: n * n - 4} if corner_policy == 'simple' else None,
    }[cfg]
    if expected is not None and prof != expected:
        problems.append('degree profile %r, expected %r' % (prof, expected))

    return {'topology': NAMES[cfg], 'n': n, 'degrees': prof,
            'loops': len(loops), 'repeats': dups, 'problems': problems}
