"""What actually maximises one-step growth on the square?

The block family of the draft is optimal at n = 5 and n = 8 and beaten at
n = 11, so what the optimal seeds look like is open.  This runs three stages:

  1. recover an optimal seed for every n <= 11 where the optimum is known, by
     solving the fixed-m integer program at the optimal population -- finding a
     witness is easy for the MILP even where certifying the optimum is not;

  2. search the class of periodically tiled configurations, to see how much of
     the optimum comes from periodicity alone;

  3. push past the reach of the exact methods with incumbents, which certify
     nothing but are valid LOWER bounds.  The question that matters: the block
     family is defined again at n = 14, where it gives 192/73 = 2.630137.  Is
     it beaten there, as it was at n = 11?

Output is appended to overnight_structure.log as it goes.
"""
import sys
import time

sys.path.insert(0, '.')

from gol import bounds as bd            # noqa: E402
from gol import simulate as sim         # noqa: E402
from gol import topology as tp          # noqa: E402

LOG = open('overnight_structure.log', 'a', encoding='utf-8', buffering=1)
NL = chr(10)


def say(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + NL)


# optimum as numerator/denominator, from the transfer-matrix dynamic program
KNOWN = {3: (7, 4), 4: (9, 5), 5: (21, 10), 6: (32, 14), 7: (32, 14),
         8: (60, 25), 9: (75, 30), 10: (75, 30), 11: (112, 44)}


def growth(n, x, nbl):
    m = sim.popcount(x)
    return 0.0 if m < bd.MIN_SEED else sim.popcount(sim.step(x, nbl, n)) / m


# ------------------------------------------------------- 1. exact witnesses

def witnesses():
    say('')
    say('=== 1. optimal seeds for n <= 11 (integer program at the optimal m) ===')
    from gol import milp
    out = {}
    for n in sorted(KNOWN):
        num, den = KNOWN[n]
        t0 = time.time()
        r = milp.solve_fixed_m(n, 1, 's', den, threads=8, time_limit=900)
        if r['board'] is None or r['final_cells'] != num:
            say('  n=%-3d FAILED (status=%s, got %s, want %d)'
                % (n, r['status'], r['final_cells'], num))
            continue
        seed = sim.decode(n, r['board'])
        out[n] = seed
        say('  n=%-3d  m=%-3d |X(1)|=%-4d ratio=%.6f   %.1f s   b64=%s'
            % (n, den, num, num / den, time.time() - t0, r['board']))
        for row in sim.to_string(n, seed).split(NL):
            say('        ' + row)
    return out


# --------------------------------------------- 2. periodically tiled search

def tiled(n, tile, k, oi, oj):
    cells = [(i, j) for i in range(n) for j in range(n)
             if (tile >> (((i - oi) % k) * k + ((j - oj) % k))) & 1]
    return sim.from_cells(n, cells)


def best_tiled(n, ks=(2, 3)):
    nbl = sim.neighbour_lists(n, tp.SQUARE)
    best, arg = 0.0, None
    for k in ks:
        for tile in range(1, 1 << (k * k)):
            for oi in range(k):
                for oj in range(k):
                    x = tiled(n, tile, k, oi, oj)
                    g = growth(n, x, nbl)
                    if g > best:
                        best, arg = g, (k, tile, oi, oj, sim.popcount(x), x)
    return best, arg


def search_class():
    say('')
    say('=== 2. best periodic tiling, against the known optimum ===')
    say('  %3s %13s %13s %9s' % ('n', 'best tiling', 'optimum', 'matches?'))
    for n in sorted(KNOWN):
        b, arg = best_tiled(n)
        num, den = KNOWN[n]
        opt = num / den
        say('  %3d %13.6f %13.6f %9s   k=%d mask=%d off=(%d,%d) m=%d'
            % (n, b, opt, 'yes' if abs(b - opt) < 1e-12 else 'below',
               arg[0], arg[1], arg[2], arg[3], arg[4]))
    say('  periodicity alone does not reach the optimum: the optima carry')
    say('  boundary corrections that a rigid tiling cannot express.')


# ------------------------------------------------- 3. lower bounds past n=11

def improve(n, x, rounds=8):
    """Greedy single-flip local search.  Captures the boundary corrections."""
    nbl = sim.neighbour_lists(n, tp.SQUARE)
    best = growth(n, x, nbl)
    for _ in range(rounds):
        moved = False
        for c in range(n * n):
            y = x ^ (1 << c)
            gy = growth(n, y, nbl)
            if gy > best + 1e-15:
                x, best, moved = y, gy, True
        if not moved:
            break
    return best, x


def lower_bounds(lo=12, hi=18, tl=600):
    from gol import milp
    say('')
    say('=== 3. lower bounds on G_1^s(n) beyond the exact methods ===')
    say('  Incumbents certify nothing, but each is a valid lower bound, and')
    say('  one above the block family refutes the family at that n.')
    say('  %3s %13s %13s %11s %11s' % ('n', 'best found', 'block family',
                                       'beats it?', 'n(3-best)'))
    # n = 14 first: it is the next board on which the block family is defined,
    # and therefore the one that decides whether n = 11 was an accident.
    order = [14] + [k for k in range(lo, hi) if k != 14]
    for n in order:
        fam = None
        if (n - 2) % 3 == 0:
            _, _, _, fam = bd.family_rate((n - 2) // 3)

        best, seed = best_tiled(n, ks=(3,))
        seed = seed[5]
        best, seed = improve(n, seed)
        if fam is not None:
            _, fs = bd.family_seed((n - 2) // 3)
            fb, fs = improve(n, fs)
            if fb > best:
                best, seed = fb, fs
        say('      n=%d start %.6f (tiling + local search)' % (n, best))

        for m in range(max(3, int(n * n / 3.2)), int(n * n / 2.3) + 1):
            need = int(best * m) + 1
            if need > min(n * n, 3 * m):
                continue
            r = milp.solve_fixed_m(n, 1, 's', m, threads=8, time_limit=tl,
                                   cutoff_cells=need)
            if r.get('final_cells'):
                cand = r['final_cells'] / m
                if cand > best:
                    best = cand
                    if r['board']:
                        seed = sim.decode(n, r['board'])
                    say('      n=%d m=%d -> %d/%d = %.6f (%s, %.0fs)'
                        % (n, m, r['final_cells'], m, cand, r['status'],
                           r['seconds']))
        if seed is not None:
            best, seed = improve(n, seed)

        verdict = ('--' if fam is None
                   else ('YES' if best > fam + 1e-12
                         else ('ties' if abs(best - fam) < 1e-12 else 'no')))
        say('  %3d %13.6f %13s %11s %11.4f'
            % (n, best, ('%.6f' % fam) if fam else '--', verdict,
               n * (3 - best)))
        if seed is not None:
            for row in sim.to_string(n, seed).split(NL):
                say('        ' + row)


if __name__ == '__main__':
    say('')
    say('=' * 70)
    say('started ' + time.strftime('%Y-%m-%d %H:%M:%S'))
    for name, fn in (('witnesses', witnesses),
                     ('class search', search_class),
                     ('lower bounds', lower_bounds)):
        try:
            fn()
        except Exception as e:
            say('  %s stage failed: %r' % (name, e))
    say('finished ' + time.strftime('%Y-%m-%d %H:%M:%S'))
