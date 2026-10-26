# Maximum growth in the Game of Life on surfaces — code and data

Companion code for *Maximising the growth rate of Conway's Game of Life on
closed surfaces: bounds, constructions and exact methods* (Bernardelli). Given an
`n × n` board and a horizon `T`, which initial configuration maximises
`|X(T)| / |X(0)|`? The board is one of six surfaces obtained by gluing the edges
of the square.

Everything here is exact: no heuristic result is reported as an optimum, and
every optimum for `n ≤ 6` in the paper was produced by two independent methods.
The one-step optima on the square for `8 ≤ n ≤ 11` come from the dynamic
program alone.

```
code/     the library, the command line, the validation suite
tasks/    the experiment specifications, verbatim
data/     results, as CSV, one file per experiment plus all.csv
```

## Requirements

| | needed for |
| --- | --- |
| Python ≥ 3.10, numpy | everything |
| gurobipy | the integer program (`milp`, `root`, `duel`) |
| ortools | the constraint program (`cpsat`, `duel`) |

Topology, simulation, bounds and exhaustive search need only numpy.

```bash
pip install -r requirements.txt   # drop gurobipy / ortools if not needed
```

`requirements.txt` records the versions the code was tested with. The
results in `data/` were produced with Python 3.10.12, numpy 1.26.2, Gurobi
11.0.0 and OR-Tools 9.15.6755, except `milp_xval.csv` (Python 3.14.7, numpy
2.5.3, Gurobi 13.0.3).

## Start here

```bash
cd code
python -m tests.test_gol     # 143 checks, ~5 min; must end ALL TESTS PASSED
python run.py check          # structural self-check and the bound table, no solver
```

`tests/test_gol.py` is the reason to trust anything else. It verifies the six
gluings against their structural invariants, checks the Möbius dynamics against
its cylinder double cover, reproduces textbook patterns, confirms the bounds,
checks that symmetry breaking preserves an optimum, and cross-validates the
dynamic program and the integer program against exhaustive search on all six
surfaces.

## The four methods

| module | method | cost | best where |
| --- | --- | --- | --- |
| `gol/brute.py` | exhaustive enumeration | `O(T n² 2^(n²))` | ground truth, `n ≤ 6` |
| `gol/dp.py` | transfer-matrix DP + Dinkelbach | `O(n 2^((2T+1)n))` | square and cylinder at `T = 1`, to `n = 11` |
| `gol/milp.py` | integer program (Gurobi) | — | small `T`, all six surfaces |
| `gol/cpsat.py` | constraint program (CP-SAT) | — | large `T`, all six surfaces |

One-off runs:

```bash
python run.py solve --method dp    --n 10 --T 1 --topologies s
python run.py solve --method brute --n 5  --T 2 --topologies p
python run.py solve --method cpsat --n 6  --T 3 --topologies s
```

## Reproducing the experiments

`tasks/*.txt` are the experiment specifications **verbatim** — every board
size, horizon, surface, initial population and solver variant that was run.
They are the precise record; the prose below only says which produced what.

```bash
cd code
python reproduce.py ../tasks/dp_big.txt      --workers 4
python reproduce.py ../tasks/duel.txt        --workers 24
python reproduce.py ../tasks/brute_n6.txt    --workers 24
python reproduce.py ../tasks/formulation.txt --workers 8 --threads 2
```

Each shards the list across local processes, writes `rerun/<name>/part_k.csv`,
and merges to `rerun/<name>.csv`; the shipped results in `data/` are never
touched. Re-running skips work already recorded, so an interrupted run resumes
by repeating the command.

**Time limits.** `--time-limit` caps each solver subproblem; `0`, the default,
means none. The shipped results were produced as follows, and a rerun matches
them only with the same settings:

| results | time limit per subproblem |
| --- | --- |
| `formulation.csv`, `milp_n6.csv`, `milp_n78.csv`, `milp_xval.csv` | none |
| `duel.csv`: the 402 tasks run first | none |
| `duel.csv`: the remaining 174 tasks (`n ≥ 6`) | 300 s |
| `cpsat_table.csv` | 300 s |

A time-limited task that does not finish is recorded with status `FEASIBLE`,
not `OPTIMAL`, and never enters a table of optima.

**Shipped files.** Each file in `data/` holds the results of the task list of
the same name, with two exceptions: `dp.csv` holds `dp_big.txt` together with
the earlier small-board runs of the dynamic program, and `brute_small.csv`
holds the exhaustive search for `n ≤ 5`, which has no task list. `milp_xval.csv`
was produced on the desktop machine with Gurobi 13.0.3, the rest on the
workstation. Three lists
were stopped before they finished: `dp_big.txt` (both `n = 12` tasks missing),
`milp_n6.txt` (5 of 12 done) and `milp_n78.txt` (16 of 36 done). No table in
the paper depends on the missing tasks.

| table in the paper | task list |
| --- | --- |
| `tab:results` (optima, `n ≤ 6`) | `brute_n6.txt`, plus `--method brute` for `n ≤ 5` |
| second certificate for `tab:results`, `n ≤ 5` | `milp_xval.txt` |
| `tab:convergence` (square, to `n = 11`) | `dp_big.txt` (shipped as `data/dp.csv`) |
| `tab:lower` (bounds, `n = 12…17`) | `code/overnight_structure.py` |
| `tab:formulation` (big-M vs unary) | `formulation.txt` |
| `tab:cp-vs-ip`, `tab:cp-vs-ip-2` | `duel.txt` |
| (CP-SAT past `n = 6`, not in the paper's tables) | `cpsat_table.txt` |

Then:

```bash
python run.py merge   --out ../rerun/all.csv ../rerun/*/part_*.csv
python run.py table   --csv ../data/all.csv     # LaTeX
python run.py analyse --csv ../data/all.csv     # convergence, family optimality
python run.py duel    --csv ../data/all.csv     # tab:cp-vs-ip and tab:cp-vs-ip-2
```

The last three read the shipped results; point them at `../rerun/all.csv` to
check a rerun against them. `tab:lower` comes from `overnight_structure.py`,
whose output is shipped as `data/overnight_structure.log`, and the preliminary
solver comparison from `bench_solvers.py`, shipped as `data/bench_solvers.csv`.
```

**Worker counts.** Exhaustive search streams large arrays and is limited by
memory bandwidth, not cores: beyond ~24 workers it gets slower, not faster.
Solver tasks are compute-bound — give those fewer workers and more `--threads`.

## Two modelling decisions

Both changed published numbers, so they are stated rather than buried.

**The projective plane is not 8-regular.** Under antipodal identification each
of the four corner cells folds onto *itself* through one of its eight offsets,
and two further offsets fold onto the same opposite corner: the adjacency is a
multigraph with a loop and a double edge at each corner. This is intrinsic to
the square model of `RP²`, not an artefact. `corner_policy` selects the
convention — `simple` (default; distinct cells, self excluded, corners of degree
6), `multiplicity` (degree 7), `raw` (degree 8). They give different optima.

**The Möbius seam reflects the *neighbour's* row index**, not the cell's.
Reflecting the cell's own index collapses three offsets onto one cell and counts
it three times; the Möbius band and the cylinder must have the same degree
profile, and any implementation where they do not is wrong. The gluing is
checked against the cylinder double cover in the test suite, which is an
independent derivation.

## Two performance notes

Measurements in `BENCHMARKS.md`.

**Exhaustive search is cache-bound, not flop-bound.** The inner loop touches
~290 temporaries per batch, so `batch` governs whether they stay in L2: 32 K
boards per batch runs at 1.34e6 boards/s against 4.5e5 at 2 M — a factor of
three from a parameter that looks like it should not matter.

**Chunks are contiguous, not strided.** Splitting by `board % chunks == chunk`
makes every chunk walk the whole index space and discard the rest. **Results
from strided and contiguous runs cannot be merged** — the union of their chunks
does not cover the space, and the merged answer is silently wrong.

## Reading the data

`data/all.csv` merges every run. Columns: `method`, `topology` (`s c t m k p`),
`n`, `T`, `m` (initial population), `rate`, `final_cells`, `status`, `seconds`,
`board` (base64 seed), and solver-specific fields.

**Only rows with `method` in `{dp, brute, milp, cpsat}` and `status = OPTIMAL`
are certified optima.** Rows from `milp-m`, `cpsat-m`, `root-lp`, `root-mip` and `duel-*`
are per-subproblem records, LP relaxations or benchmark entries; a `rate` there
is not `G_T(n)`. `run.py analyse` enforces this.

Seeds decode with:

```python
from gol import simulate as sim
x = sim.decode(11, 'ACSW2ADJK2wAZJW2ADJS2A==')   # the optimal 11x11 seed
print(sim.to_string(11, x))
```

## Layout

```
code/gol/topology.py    the six gluings, adjacency, structural invariants
code/gol/simulate.py    dynamics on bitmask boards, encoding, TikZ output
code/gol/bounds.py      certified bounds, the block family
code/gol/symmetry.py    verified automorphism subgroups, symmetry breaking
code/gol/milp.py        Gurobi models: fixed-m and ratio, big-M and unary
code/gol/cpsat.py       CP-SAT model
code/gol/dp.py          transfer-matrix dynamic program
code/gol/brute.py       exhaustive search
code/run.py             CLI: check / tasks / worker / solve / merge / table / analyse / duel
code/reproduce.py       run a task list across local processes
code/tests/test_gol.py  validation suite
code/overnight_structure.py, code/bench_solvers.py   the one-off experiments
```

## License

The code is released under the MIT License; see `LICENSE`. The data in
`data/` and `tasks/` is released under the Creative Commons Attribution 4.0
International License (CC BY 4.0); see `LICENSE-DATA`. If you use either,
please cite the paper.
