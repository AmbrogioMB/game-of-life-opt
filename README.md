# Maximum growth in the Game of Life on surfaces — code and data

Companion code for *Maximising the growth rate of Conway's Game of Life on
finite boards and surfaces: bounds, constructions and exact methods* (Bernardelli). Given an
`n × n` board and a horizon `T`, which initial configuration maximises
`|X(T)| / |X(0)|`? The board is one of six surfaces obtained by gluing the edges
of the square.

Every reported optimum has a completed exact computation. All 72 small-board
values have enumeration certificates; 71 also have a certificate from a second
method. The remaining value (square, n=6, T=3) has a combined MILP and
fixed-population exhaustive certificate, described below.
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

`requirements.txt` records tested versions. The software used for each
experiment is listed under "Machines and software" below.

## Start here

```bash
cd code
python -m tests.test_gol     # must end ALL TESTS PASSED
python run.py check          # structural self-check and the bound table, no solver
```

`tests/test_gol.py` validates the implementation. It verifies the six
gluings against their structural invariants, checks the Möbius dynamics against
its cylinder double cover, reproduces textbook patterns, confirms the bounds,
checks that symmetry breaking preserves an optimum, and cross-validates the
dynamic program and the integer program against exhaustive search on all six
surfaces.

## The four methods

| module | method | cost | best where |
| --- | --- | --- | --- |
| `gol/brute.py` | exhaustive enumeration | `O(T n² 2^(n²))` | ground truth, `n ≤ 6` |
| `gol/dp.py` | transfer-matrix DP + Dinkelbach | `O(K n^2 2^(3n))` | square and cylinder at `T = 1`, to `n = 11` |
| `gol/milp.py` | integer program (Gurobi) | — | small `T`, all six surfaces |
| `gol/cpsat.py` | constraint program (CP-SAT) | — | large `T`, all six surfaces |

For the implemented DP, `K` is the number of Dinkelbach evaluations (two or
three in the shipped runs); working memory is `O(n 2^(2n))`. It returns the
ratio and population counts, and a seed attaining the ratio (backtracked
through the stored transitions), so the value can be checked by simulation. A
seed confirms the ratio as a lower bound, not its optimality. An iteration-limited
DP result has status `FEASIBLE` until a final evaluation certifies optimality.

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
python reproduce.py ../tasks/dp_small.txt    --workers 1
python reproduce.py ../tasks/duel.txt        --workers 24
python reproduce.py ../tasks/brute_n6.txt    --workers 24
python reproduce.py ../tasks/formulation.txt --workers 8 --threads 2
```

The runner shards each list across local processes, writes `reproduced/<name>/part_k.csv`,
and merges to `reproduced/<name>.csv`; the shipped results in `data/` are never
touched. Re-running skips work already recorded, so an interrupted run resumes
by repeating the command. Each population of a sweep is written as soon as it
is solved, so an interrupted sweep keeps its finished populations on record;
without its terminal summary it is still rerun from the start. Use `run.py worker --retry-feasible` to retry completed
but unresolved sweeps. Merging retains the latest record for a retried task.

**Time limits.** `--time-limit` caps each solver subproblem; `0`, the default,
means none. The shipped results use the following limits. Reproduction requires
matching model settings and limits; runtimes also depend on hardware, software
versions and concurrent load.

| results | time limit per subproblem |
| --- | --- |
| `formulation.csv`, `milp_n6_t3.csv`, `milp_n78.csv` | none |
| `milp_xval.csv`, `milp_n6.csv` | 3600 s |
| `milp_n6_long.csv` | 3600 s, raised to 14400 s for populations still unresolved |
| `duel.csv`: 402 tasks | none |
| `duel.csv`: the other 174 tasks (all with `n ≥ 6`) | 300 s |
| `cpsat_table.csv` | 300 s |

An unresolved sweep is recorded as `FEASIBLE` or `NO_SOLUTION` and does not
enter a table of optima. A solver can also certify an optimum before reaching
its time limit.

**Shipped files.** Each main experiment CSV corresponds to the task list of
the same name in `tasks/`. `all.csv` merges these results; `bench_solvers.csv`
is a separate preliminary benchmark produced by `code/bench_solvers.py`.
Ten of the 32 tasks of `formulation.txt` and 20 of
the 36 tasks of `milp_n78.txt` did not finish. Tables use only completed
subproblems or certified full sweeps; missing populations are not treated as solved.

| table in the paper | task list |
| --- | --- |
| `tab:results` (optima, `n ≤ 6`) | `brute_small.txt`, `brute_n6.txt` |
| second certificate for `tab:results`, `n ≤ 5` | `milp_xval.txt`; `dp_small.txt` for square/cylinder at n=5, T=1 |
| second certificate for `tab:results`, `n = 6` | `milp_n6.txt`, `milp_n6_long.txt`, `milp_n6_t3.txt`; `cpsat_table.txt` at `T = 3` |
| combined certificate, square `n = 6, T = 3` | `milp_n6_long.txt` + `code/exact_m8.cpp`, checked by `code/verify_certificates.py` |
| `tab:convergence` (square, to `n = 11`) | `dp_small.txt`, `dp_large.txt` |
| `tab:lower` (bounds, `n = 12…17`) | `code/overnight_structure.py` |
| `tab:formulation` (big-M vs unary) | `formulation.txt` |
| `tab:cp-vs-ip`, `tab:cp-vs-ip-2` | `duel.txt` |
| (CP-SAT past `n = 6`, not in the paper's tables) | `cpsat_table.txt` |

## Certificate audit and raw output

`data/raw/batch.zip` is the raw output of the serial batch that produced
`brute_small`, `milp_xval`, `milp_n6`, `dp_small`, `dp_large` and `exact_m8.json`:
source manifest, task lists, per-worker result files, logs, environment, run
state and the batch's own audit. Two manifest entries that only configure a
private notification channel (`NTFY_TOPIC.txt`, `README.md`) are withheld.
`data/environment.json`, `data/run.json` and `data/run_audit.json` are copies
of its metadata. `data/raw/milp_n6_long.zip` is the raw output of the long
`n = 6` sweeps in `milp_n6_long.csv`, with `data/milp_n6_long_run.json` its
final run state. Both archives are verbatim apart from the withheld entries;
they include supervisor logs and, in the second, an additional DP run that the
release does not use.

`code/exact_m8.cpp` enumerates every one of the 30,260,340 eight-cell seeds on
the finite square, without symmetry reduction. Their maximum final population
after three steps is 22, less than the 23 required to beat the MILP witness
20/7. Combining this result with the other population proofs establishes the
global optimum 20/7. This C++ enumerator is separate from the NumPy
enumeration. It uses one thread and constant memory; no solver is needed.

To reproduce the search and audit the released certificates (from `code/`):

```bash
mkdir -p ../reproduced
g++ -O3 -std=c++17 exact_m8.cpp -o ../reproduced/exact_m8
python verify_certificates.py --executable ../reproduced/exact_m8 --out ../reproduced/certificates.json
```

On Windows compile to `../reproduced/exact_m8.exe` and pass that executable path.
The verifier reruns the full search and compares it with `data/exact_m8.json`,
checks its simulator against the Python reference on 498 steps, checks both raw
archives against the released CSVs, task lists and metadata, simulates every
returned seed, recomputes every MILP sweep certificate, and checks agreement and
coverage of all 72 enumeration results with the 71 second-method certificates
and the combined proof. The C++ program also checks 60,000 transitions against
its own scalar reference. The shipped report is `data/certificates.json`, with
input hashes. The combined certificate stays outside the MILP summary CSVs and
`all.csv`: the solver summary of that sweep keeps its status `FEASIBLE`.

To regenerate tables from the experiment CSVs:

```bash
python run.py merge   --out ../reproduced/all.csv ../reproduced/*/part_*.csv
python run.py table   --csv ../data/all.csv     # LaTeX
python run.py analyse --csv ../data/all.csv     # convergence, family optimality
python run.py duel    --csv ../data/all.csv     # tab:cp-vs-ip and tab:cp-vs-ip-2
```

The last three read the shipped results; point them at `../reproduced/all.csv` to
check a rerun against them. `tab:lower` comes from `overnight_structure.py`,
whose output is shipped as `data/overnight_structure.log`, and the preliminary
solver comparison from `bench_solvers.py`, shipped as `data/bench_solvers.csv`.

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

**The NumPy exhaustive search is cache-bound.** The inner loop touches
~290 temporaries per batch, so `batch` governs whether they stay in L2: 32 K
boards per batch runs at 1.34e6 boards/s against 4.5e5 at 2 M — a factor of
three from a parameter that looks like it should not matter.

**Chunks are contiguous, not strided.** Splitting by `board % chunks == chunk`
makes every chunk walk the whole index space and discard the rest. **Results
from strided and contiguous runs cannot be merged** — the union of their chunks
does not cover the space, and the merged answer is silently wrong.

## Reading the data

`data/all.csv` merges the experiment CSVs listed in `code/verify_certificates.py`
(`SOURCES`); no two of them share a `(method, task, m)` key.
`bench_solvers.csv` is a separate preliminary benchmark, and the JSON combined
certificate is also separate. Columns: `method`, topology (`s c t m k p`),
`n`, `T`, `m` (initial population), `rate`, `final_cells`, `status`, `seconds`,
`board` (base64 seed), and solver-specific fields.

**Only rows with `method` in `{dp, brute, milp, cpsat}` and `status = OPTIMAL`
are candidates for certified optima.** A `brute` row with `chunks > 1` certifies
only its chunk: a global optimum requires all chunk indices for the same board,
horizon and surface, using the same partition convention, and taking their
maximum rate. Missing chunks do not certify a global optimum. Rows from `milp-m`, `cpsat-m`, `root-lp`, `root-mip` and `duel-*`
are per-subproblem records, LP relaxations or benchmark entries; a `rate` there
is not `G_T(n)`. `run.py table` and `run.py analyse` leave out, with a
warning, any chunked enumeration whose partition is incomplete.

The `status` of a `milp` or `cpsat` summary row is `OPTIMAL` only if every
population of its sweep is settled: solved to optimality, proved unable to
improve on the best rate, or excluded by the cap `min(n^2, 3^T m, (2T+1)^2 m)`.
Otherwise it is `FEASIBLE`, the rate is a lower bound, and `bound` holds an
upper bound on `G_T(n)`.

Three things to know when using the timings:

- **The `seconds` of a `milp` summary row.** In `milp_xval.csv`, `milp_n6.csv`
  and `milp_n6_long.csv` it is the total over the sweep. In `milp_n6_t3.csv` and
  `milp_n78.csv` it is the time of the optimal subproblem only; for the cost of a
  whole sweep over `m`, sum the `milp-m` rows with the same `task`.
- **`formulation.csv` does not cover every population.** The paper's
  formulation table uses the 190 subproblems that all three formulations solved.
  The unary formulation ignores the `cuts` flag, so its `cuts=0` and `cuts=1`
  rows describe the same model.
- **Machines and software.** Every certificate and timing in the paper comes
  from a Linux workstation (Intel Xeon w9-3495X, 56 cores / 112 threads,
  128 GB), in three configurations:
  (a) `brute_small`, `milp_xval`, `milp_n6`, `dp_small`, `dp_large` and
  `exact_m8.json`: Python 3.10.12, numpy 2.2.6, Gurobi 13.0.3, g++ 13.4.0; jobs
  run one at a time with one solver and numerical-library thread and a 3 GiB
  address-space limit (`data/environment.json`, `data/run.json`);
  (b) `milp_n6_long`: numpy 2.2.6 and Gurobi 13.0.3, at most two processes of
  two threads each, 3 GiB per process;
  (c) all other CSVs: Python 3.10.12, numpy 1.26.2, Gurobi 11.0.0 and OR-Tools
  9.15.6755, one thread per subproblem unless the task says otherwise.
  In configuration (a) the square n=6 DP took 0.128 s against 12.278 s for the
  MILP sweep, about 96 times as long; the square DP took 1244.561 s at n=10 and
  14527.747 s at n=11. These are single runs, not repeated timing trials. The
  lower-bound search in `overnight_structure.log` and the measurements in
  `BENCHMARKS.md` were made on a desktop (Intel Core i7-1185G7, 16 GB); the
  lower-bound witnesses are explicit seeds checked by simulation.

Seeds decode with:

```python
from gol import simulate as sim
x = sim.decode(11, 'ADaUmADbUkwAbakmADbSSA==')   # an optimal 11x11 seed from data/dp_large.csv
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
code/exact_m8.cpp      exhaustive fixed-population search, square n=6, T=3, m=8
code/verify_certificates.py   rerun the m=8 search and audit all 72 small-board certificates
code/overnight_structure.py, code/bench_solvers.py   the one-off experiments
```

## License

The code is released under the MIT License; see `LICENSE`. The data in
`data/` and `tasks/` is released under the Creative Commons Attribution 4.0
International License (CC BY 4.0); see `LICENSE-DATA`. If you use either,
please cite the paper.
