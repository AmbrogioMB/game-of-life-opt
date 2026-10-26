# Enumeration baseline: measured cost

All figures from one core of the development desktop, task
`brute n=6 T=1 top=s chunk=0/64` (~7.9e8 boards after filtering).

| version | per task | speedup | whole n=6 job (1152 tasks) |
| --- | --- | --- | --- |
| striding chunks, 2^21 batch | 1917.1 s | — | ~600 core-hours |
| contiguous chunks, 2^15 batch | **631.7 s** | **3.0x** | **202 core-hours** |

Two independent changes, both needed:

**Contiguous chunks.** Partitioning by `board % chunks == chunk` made every
chunk walk the entire 2^36 index space and discard all but 1/64 of it, so the
scan was repeated `chunks` times over the whole job. Contiguous slices generate
each board index exactly once.

**Batch size, which matters far more than it looks.** The inner loop touches
~290 temporaries per batch, so throughput is governed by whether those stay in
cache, not by arithmetic:

| batch | array size | boards/sec |
| --- | --- | --- |
| 8 192 | 0.1 MB | 1.35e6 |
| 32 768 | 0.3 MB | 1.34e6 |
| 131 072 | 1.0 MB | 5.2e5 |
| 2 097 152 | 16.8 MB | 4.5e5 |

Below ~0.3 MB the temporaries are L2-resident; above it every one of the 290
passes goes to memory. Note the interaction: the striding version was
accidentally protected, because filtering to 1/64 left 32 K-element arrays that
happened to fit. Fixing the chunking *alone* made things slower, since it
restored 2 M-element batches. Both changes together give the 3x.

For the paper: the enumeration baseline is memory-hierarchy bound, not
flop bound, so quoting it as an operation count alone would misstate it in
either direction.

Correctness of the chunking is checked in `tests/test_gol.py` and directly:
chunked equals unchunked on all six surfaces for n=4 (T=1,2) and n=5 (T=1,2),
at K=3 and K=8.

## Caution when mixing runs

Striding and contiguous chunks partition the board space differently. Results
from the two **cannot be combined** -- the union of their chunks does not cover
the space, and the merged answer would be silently wrong. If switching versions,
delete `results/brute_n6/` and restart that job from zero.

The run of 2026-09-23 used the striding version throughout, so it is internally
consistent and its results are valid.
