"""Maximum growth rate of Conway's Game of Life on six surfaces.

Modules
-------
topology   the six gluings, their adjacency, and structural invariants
simulate   reference dynamics on bitmask boards, encoding, LaTeX output
bounds     certified upper and lower bounds on the optimal growth rate
symmetry   verified automorphism subgroups and valid symmetry breaking
milp       Gurobi models (fixed-m and ratio; big-M and unary encodings)
dp         transfer-matrix dynamic program for T = 1
brute      exhaustive search, the validation baseline
"""
__all__ = ['topology', 'simulate', 'bounds', 'symmetry', 'milp', 'dp', 'brute']
__version__ = '0.1.0'
