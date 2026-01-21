We developed a low-complexity, rank-one Riemannian subspace descent algorithm suitable for high-dimensional, dense SPD-constrained matrix equations such as CARE and DARE, which arise in control theory,
dynamic programming, and stochastic filtering. The proposed approach has a per-iteration computational cost of O(n^2). In contrast, standard methods for solving dense n×n matrix equations typically 
incur a per-iteration cost of  O(n^3), while existing  O(n^2) methods either rely on sparsity or low-rank structure, or suffer from iteration counts that scale poorly with problem size.
