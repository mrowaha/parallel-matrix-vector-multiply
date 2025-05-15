import numpy as np
from sklearn.datasets import make_spd_matrix

n = 1024
seed = 5

A = make_spd_matrix(n_dim=n, random_state=seed)
b = np.random.rand(n)

np.savetxt(f"A_matrix.txt", A, fmt="%.6f")
np.savetxt(f"b_vector.txt", b, fmt="%.6f")

print("Generated SPD matrix A and vector b:")
print("A shape:", A.shape)
print("b shape:", b.shape)
