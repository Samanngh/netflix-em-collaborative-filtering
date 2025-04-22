import numpy as np
from em import run, fill_matrix
from common import init, rmse

# Load data
X_incomplete = np.loadtxt("/Users/saman/Desktop/Project 4/netflix/netflix_incomplete.txt")
X_gold = np.loadtxt("/Users/saman/Desktop/Project 4/netflix/netflix_complete.txt")

# Re-run EM to get best_mixture_k12
K = 12
best_ll = -np.inf
best_mixture_k12 = None
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    mixture, post = init(X_incomplete, K, seed)
    mixture, post, ll = run(X_incomplete, mixture, post)
    if ll > best_ll:
        best_ll = ll
        best_mixture_k12 = mixture

# Predict completed matrix
X_pred = fill_matrix(X_incomplete, best_mixture_k12)

# RMSE on missing entries only
mask_missing = (X_incomplete == 0)
rmse_missing = rmse(X_gold[mask_missing], X_pred[mask_missing])

# RMSE on all entries
rmse_all = rmse(X_gold, X_pred)

# Print both
print(f"✅ RMSE on missing entries only: {rmse_missing:.10f}")
print(f"✅ RMSE on all entries:         {rmse_all:.10f}")
