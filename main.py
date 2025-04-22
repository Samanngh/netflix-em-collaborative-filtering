import numpy as np
import matplotlib.pyplot as plt
import kmeans
import common
import naive_em
import em
from kmeans import run

X = np.loadtxt("/Users/saman/Desktop/Project 4/netflix/toy_data.txt")

# TODO: Your code here
import numpy as np
import matplotlib.pyplot as plt
from kmeans import run
import common

Ks = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]

for K in Ks:
    best_cost = float("inf")
    best_mixture = None
    best_post = None
    best_seed = None

    for seed in seeds:
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = run(X, mixture, post)

        if cost < best_cost:
            best_cost = cost
            best_mixture = mixture
            best_post = post
            best_seed = seed

    print(f"Lowest cost for K={K}: {best_cost:.2f} (Seed: {best_seed})")

    # Plot the best result
    common.plot(X, best_mixture, best_post, title=f"K={K}, cost={best_cost:.2f}")
    plt.savefig(f"kmeans_K{K}.png")


print("\n--- Running Naive EM + BIC ---\n")

best_overall_K = None
best_overall_bic = -np.inf

for K in Ks:
    best_likelihood = -np.inf
    best_mixture = None
    best_post = None
    best_seed = None

    for seed in seeds:
        mixture, post = common.init(X, K, seed)
        mixture, post, log_likelihood = naive_em.run(X, mixture, post)

        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_mixture = mixture
            best_post = post
            best_seed = seed

    bic_score = common.bic(X, best_mixture, best_likelihood)
    print(f"K={K} | Best Log-Likelihood: {best_likelihood:.2f} | BIC: {bic_score:.2f} (Seed: {best_seed})")

    if bic_score > best_overall_bic:
        best_overall_bic = bic_score
        best_overall_K = K

    common.plot(X, best_mixture, best_post, title=f"Naive EM K={K}, BIC={bic_score:.2f}")
    plt.savefig(f"naive_em_K{K}_bic.png")

print(f"\n✅ Best K = {best_overall_K}")
print(f"✅ Best BIC = {best_overall_bic:.2f}")




import numpy as np
from em import run
from common import init, GaussianMixture
from pathlib import Path

# Load the Netflix incomplete dataset
X = np.loadtxt("/Users/saman/Desktop/Project 4/netflix/netflix_incomplete.txt")

Ks = [1, 12]
seeds = [0, 1, 2, 3, 4]

best_mixture_k12 = None  # ✅ Define here
best_ll_k12 = -np.inf

for K in Ks:
    best_ll = -np.inf
    best_seed = None
    best_mixture = None

    for seed in seeds:
        mixture, post = init(X, K, seed)
        mixture, post, ll = run(X, mixture, post)

        if ll > best_ll:
            best_ll = ll
            best_seed = seed
            best_mixture = mixture

    print(f"Best Log-Likelihood for K={K}: {best_ll:.2f} (Seed: {best_seed})")

    if K == 12:
        best_mixture_k12 = best_mixture  # ✅ Save correctly



