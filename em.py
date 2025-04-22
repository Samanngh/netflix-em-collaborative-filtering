from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    K, _ = mixture.mu.shape
    log_resp = np.zeros((n, K))

    for i in range(n):
        observed = X[i] != 0
        if not np.any(observed):
            log_resp[i] = np.log(mixture.p + 1e-16)
            continue

        x_obs = X[i, observed]
        for j in range(K):
            mu_obs = mixture.mu[j, observed]
            var_j = mixture.var[j]
            p_j = mixture.p[j]

            squared_error = np.sum((x_obs - mu_obs) ** 2)
            d_obs = np.sum(observed)

            log_prob = -0.5 * squared_error / var_j
            log_prob -= 0.5 * d_obs * np.log(2 * np.pi * var_j)
            log_prob += np.log(p_j + 1e-16)

            log_resp[i, j] = log_prob

    log_sum = logsumexp(log_resp, axis=1, keepdims=True)
    responsibilities = np.exp(log_resp - log_sum)
    log_likelihood = np.sum(log_sum)

    return responsibilities, log_likelihood

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = 0.25) -> GaussianMixture:
    n, d = X.shape
    K = post.shape[1]
    n_k = np.sum(post, axis=0)
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        for l in range(d):
            mask = X[:, l] != 0
            weights = post[mask, j]
            values = X[mask, l]
            if np.sum(weights) >= 1.0:
                mu[j, l] = np.sum(weights * values) / np.sum(weights)
            else:
                mu[j, l] = mixture.mu[j, l]

        squared_error = 0.0
        count = 0
        for i in range(n):
            obs = X[i] != 0
            if np.any(obs):
                diff = X[i, obs] - mu[j, obs]
                squared_error += post[i, j] * np.sum(diff ** 2)
                count += post[i, j] * np.sum(obs)

        var[j] = max(squared_error / count, min_variance) if count > 0 else min_variance

    p = n_k / n
    return GaussianMixture(mu, var, p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    prev_ll = None
    threshold = 1e-6
    max_iter = 100

    for _ in range(max_iter):
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

        if prev_ll is not None and abs(ll - prev_ll) <= threshold * abs(ll):
            break
        prev_ll = ll

    return mixture, post, ll

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    n, d = X.shape
    K, _ = mixture.mu.shape
    log_post = np.zeros((n, K))

    for k in range(K):
        diff = X - mixture.mu[k]
        observed = (X != 0)
        n_obs = np.sum(observed, axis=1)
        sq_norm = np.sum(((diff ** 2) * observed), axis=1)
        log_prob = -0.5 * sq_norm / mixture.var[k]
        log_prob -= (n_obs / 2) * np.log(2 * np.pi * mixture.var[k])
        log_prob += np.log(mixture.p[k] + 1e-16)
        log_post[:, k] = log_prob

    log_norm = logsumexp(log_post, axis=1, keepdims=True)
    post = np.exp(log_post - log_norm)

    X_pred = X.copy()
    for i in range(n):
        for j in range(d):
            if X[i, j] == 0:
                X_pred[i, j] = np.dot(post[i], mixture.mu[:, j])

    return X_pred
