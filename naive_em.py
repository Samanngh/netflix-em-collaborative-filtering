"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    K, _ = mixture.mu.shape
    responsibilities = np.zeros((n, K))
    log_likelihood = 0.0

    for k in range(K):
        diff = X - mixture.mu[k]
        norm_sq = np.sum(diff ** 2, axis=1)
        coef = 1.0 / ((2 * np.pi * mixture.var[k]) ** (d / 2))
        exponent = np.exp(-norm_sq / (2 * mixture.var[k]))
        responsibilities[:, k] = mixture.p[k] * coef * exponent

    total = np.sum(responsibilities, axis=1, keepdims=True)
    log_likelihood = np.sum(np.log(total))
    responsibilities /= total

    return responsibilities, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    n, d = X.shape
    n_k = np.sum(post, axis=0)  # shape (K,)
    K = post.shape[1]

    # New means
    mu = (post.T @ X) / n_k[:, np.newaxis]  # shape (K, d)

    # New variances (assuming spherical covariance: scalar per component)
    var = np.zeros(K)
    for k in range(K):
        diff = X - mu[k]
        var[k] = np.sum(post[:, k] * np.sum(diff**2, axis=1)) / (n_k[k] * d)

    # New mixing coefficients
    p = n_k / n

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    prev_likelihood = None
    threshold = 1e-6
    max_iter = 100

    for _ in range(max_iter):
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

        if prev_likelihood is not None and abs(log_likelihood - prev_likelihood) <= threshold * abs(log_likelihood):
            break
        prev_likelihood = log_likelihood

    return mixture, post, log_likelihood
