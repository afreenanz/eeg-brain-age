"""Minimal ComBat (Johnson et al., 2007; Fortin et al., 2018) for feature tables.

Removes per-site location/scale effects while protecting covariates (age).
Written in-house because neuroHarmonize pins numpy<2 (breaks MNE) and its apply
step fails on single-site batches - the exact case of a new subject in the app.
"""
from __future__ import annotations

import numpy as np


class ComBat:
    def __init__(self, eb: bool = True):
        self.eb = eb

    @staticmethod
    def _design(age: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(age)), age])

    def fit(self, X: np.ndarray, site: np.ndarray, age: np.ndarray) -> "ComBat":
        X, site, age = np.asarray(X, float), np.asarray(site), np.asarray(age, float)
        self.sites_ = np.unique(site)
        n, p = X.shape
        S = np.column_stack([(site == s).astype(float) for s in self.sites_])
        C = self._design(age)[:, 1:]
        D = np.column_stack([S, C])
        B = np.linalg.lstsq(D, X, rcond=None)[0]
        w = S.sum(0) / n
        self.grand_mean_ = w @ B[: len(self.sites_)]
        self.beta_cov_ = B[len(self.sites_):]
        resid = X - D @ B
        self.var_pooled_ = (resid ** 2).mean(0) + 1e-12
        Z = self._standardize(X, age)

        self.gamma_, self.delta_ = {}, {}
        for j, s in enumerate(self.sites_):
            Zs = Z[site == s]
            g_hat, d_hat = Zs.mean(0), Zs.var(0, ddof=1) + 1e-12
            if self.eb:
                g_hat, d_hat = self._eb(Zs, g_hat, d_hat)
            self.gamma_[s], self.delta_[s] = g_hat, d_hat
        return self

    def _standardize(self, X, age):
        mean = self.grand_mean_ + self._design(age)[:, 1:] @ self.beta_cov_
        return (X - mean) / np.sqrt(self.var_pooled_)

    @staticmethod
    def _eb(Zs, g_hat, d_hat, tol=1e-4, max_iter=500):
        """Parametric empirical Bayes shrinkage of site parameters across features."""
        n = len(Zs)
        g_bar, t2 = g_hat.mean(), g_hat.var(ddof=1)
        m, s2 = d_hat.mean(), d_hat.var(ddof=1)
        a = (2 * s2 + m ** 2) / s2
        b = (m * s2 + m ** 3) / s2
        g_new, d_new = g_hat.copy(), d_hat.copy()
        for _ in range(max_iter):
            g_upd = (n * t2 * g_hat + d_new * g_bar) / (n * t2 + d_new)
            ssr = ((Zs - g_upd) ** 2).sum(0)
            d_upd = (b + 0.5 * ssr) / (n / 2 + a - 1)
            done = max(np.abs(g_upd - g_new).max(), np.abs(d_upd - d_new).max()) < tol
            g_new, d_new = g_upd, d_upd
            if done:
                break
        return g_new, d_new

    def transform(self, X: np.ndarray, site: np.ndarray, age: np.ndarray) -> np.ndarray:
        X, site, age = np.asarray(X, float), np.asarray(site), np.asarray(age, float)
        Z = self._standardize(X, age)
        out = np.empty_like(Z)
        for s in np.unique(site):
            if s not in self.gamma_:
                raise ValueError(f"Site '{s}' was not seen when fitting ComBat.")
            m = site == s
            out[m] = (Z[m] - self.gamma_[s]) / np.sqrt(self.delta_[s])
        mean = self.grand_mean_ + self._design(age)[:, 1:] @ self.beta_cov_
        return out * np.sqrt(self.var_pooled_) + mean
