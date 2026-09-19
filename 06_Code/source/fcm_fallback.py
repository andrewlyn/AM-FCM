"""Minimal fuzzy-c-means fallback used only when fcmeans is unavailable.

The update equations match fuzzy-c-means 2.0.2's default Euclidean FCM.
"""

from __future__ import annotations

import numpy as np

class FCM:
    def __init__(self, n_clusters=5, max_iter=150, m=2.0,
                 error=1e-5, random_state=None, **kwargs):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.m = float(m)
        self.error = float(error)
        self.random_state = random_state
        self.trained = False

    @staticmethod
    def _dist(A, B):
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.u = rng.uniform(size=(n_samples, self.n_clusters))
        self.u /= self.u.sum(axis=1, keepdims=True)
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            um = self.u ** self.m
            self._centers = (X.T @ um / np.sum(um, axis=0)).T
            temp = self._dist(X, self._centers) ** (2 / (self.m - 1))
            self.u = 1.0 / (temp * (1.0 / temp).sum(axis=1, keepdims=True))
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True

    @property
    def centers(self):
        if not self.trained:
            raise ReferenceError("FCM has not been fitted.")
        return self._centers
