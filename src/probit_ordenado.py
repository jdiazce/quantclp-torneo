"""
src/probit_ordenado.py — Probit ordenado estimado por máxima verosimilitud (EMV).

Modelo:  y* = X·β + ε,  ε ~ N(0,1);   y = j  si  κ_{j-1} < y* <= κ_j
         P(y=j | X) = Φ(κ_j − Xβ) − Φ(κ_{j-1} − Xβ)

Los cortes se parametrizan como κ_1, κ_1 + exp(δ_2), ... para que queden
ordenados sin restricciones. Sin constante en X (la absorben los cortes).
Solo depende de numpy y scipy.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class ProbitOrdenado:
    def __init__(self, n_cat):
        self.J = n_cat

    def _cortes(self, theta):
        k = np.empty(self.J - 1)
        k[0] = theta[0]
        k[1:] = theta[0] + np.cumsum(np.exp(theta[1:]))
        return k

    def _probs(self, X, beta, kap):
        xb = X @ beta
        lim = np.concatenate([[-np.inf], kap, [np.inf]])
        cdf = norm.cdf(lim[None, :] - xb[:, None])
        return np.clip(np.diff(cdf, axis=1), 1e-12, 1.0)

    def _nll(self, par, X, y):
        p = X.shape[1]
        P = self._probs(X, par[:p], self._cortes(par[p:]))
        return -np.log(P[np.arange(len(y)), y]).sum()

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        p = X.shape[1]
        frec = np.bincount(y, minlength=self.J) / len(y)
        k0 = norm.ppf(np.clip(np.cumsum(frec)[:-1], 1e-3, 1 - 1e-3))
        th0 = np.concatenate([[k0[0]], np.log(np.maximum(np.diff(k0), 1e-3))])
        x0 = np.concatenate([np.zeros(p), th0])
        r = minimize(self._nll, x0, args=(X, y), method="BFGS",
                     options={"maxiter": 2000, "gtol": 1e-6})
        self.par_, self.ok_ = r.x, r.success
        self.beta_ = r.x[:p]
        self.kap_ = self._cortes(r.x[p:])
        Hs = self._hessiano(r.x, X, y)
        cov = np.linalg.pinv(Hs)
        self.se_beta_ = np.sqrt(np.clip(np.diag(cov)[:p], 0, None))
        self.loglik_ = -r.fun
        # pseudo-R2 de McFadden contra el modelo solo con cortes
        ll0 = (np.bincount(y, minlength=self.J) * np.log(np.clip(frec, 1e-12, 1))).sum()
        self.r2_mcfadden_ = 1 - self.loglik_ / ll0
        return self

    def _hessiano(self, x, X, y, h=1e-4):
        """Hessiano numérico (diferencias centrales) de la log-verosimilitud negativa."""
        n = len(x)
        Hs = np.empty((n, n))
        f = lambda z: self._nll(z, X, y)
        for i in range(n):
            for j in range(i, n):
                ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
                v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
                Hs[i, j] = Hs[j, i] = v
        return Hs

    def predict_proba(self, X):
        return self._probs(np.asarray(X, float), self.beta_, self.kap_)
