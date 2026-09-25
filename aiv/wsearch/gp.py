"""Composite-kernel Gaussian process over candidate regions.

Two corrections separate this from a default surrogate. The loss-geometry kernel
is cosine-normalized, so prior variance does not track how concentrated a region
happens to be in leaf space, and the amplitude is carried by a single factor
outside the mixing weight, so a strong loss-geometry correlation at high signal
amplitude is representable. The likelihood uses the overlap-induced covariance of
the measured contrasts rather than an independent-noise term, because candidates
built by an inflation rule share members and therefore share sampling error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

_JITTER = 1e-8


def matern52(A: np.ndarray, B: np.ndarray, lengthscale) -> np.ndarray:
    """Matern 5/2 with per-coordinate lengthscales and unit diagonal."""
    ls = np.asarray(lengthscale, dtype=float)
    d = (A[:, None, :] - B[None, :, :]) / ls
    r = np.sqrt(np.maximum((d * d).sum(-1), 0.0))
    s = np.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


def cosine_kernel(E: np.ndarray, F: np.ndarray | None = None) -> np.ndarray:
    """Cosine-normalized inner product of region embeddings.

    The raw inner product has a diagonal equal to the squared embedding norm,
    which measures how few leaves a region occupies: a compact region would
    receive a large prior variance and a diffuse one a small variance, so the
    surrogate's uncertainty would track region compactness rather than anything
    about weakness. Normalizing fixes the diagonal at one and leaves a valid
    kernel, being a Gram matrix of unit vectors.
    """
    F = E if F is None else F
    ne = np.linalg.norm(E, axis=1, keepdims=True)
    nf = np.linalg.norm(F, axis=1, keepdims=True)
    ne[ne == 0] = 1.0
    nf[nf == 0] = 1.0
    return (E / ne) @ (F / nf).T


def lengthscale_groups(n_center: int, n_shape: int) -> np.ndarray:
    """One lengthscale for the center block and one for the shape block.

    Full per-coordinate lengthscales cost a numerical gradient of the marginal
    likelihood in as many dimensions as the candidate has coordinates, which
    dominates the fit. Two groups keep the anisotropy that matters, between
    moving a center and reshaping the faces, at a fraction of the optimizer cost.
    """
    return np.concatenate([np.zeros(n_center, dtype=int), np.ones(n_shape, dtype=int)])


@dataclass
class CompositeGP:
    """GP over candidate coordinates with a loss-informed composite kernel.

    `lam` is the mixing weight between the Matern component and the loss
    geometry; `lam = 1` is the Matern-only ablation and `lam = 0` the
    loss-geometry-only ablation. `groups` assigns each candidate coordinate to a
    lengthscale, defaulting to one shared lengthscale; pass
    `lengthscale_groups(p, p)` to separate the center from the shape, or
    `np.arange(d)` for full anisotropy. `noise` accepts a covariance matrix, but
    the discovery response is deterministic and the nugget is misspecification
    rather than measurement error, so the default fitted term is the right one.
    """

    lam: float = 0.5
    groups: np.ndarray = None
    lengthscale: np.ndarray = None
    sigma2: float = 1.0
    jitter: float = _JITTER
    mean: float = 0.0
    fit_noise_scale: bool = True
    noise_scale: float = 1.0
    _X: np.ndarray = field(default=None, repr=False)
    _E: np.ndarray = field(default=None, repr=False)
    _L: tuple = field(default=None, repr=False)
    _alpha: np.ndarray = field(default=None, repr=False)
    _noise: np.ndarray = field(default=None, repr=False)
    _kloss: np.ndarray = field(default=None, repr=False)

    # -- kernel -------------------------------------------------------------

    def _k(self, Xa, Ea, Xb=None, Eb=None, kloss=None) -> np.ndarray:
        """Composite covariance. `kloss` accepts the cached loss Gram, which does
        not depend on any hyperparameter and so is computed once per fit."""
        Xb = Xa if Xb is None else Xb
        Eb = Ea if Eb is None else Eb
        K = np.zeros((len(Xa), len(Xb)))
        if self.lam > 0:
            K += self.lam * matern52(Xa, Xb, self.lengthscale)
        if self.lam < 1:
            K += (1.0 - self.lam) * (cosine_kernel(Ea, Eb) if kloss is None else kloss)
        return self.sigma2 * K

    # -- fitting ------------------------------------------------------------

    def fit(self, X: np.ndarray, E: np.ndarray, y: np.ndarray,
            noise: np.ndarray | None = None) -> "CompositeGP":
        X = np.asarray(X, float)
        E = np.asarray(E, float)
        y = np.asarray(y, float)
        if self.groups is None:
            self.groups = np.zeros(X.shape[1], dtype=int)
        self.groups = np.asarray(self.groups, dtype=int)
        self._X, self._E, self._y = X, E, y
        self._noise = None if noise is None else np.asarray(noise, float)
        self._kloss = None if self.lam >= 1 else cosine_kernel(E)
        self.mean = float(y.mean())

        # With lam = 0 the Matern component is never evaluated, so its
        # lengthscales are not free parameters and are left out of the search.
        n_g = 0 if self.lam <= 0 else int(self.groups.max()) + 1
        p0 = np.concatenate([np.zeros(n_g), [np.log(max(y.var(), 1e-6))],
                             [np.log(self.noise_scale)]])
        bounds = [(-3.0, 4.0)] * n_g + [(-12.0, 8.0), (-6.0, 6.0)]
        res = minimize(self._nll, p0, args=(X, E, y), method="L-BFGS-B", bounds=bounds)
        self._unpack(res.x)
        self._factor(X, E, y)
        self.nll = float(res.fun)
        self.n_nll_calls = int(res.nfev)
        return self

    def _unpack(self, params) -> None:
        n_g = len(params) - 2
        self.lengthscale = (np.ones(len(self.groups)) if n_g == 0
                            else np.exp(params[:n_g])[self.groups])
        self.sigma2 = float(np.exp(params[n_g]))
        self.noise_scale = float(np.exp(params[n_g + 1]))

    def _noise_matrix(self, n: int) -> np.ndarray:
        if self._noise is None:
            return self.noise_scale * np.eye(n)
        scale = self.noise_scale if self.fit_noise_scale else 1.0
        return scale * self._noise

    def _nll(self, params, X, E, y) -> float:
        self._unpack(params)
        K = self._k(X, E, kloss=self._kloss) + self._noise_matrix(len(X)) + self.jitter * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        d = y - self.mean
        a = cho_solve(c, d)
        return float(0.5 * d @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(y) * np.log(2 * np.pi))

    def _factor(self, X, E, y) -> None:
        K = self._k(X, E, kloss=self._kloss) + self._noise_matrix(len(X)) + self.jitter * np.eye(len(X))
        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, y - self.mean)

    # -- prediction ---------------------------------------------------------

    def predict(self, Xs: np.ndarray, Es: np.ndarray, noise_var=None):
        """Posterior mean and standard deviation at new candidates.

        `noise_var` switches from the latent function to a new observation. It
        is required whenever the prediction is compared against a measured
        contrast, because a measurement carries its own sampling error: pass the
        diagonal of the overlap covariance at the test candidates, or None to
        use the fitted independent term. Scoring measured values against latent
        intervals understates the spread and makes a surrogate that has
        attributed most variation to noise look badly calibrated when it is not.
        """
        Xs, Es = np.asarray(Xs, float), np.asarray(Es, float)
        Ks = self._k(self._X, self._E, Xs, Es)
        mu = self.mean + Ks.T @ self._alpha
        v = cho_solve(self._L, Ks)
        kss = self.sigma2 * (self.lam + (1.0 - self.lam))  # both components have unit diagonal
        var = np.maximum(kss - (Ks * v).sum(0), 1e-12)
        if noise_var is not None:
            scale = self.noise_scale if self.fit_noise_scale else 1.0
            var = var + scale * np.asarray(noise_var, float)
        return mu, np.sqrt(var)

    def predictive(self, Xs: np.ndarray, Es: np.ndarray, noise_var=None, noise_cross=None):
        """Predictive distribution for a new measurement at each candidate.

        With correlated observation noise the training measurements carry
        information about the test measurement's error, not only about the
        latent function, so the cross-covariance enters the conditioning. The
        conditional is taken on the joint covariance K + Sigma rather than on K
        alone: ignoring `noise_cross` treats a test candidate that shares most of
        its members with the training candidates as though its error were fresh,
        which inflates the interval by the whole shared component instead of
        leaving only the part the training data has not already seen.

        `noise_cross` is the (training, test) block of the observation covariance.
        Falls back on the fitted independent term when no covariance is supplied.
        """
        Xs, Es = np.asarray(Xs, float), np.asarray(Es, float)
        scale = self.noise_scale if self.fit_noise_scale else 1.0
        if noise_var is None:
            noise_var = np.full(len(Xs), self.noise_scale if self._noise is None else 0.0)
            scale_diag = 1.0
        else:
            noise_var, scale_diag = np.asarray(noise_var, float), scale
        Ks = self._k(self._X, self._E, Xs, Es)
        if noise_cross is not None:
            Ks = Ks + scale * np.asarray(noise_cross, float)
        mu = self.mean + Ks.T @ self._alpha
        v = cho_solve(self._L, Ks)
        kss = self.sigma2 + scale_diag * noise_var
        return mu, np.sqrt(np.maximum(kss - (Ks * v).sum(0), 1e-12))


def expected_improvement(mu, sd, best, xi: float = 0.0) -> np.ndarray:
    """Expected improvement for a maximized objective."""
    mu, sd = np.asarray(mu, float), np.maximum(np.asarray(sd, float), 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def upper_confidence_bound(mu, sd, beta: float = 2.0) -> np.ndarray:
    return np.asarray(mu, float) + beta * np.asarray(sd, float)


def interval_coverage(mu, sd, y, level: float = 0.90) -> float:
    """Fraction of held-out candidates inside the nominal posterior interval.

    Reported under the overlap covariance and under an independent-noise term:
    the argument for modeling overlap is that agreement among candidates sharing
    members is not independent confirmation, and coverage is where that shows.
    """
    z = norm.ppf(0.5 + level / 2)
    mu, sd, y = np.asarray(mu, float), np.asarray(sd, float), np.asarray(y, float)
    return float((np.abs(y - mu) <= z * sd).mean())
