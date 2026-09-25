"""Population conflict under limited expert capacity.

The existing synthetic benchmark (``synthetic_population_data``) separates routing
variables from outcome variables, and gives each latent population its own expert
and its own disjoint pair of predictors. A mixture with as many experts as
populations can therefore fit every population at once, so mean loss and
worst-population loss are minimised by the same model and no trade-off exists.
That is why its worst-group arm does not improve worst-population loss.

This generator supplies the missing mechanism. All populations share the same two
outcome signals and differ only in the direction of their coefficient vectors,
placed at 120 degrees to each other so that no single logistic expert can serve
two populations without loss. When the mixture has fewer experts than populations,
at least two populations must share an expert, the shared expert tilts toward the
larger of them, and the smaller one is sacrificed. Which population is sacrificed
is what the mean and worst-population objectives disagree about.

Two further properties are kept from the earlier generator. Signals are identically
distributed in every population, so the populations cannot be recovered from signal
geometry, and each population has event rate one half in expectation, so context
carries no marginal outcome association while deciding which relationship applies.

Coefficient norms set each population's Bayes risk. With equal norms all
populations are equally predictable and worst-population loss measures misfit
alone. With unequal norms one population is intrinsically noisier, its loss floor
is higher, and a raw worst-population objective chases a floor no model can lower;
``bayes_losses`` gives the floors so that excess risk can be reported instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

CONTEXT_FEATURES = ("context_a", "context_b")
SIGNAL_FEATURES = ("signal_1", "signal_2")
NOISE_FEATURES = ("noise_1", "noise_2")
ALL_FEATURES = CONTEXT_FEATURES + SIGNAL_FEATURES + NOISE_FEATURES

PRIORS = np.array([0.60, 0.28, 0.12])
CONTEXT_RADIUS = 3.0
CONTEXT_SD = 0.8
# Population centres at the vertices of an equilateral triangle: each is linearly
# separable from the other two, so a linear gate can isolate any single population
# and the capacity limit falls on the experts rather than on the routing.
CONTEXT_ANGLES = np.array([90.0, 210.0, 330.0])
CENTERS = CONTEXT_RADIUS * np.column_stack(
    [np.cos(np.radians(CONTEXT_ANGLES)), np.sin(np.radians(CONTEXT_ANGLES))]
)
# Coefficient directions 120 degrees apart, so every pair of populations conflicts
# by the same amount and only the population sizes distinguish the arrangements.
COEF_ANGLES = np.array([0.0, 120.0, 240.0])
EQUAL_NORMS = np.array([2.5, 2.5, 2.5])
UNEQUAL_NORMS = np.array([2.5, 2.5, 1.0])


def coefficients(norms=EQUAL_NORMS) -> np.ndarray:
    """Per-population logistic coefficients on the two shared signals."""
    directions = np.column_stack(
        [np.cos(np.radians(COEF_ANGLES)), np.sin(np.radians(COEF_ANGLES))]
    )
    return np.asarray(norms, dtype=float)[:, None] * directions


def gate_probabilities(X, priors=PRIORS) -> np.ndarray:
    """Posterior population probability given context, under the generating law."""
    Z = np.asarray(X[list(CONTEXT_FEATURES)], dtype=float)
    logits = -((Z[:, None, :] - CENTERS[None, :, :]) ** 2).sum(2) / (2 * CONTEXT_SD**2)
    return softmax(logits + np.log(priors)[None, :], axis=1)


def population_probabilities(X, norms=EQUAL_NORMS) -> np.ndarray:
    """Event probability under each population's own relationship."""
    S = np.asarray(X[list(SIGNAL_FEATURES)], dtype=float)
    return expit(S @ coefficients(norms).T)


def oracle_probability(X, norms=EQUAL_NORMS) -> np.ndarray:
    return (gate_probabilities(X) * population_probabilities(X, norms)).sum(1)


def generate(n: int, seed: int, norms=EQUAL_NORMS):
    """Design matrix, outcome and latent population label.

    The population label is evaluation metadata. It is never a predictor and never
    a fitting input; it defines the fixed populations over which worst-case loss is
    taken, which is what makes that maximum comparable across candidate models.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(len(PRIORS), n, p=PRIORS)
    Z = CENTERS[group] + CONTEXT_SD * rng.normal(size=(n, 2))
    S = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 2))
    X = pd.DataFrame(np.column_stack([Z, S, noise]), columns=ALL_FEATURES)
    P = population_probabilities(X, norms)
    y = rng.binomial(1, P[np.arange(n), group])
    return X, y, group


def bayes_losses(norms=EQUAL_NORMS, n_nodes: int = 200) -> np.ndarray:
    """Per-population Bayes log-loss, by Gauss-Hermite quadrature.

    With signals standard normal and coefficient vector b, the logit is normal with
    standard deviation ||b||, so the floor depends on the norm alone. These are the
    losses no model can go below, and the reference for excess risk.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    out = []
    for a in np.asarray(norms, dtype=float):
        u = np.sqrt(2.0) * a * nodes
        p = expit(u)
        # H(p) = p*softplus(-u) + (1-p)*softplus(u), stable far into the tails
        entropy = p * np.logaddexp(0.0, -u) + (1 - p) * np.logaddexp(0.0, u)
        out.append(float((weights * entropy).sum() / np.sqrt(np.pi)))
    return np.array(out)


def truth(norms=EQUAL_NORMS) -> dict:
    return dict(
        population_priors=PRIORS.tolist(),
        context_centers=CENTERS.tolist(),
        context_sd=CONTEXT_SD,
        routing_features=list(CONTEXT_FEATURES),
        outcome_features=list(SIGNAL_FEATURES),
        noise_features=list(NOISE_FEATURES),
        coefficients=coefficients(norms).tolist(),
        coefficient_norms=np.asarray(norms, dtype=float).tolist(),
        pairwise_angle_degrees=120.0,
        bayes_losses=bayes_losses(norms).tolist(),
        marginal_event_rate_by_population=0.5,
        explanation=(
            "All populations share two signals and differ only in coefficient "
            "direction, 120 degrees apart, so one logistic expert cannot serve two "
            "populations without loss. Symmetric signals and a zero intercept give "
            "every population event rate one half, so context has no marginal "
            "outcome association. Coefficient norms set the Bayes floors."
        ),
    )
