"""Fictional account-month AML data in which customer populations compete for
monitoring capacity. No real customers, accounts or operational rules.

`aml_synthetic_data` gives each customer population its own disjoint pair of
behavioral deviation scores, so a mixture with one expert per population fits all
three at once and the overall-mean and worst-population objectives share an
optimum. Its completed searches show that: the two discovery endpoints differ by
about 0.0002 in overall log-loss and 0.001 in worst-population log-loss. Nothing
is being traded, so a frontier drawn there reports optimizer noise.

Here the populations share one set of behavioral deviation scores and disagree
about what those scores mean. Each population has a signature deviation that
indicates the laundering state for it, and the same deviation in another
population is that population's ordinary operating pattern, so conditioned on the
declared profile it argues against the laundering state rather than for it. An
elevated cash share is a departure from a personal remittance customer's declared
profile, while for a cash business it is the expected pattern and the anomaly runs
the other way, toward receipts that fail to arrive as cash. One expert cannot
serve two populations that read the same observable in opposite directions, so
when the mixture has fewer experts than populations one population is sacrificed,
and which one is what the two objectives disagree about.

A fourth deviation, unmatched documentation, raises risk in every population and
carries no conflict, so the design is not purely antisymmetric.

Detectability is set separately from conflict. Under `NO_HIDDEN_DRIVER` every
population is equally detectable from the recorded fields, so worst-population
loss measures misfit alone. Under `OPAQUE_TRADING` part of the trading mechanism
runs through a driver the data dictionary does not contain, such as the quality of
trade documentation that was never captured, so the best attainable loss for that
population is higher and no model can lower it. `bayes_losses` integrates the
hidden driver out and supplies those floors, so excess risk can be reported beside
the raw maximum. The distinction the two regimes separate is whether a population
is under-monitored because the model fails it or because the recorded data does
not contain what would detect it.

The label is a simulator state, not an alert, a SAR or a legal conclusion.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import beta

POPULATIONS = ("personal_remittance", "cash_business", "trading_business")
PRIORS = np.array([0.60, 0.25, 0.15])

# Declared expectations recorded before the monitoring month; these route.
PROFILE_FEATURES = ("business_receipt_share", "expected_cash_share",
                    "expected_crossborder_share")
PROFILE_MEANS = np.array([[0.12, 0.15, 0.25],
                          [0.83, 0.75, 0.13],
                          [0.85, 0.13, 0.77]])
CONCENTRATION = 10.0

# Behavioral deviations, shared by every population and standardized against the
# account's own baseline. The first three are signatures, one per population; the
# fourth raises risk everywhere.
SIGNATURE_FEATURES = ("cash_intensity_z", "outflow_mismatch_z",
                      "counterparty_concentration_z")
COMMON_FEATURE = "documentation_gap_z"
DEVIATION_FEATURES = SIGNATURE_FEATURES + (COMMON_FEATURE,)

CONTEXT_FEATURES = ("log_account_balance", "log_monthly_turnover",
                    "observed_cash_share", "account_tenure_years",
                    "online_activity_share")
FEATURES = PROFILE_FEATURES + CONTEXT_FEATURES + DEVIATION_FEATURES

SIGNATURE_WEIGHT = 2.4      # own signature against the other populations' signatures
COMMON_WEIGHT = 0.9         # unmatched documentation, the same in every population
INTERCEPT = -5.0
EXCURSION_RATE = 0.035      # rare legitimate excursions, in every deviation score
SCALES = np.array([1.0, 1.0, 1.0])
# Standard deviation of a per-population driver absent from the data dictionary.
NO_HIDDEN_DRIVER = np.array([0.0, 0.0, 0.0])
OPAQUE_TRADING = np.array([0.0, 0.0, 2.6])
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(41)


def coefficients(scales=SCALES) -> np.ndarray:
    """Per-population coefficients on the four shared deviation scores.

    A population loads positively on its own signature and negatively on the other
    two, which is what puts the populations in conflict: the row sums of the
    signature block are zero, so a single expert fitted to two populations keeps
    only their common part.
    """
    signature = SIGNATURE_WEIGHT * (np.eye(3) - 1.0 / 3.0)
    common = np.full((3, 1), COMMON_WEIGHT)
    return np.asarray(scales, dtype=float)[:, None] * np.hstack([signature, common])


def _deviation_draws(rng, n):
    s = rng.normal(size=(n, len(DEVIATION_FEATURES)))
    # Unusual activity is not always illicit: rare excursions land in every score,
    # including scores that carry no signature for the account's own population.
    s += (rng.binomial(1, EXCURSION_RATE, size=s.shape)
          * rng.uniform(1.5, 3.0, size=s.shape))
    return s


def _integrate_hidden(logits: np.ndarray, hidden) -> np.ndarray:
    """Average sigmoid over the unobserved driver, by Gauss-Hermite quadrature.

    With the driver absent from the recorded fields, the probability an ideal
    model can state given those fields is this average, not the realized sigmoid.
    """
    hidden = np.asarray(hidden, dtype=float)
    out = np.empty_like(logits)
    for h in range(logits.shape[1]):
        if hidden[h] == 0:
            out[:, h] = expit(logits[:, h])
            continue
        shifts = np.sqrt(2.0) * hidden[h] * _GH_NODES
        out[:, h] = (expit(logits[:, h][:, None] + shifts[None, :])
                     @ _GH_WEIGHTS) / np.sqrt(np.pi)
    return out


@lru_cache(maxsize=16)
def _intercept_offsets(scales: tuple, hidden: tuple, n: int = 400_000,
                       seed: int = 515) -> tuple:
    """Per-population intercept shifts that hold marginal prevalence fixed.

    Averaging the sigmoid over an unobserved driver raises the marginal rate,
    because the sigmoid is convex where the base rate is low. Without a correction
    the opaque population would be both more prevalent and harder to detect, and
    the two effects could not be told apart. These offsets restore each
    population's marginal rate to the value it has with no hidden driver, so the
    regimes differ in detectability alone.
    """
    rng = np.random.default_rng(seed)
    s = _deviation_draws(rng, n)
    base = INTERCEPT + s @ coefficients(np.asarray(scales)).T
    target = expit(base).mean(0)
    offsets = []
    for h, sd in enumerate(hidden):
        if sd == 0:
            offsets.append(0.0)
            continue
        lo, hi = -6.0, 6.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rate = _integrate_hidden((base[:, [h]] + mid),
                                     np.asarray([sd]))[:, 0].mean()
            lo, hi = (mid, hi) if rate < target[h] else (lo, mid)
        offsets.append(0.5 * (lo + hi))
    return tuple(offsets)


def state_logits(X, scales=SCALES, hidden=NO_HIDDEN_DRIVER) -> np.ndarray:
    S = np.asarray(X[list(DEVIATION_FEATURES)], dtype=float)
    offsets = np.asarray(_intercept_offsets(tuple(np.asarray(scales, dtype=float)),
                                            tuple(np.asarray(hidden, dtype=float))))
    return INTERCEPT + offsets[None, :] + S @ coefficients(scales).T


def state_probabilities(X, scales=SCALES, hidden=NO_HIDDEN_DRIVER) -> np.ndarray:
    """Laundering-state probability given the recorded fields, per population."""
    return _integrate_hidden(state_logits(X, scales, hidden), hidden)


def posterior_population(X) -> np.ndarray:
    """Posterior population probability given the declared profile."""
    z = np.asarray(X[list(PROFILE_FEATURES)], dtype=float)
    logs = np.column_stack([
        beta.logpdf(z, m * CONCENTRATION, (1 - m) * CONCENTRATION).sum(1)
        for m in PROFILE_MEANS])
    return softmax(logs + np.log(PRIORS)[None, :], axis=1)


def oracle_probability(X, scales=SCALES, hidden=NO_HIDDEN_DRIVER) -> np.ndarray:
    return (posterior_population(X) * state_probabilities(X, scales, hidden)).sum(1)


def feature_dictionary() -> list:
    meanings = {
        "business_receipt_share": ("Declared expected share of receipts from business activity", "pre_period"),
        "expected_cash_share": ("Expected fraction of receipts in cash, recorded before the monitoring month", "pre_period"),
        "expected_crossborder_share": ("Expected fraction of activity involving cross-border payments", "pre_period"),
        "log_account_balance": ("Log of balance at start of account-month", "pre_period"),
        "log_monthly_turnover": ("Log of total value moving through the account during the month", "period_end"),
        "observed_cash_share": ("Fraction of observed incoming value paid in cash", "period_end"),
        "account_tenure_years": ("Years since account opening", "pre_period"),
        "online_activity_share": ("Expected fraction of activity through online channels", "pre_period"),
        "cash_intensity_z": ("Standardized deviation of cash receipts from the account's own baseline", "period_end"),
        "outflow_mismatch_z": ("Standardized deviation of outflows not matched to recorded business purpose", "period_end"),
        "counterparty_concentration_z": ("Standardized deviation of counterparty concentration from the account's own baseline", "period_end"),
        "documentation_gap_z": ("Standardized deviation of payments lacking supporting documentation", "period_end"),
    }
    units = {"_z": "standardized score", "_share": "fraction", "_years": "years"}
    def unit(f):
        for suffix, u in units.items():
            if f.endswith(suffix):
                return u
        return "natural log of arbitrary currency units"
    return [dict(feature=f, meaning=meanings[f][0], available=meanings[f][1],
                 unit=unit(f),
                 eligibility="Available before the monitoring decision; not a post-investigation field")
            for f in FEATURES]


def generate(n: int, seed: int, scales=SCALES, hidden=NO_HIDDEN_DRIVER):
    """Design matrix, simulator laundering state and latent customer population.

    The population label is evaluator metadata. It is never an estimator input and
    never an expert-training label; it fixes the populations over which worst-case
    loss is taken, which is what makes that maximum comparable across models.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(3, n, p=PRIORS)
    m = PROFILE_MEANS[group]
    profile = rng.beta(m * CONCENTRATION, (1 - m) * CONCENTRATION)
    deviations = _deviation_draws(rng, n)
    frame = pd.DataFrame({
        "business_receipt_share": profile[:, 0],
        "expected_cash_share": profile[:, 1],
        "expected_crossborder_share": profile[:, 2],
        "log_account_balance": rng.normal(9.5, 1.1, n),
        "log_monthly_turnover": rng.normal(10.3, 1.15, n),
        "observed_cash_share": np.clip(profile[:, 1] + rng.normal(0, 0.15, n), 0, 1),
        "account_tenure_years": rng.gamma(2.0, 3.0, n),
        "online_activity_share": rng.beta(3, 2, n),
    })
    for j, f in enumerate(DEVIATION_FEATURES):
        frame[f] = deviations[:, j]
    X = frame[list(FEATURES)]
    rows = np.arange(n)
    hidden = np.asarray(hidden, dtype=float)
    logit = state_logits(X, scales, hidden)[rows, group]
    logit = logit + hidden[group] * rng.normal(size=n)
    return X, rng.binomial(1, expit(logit)), group


def bayes_losses(scales=SCALES, hidden=NO_HIDDEN_DRIVER, n: int = 2_000_000,
                 seed: int = 909) -> np.ndarray:
    """Per-population Bayes log-loss given the recorded fields.

    The excursion mixture is not normal, so the deviation law is sampled rather
    than integrated; the unobserved driver is integrated by quadrature. The floor
    is the entropy of the probability an ideal model can state from the recorded
    fields, which exceeds the entropy of the realized probability whenever part of
    the mechanism is unobserved.
    """
    rng = np.random.default_rng(seed)
    s = _deviation_draws(rng, n)
    offsets = np.asarray(_intercept_offsets(tuple(np.asarray(scales, dtype=float)),
                                            tuple(np.asarray(hidden, dtype=float))))
    logits = INTERCEPT + offsets[None, :] + s @ coefficients(scales).T
    p = np.clip(_integrate_hidden(logits, hidden), 1e-12, 1 - 1e-12)
    entropy = -(p * np.log(p) + (1 - p) * np.log1p(-p))
    return entropy.mean(0)


def truth(scales=SCALES, hidden=NO_HIDDEN_DRIVER) -> dict:
    B = coefficients(scales)
    norms = np.linalg.norm(B, axis=1)
    cos = (B @ B.T) / np.outer(norms, norms)
    return dict(
        populations=list(POPULATIONS), priors=PRIORS.tolist(),
        routing_features=list(PROFILE_FEATURES),
        profile_means=PROFILE_MEANS.tolist(),
        profile_beta_concentration=CONCENTRATION,
        deviation_features=list(DEVIATION_FEATURES),
        signature_features=list(SIGNATURE_FEATURES),
        common_feature=COMMON_FEATURE,
        coefficients=B.tolist(), coefficient_norms=norms.tolist(),
        pairwise_cosines=cos.tolist(),
        intercept=INTERCEPT, excursion_rate=EXCURSION_RATE,
        coefficient_scales=np.asarray(scales, dtype=float).tolist(),
        hidden_driver_sd=np.asarray(hidden, dtype=float).tolist(),
        intercept_offsets=list(_intercept_offsets(
            tuple(np.asarray(scales, dtype=float)),
            tuple(np.asarray(hidden, dtype=float)))),
        bayes_losses=bayes_losses(scales, hidden).tolist(),
        positive_state=("Fictional laundering state sampled by the simulator; "
                        "not an alert, a SAR or a legal determination"),
        account_unit="One independently sampled account-month per account",
        limitations=[
            "Stipulated mechanisms and prevalence, not estimated from real AML cases",
            "Opposite-signed loadings across populations are a stipulated design "
            "device that creates capacity conflict, not a measured effect",
            "The unobserved driver is a stipulated device for an undetectable "
            "share of a typology, not an estimate of real detection limits",
            "Standardized deviation fields are assumed available for every account",
            "Labels are complete and correct in simulation; real investigative "
            "labels need separate treatment",
            "Declared profiles are assumed reliable; their error is not modeled",
        ])
