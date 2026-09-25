"""Fixed-capacity synthetic AML challenge whose populations differ in detectability;
freeze before confirmation."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import torch
from scipy.special import expit
from aml_capacity_data import (
    generate as base_generate, _deviation_draws, PROFILE_FEATURES,
    CONTEXT_FEATURES, DEVIATION_FEATURES, POPULATIONS, coefficients,
)
from capacity_moe import fit, predict, standardizer, simplex_grid, pareto_mask, device

ROOT = Path(__file__).resolve().parent / 'aml_frontier_challenge_20260919'

# The three populations are equally rare but not equally detectable, so a model of any
# given quality has a different error rate in each and no aggregate figure stands in for
# the three.
#
# DEVIATION_NORMS gives the length of each population's coefficient vector on the four
# behavioral deviations, which fixes how far the recorded fields separate the states.
# HIDDEN_DRIVER_SD adds a driver the data dictionary does not contain, which raises the
# attainable loss without being learnable: trading business carries part of its mechanism
# in documentation quality that was never captured. EVENT_RATES fixes each population's
# marginal rate and the intercepts are solved to reach it, so prevalence and detectability
# move independently.
#
# Every population is held at the same rate because the second objective is the maximum
# per-population log loss, and log loss is bounded below by the entropy of the base rate.
# Rates differing by a factor of two or more would make that maximum a reading of which
# population is least rare rather than which is worst served, and the min-max fit would
# chase prevalence instead of misfit.
#
# The deviation scores keep the parent generator's rare positive excursions, so they are
# right-skewed with a heavy upper tail rather than normal. An excursion in a score that
# is not the account's population signature enters through a negative loading and lowers
# its probability, which is the unusual-but-legitimate activity a monitoring model has to
# separate from the state it is looking for.
#
# The resulting within-population Bayes ROC-AUC is 0.889, 0.848 and 0.776 with log losses
# 0.078, 0.086 and 0.089, pooled AUC 0.866 at a pooled event rate of 0.0226. Gradient
# boosting fitted on 40,000 training rows reaches pooled 0.838 and 0.874/0.809/0.702 by
# population, the range reported for deployed transaction-monitoring models; a logistic
# regression on the same twelve fields reaches 0.729, because the signature loadings
# reverse sign across populations and no single linear surface represents them.
# Rescaling changes lengths only; the opposite-signed coefficient directions that put the
# populations in conflict come from the generator and are untouched, so the capacity
# constraint is preserved.
EVENT_RATES = np.array([0.0225, 0.0225, 0.0225])
DEVIATION_NORMS = np.array([1.65, 1.30, 1.10])
HIDDEN_DRIVER_SD = np.array([0.0, 0.0, 1.2])

# Sample sizes every AML study built on this generator draws. At a 2.25% event rate a
# 8,000-row training sample holds about 180 positives, and a model fitted on it scores
# near 0.80 through sample size rather than through the difficulty of the problem; these
# sizes put fitted models at the attainable level and leave the smallest population
# roughly 680 confirmation positives.
TRAIN_N, DISCOVERY_N, CONFIRM_N = 40_000, 40_000, 200_000

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(41)


def scaled_coefficients():
    """Per-population coefficients on the four deviations, at the declared lengths."""
    B = coefficients()
    return DEVIATION_NORMS[:, None] * B / np.linalg.norm(B, axis=1)[:, None]


def _integrate_hidden(logits, sd):
    """Event probability given the recorded fields, averaging out the hidden driver."""
    out = np.empty_like(logits)
    zero = sd == 0
    out[zero] = expit(logits[zero])
    if (~zero).any():
        shifts = np.sqrt(2.0) * sd[~zero][:, None] * _GH_NODES[None, :]
        out[~zero] = ((expit(logits[~zero][:, None] + shifts) @ _GH_WEIGHTS)
                      / np.sqrt(np.pi))
    return out


CALIBRATION_N, CALIBRATION_SEED = 1_000_000, 515


@lru_cache(maxsize=1)
def intercepts():
    """Per-population intercepts that deliver EVENT_RATES under the declared spreads.

    The deviation law is a normal plus a rare positive excursion, so the observable
    part of the logit is not normal and the marginal rate has no closed form. It is
    monotone in the intercept, so a fixed sample of the deviation law and of the hidden
    driver is drawn once and bisected. The sample is fixed by CALIBRATION_SEED and the
    result is cached, so the intercepts are a deterministic function of the declared
    constants rather than of the draw a study happens to make.
    """
    rng = np.random.default_rng(CALIBRATION_SEED)
    base = _deviation_draws(rng, CALIBRATION_N) @ scaled_coefficients().T
    base = base + HIDDEN_DRIVER_SD[None, :] * rng.normal(size=base.shape)
    out = []
    for rate, column in zip(EVENT_RATES, base.T):
        lo, hi = -16.0, 4.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if expit(mid + column).mean() < rate else (lo, mid)
        out.append(0.5 * (lo + hi))
    return np.array(out)


def oracle_probability(X, group):
    """Event probability an ideal model can state from the recorded fields."""
    S = np.asarray(X[list(DEVIATION_FEATURES)], dtype=float)
    logit = intercepts()[group] + np.sum(S * scaled_coefficients()[group], axis=1)
    return _integrate_hidden(logit, HIDDEN_DRIVER_SD[group])


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def generate(n, seed):
    """Design matrix, simulator laundering state and latent customer population.

    The deviation scores are the parent generator's own draw, so they carry its rare
    positive excursions: an account can show an unusual score for reasons unrelated to
    the laundering state, and an excursion in a score that is not its population's
    signature argues against that state through the negative loading. Only the outcome
    mechanism is replaced, by the per-population lengths, intercepts and hidden driver
    declared above.
    """
    X, _, group = base_generate(n, seed)
    signals = np.asarray(X[list(DEVIATION_FEATURES)], dtype=float)
    rng = np.random.default_rng(seed + 900000)
    logit = (intercepts()[group]
             + np.sum(signals * scaled_coefficients()[group], axis=1)
             + HIDDEN_DRIVER_SD[group] * rng.normal(size=n))
    return X, rng.binomial(1, expit(logit)), group


def losses(p, y, group):
    p = np.clip(p.astype(float), 1e-7, 1-1e-7)
    row = -(y[None, :]*np.log(p) + (1-y[None, :])*np.log1p(-p))
    groups = np.array([row[:, group == g].mean(1) for g in range(3)]).T
    return np.column_stack([row.mean(1), groups.max(1)]), groups, row


def main():
    ROOT.mkdir(exist_ok=False)
    protocol = '''# AML capacity challenge with unequal population detectability

Design fixed before this run is scored. Preserve earlier AML experiments unchanged.
This is a deliberately constructed teaching example, not calibrated AML data.
Three fictional customer populations have priors 0.60/0.25/0.15 and the declared
profile distributions of aml_capacity_data. Four behavioral deviations are drawn from
that generator, so each is a standard normal plus a rare positive excursion at rate
0.035, independent across scores and independent of the population. Excursions stand for
unusual activity that is not illicit; one in a score that is not the account's signature
lowers its probability through the negative loading. Conflicting population-specific
coefficient directions come from the same generator, normalized to per-population lengths
1.65, 1.30 and 1.10. Intercepts are solved so that every population has marginal event
rate 0.0225; the deviation law is not normal, so the solve bisects a fixed 1,000,000-row
sample under seed 515 rather than a quadrature, and is cached. Trading business
additionally carries an unrecorded driver of standard deviation 1.2, standing for
documentation quality the data dictionary does not contain, which raises its attainable
loss without being learnable. The populations are therefore equally rare and unequally
detectable: within-population Bayes ROC-AUC is 0.889, 0.848 and 0.776, with log losses
0.078, 0.086 and 0.089 and pooled ROC-AUC 0.866. Holding the rates equal keeps the
maximum per-population log loss a reading of misfit; unequal rates would make it a
reading of which population is least rare, because log loss is bounded below by the
entropy of the base rate. These are stipulated design values for a teaching example,
neither real prevalence estimates nor a sampled production population.

Two logistic experts with a linear softmax profile gate create a capacity constraint.
Expert inputs are the five other context fields and four deviations. Routing roles and
expert capacity are stipulated, not discovered by a new LLM call. Known population
labels enter training objective weights and evaluation but are never predictor inputs.
This is a separate numerical frontier demonstration, not the previous centroid search.

Train 40000 rows seed 62001; discovery 40000 seed 62002; confirmation 200000 seed 62003.
At a 2.25% event rate the earlier 8000/8000/40000 sizes leave about 180 training positives,
and fitted models then sit near 0.80 through sample size rather than through the difficulty
of the problem; these sizes put them at the attainable level.
Fit 21 population-weight vectors at simplex step 0.2 plus empirical training proportions,
each at optimizer seeds 0/1/2; also three direct minmax fits. Total 69 configurations.
Full-batch Adam, learning rate .05, 4000 steps per configuration; no outcome-driven
hyperparameter revisions. Candidate generation is weighted and minmax optimization;
Pareto comparison retains empirical nondominated candidates, not a proof of optimality.

Use discovery loss pairs to retain the frontier and select three models with deterministic
index tie-breaks: minimum overall loss, minimum worst-population loss, and minimum worst
loss subject to overall loss <= minimum overall + .01 nats. Save models and freeze the
frontier, endpoints and source hashes before generating confirmation. Score the frozen
choices unchanged. Any frontier over confirmation scores is descriptive only and may not
reselect models. Report per-population losses and a paired, within-population bootstrap
95% interval for worst-loss reduction versus the mean endpoint (1000 replicates, seed
62004). The interval is conditional on these fitted models and this synthetic generator.
Report failures or a null result without tuning or retrying. One data split and three
optimizer initializations do not establish general reliability or LLM superiority.
'''
    (ROOT/'protocol.md').write_text(protocol)
    source = ROOT/'source'; source.mkdir()
    for name in ('run_aml_frontier_challenge.py', 'aml_capacity_data.py', 'capacity_moe.py', 'capacity_moe_data.py'):
        shutil.copy2(Path(__file__).resolve().parent/name, source/name)
    dump(ROOT/'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    dev = device()
    X, y, g = generate(TRAIN_N, 62001)
    Xd, yd, gd = generate(DISCOVERY_N, 62002)
    std = standardizer(X, PROFILE_FEATURES, CONTEXT_FEATURES+DEVIATION_FEATURES)
    weights = np.vstack([simplex_grid(step=.2), np.bincount(g, minlength=3)/len(g)])
    print('Fitting 66 weighted configurations on', dev, flush=True)
    weighted, _, _ = fit(X, y, g, 2, weights=weights, seeds=(0,1,2), steps=4000, std=std, dev=dev)
    print('Fitting 3 minmax configurations', flush=True)
    robust, _, _ = fit(X, y, g, 2, worst=True, seeds=(0,1,2), steps=4000, std=std, dev=dev)
    params = type(weighted)(*[torch.cat([getattr(weighted,k),getattr(robust,k)],0)
        for k in ('gate_w','gate_b','exp_w','exp_b')])
    torch.save({k:getattr(params,k).cpu() for k in ('gate_w','gate_b','exp_w','exp_b')}, ROOT/'models.pt')
    dump(ROOT/'standardizer.json', {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(std).items()})
    pd = predict(params, Xd, std, dev=dev)
    objectives, groups, _ = losses(pd, yd, gd)
    front = np.flatnonzero(pareto_mask(objectives)).tolist()
    mean = min(front, key=lambda i:(objectives[i,0],objectives[i,1],i))
    worst = min(front, key=lambda i:(objectives[i,1],objectives[i,0],i))
    knee = min([i for i in front if objectives[i,0]<=objectives[mean,0]+.01],
               key=lambda i:(objectives[i,1],objectives[i,0],i))
    endpoints = dict(mean=mean, compromise=knee, robust=worst)
    configs = [dict(objective='weighted',weights=w.tolist(),seed=s) for w in weights for s in (0,1,2)]
    configs += [dict(objective='minmax',seed=s) for s in (0,1,2)]
    frozen = dict(configurations=configs,frontier=front,endpoints=endpoints,
        discovery_objectives=objectives.tolist(),discovery_group_losses=groups.tolist(),
        model_sha256=hashlib.sha256((ROOT/'models.pt').read_bytes()).hexdigest())
    dump(ROOT/'frozen_selection.json',frozen)
    print('Frozen endpoints:',endpoints,flush=True)
    Xc, yc, gc = generate(CONFIRM_N, 62003)
    pc = predict(params,Xc,std,dev=dev)
    co,cg,rows = losses(pc,yc,gc)
    np.savez_compressed(ROOT/'predictions.npz',discovery=pd,yd=yd,gd=gd,confirmation=pc,yc=yc,gc=gc)
    rng=np.random.default_rng(62004);boot={role:[] for role in ('compromise','robust')}
    selected=list(endpoints.values())
    selected_rows=rows[selected]
    for _ in range(1000):
        lg=[]
        for h in range(3):
            ids=np.flatnonzero(gc==h); sample=rng.choice(ids,len(ids),replace=True)
            lg.append(selected_rows[:,sample].mean(1))
        worsts=np.array(lg).max(0)
        for j,role in enumerate(('compromise','robust'),1):boot[role].append(worsts[0]-worsts[j])
    result=dict(populations=POPULATIONS,confirmation_n=40000,
        confirmation_objectives=co.tolist(),confirmation_group_losses=cg.tolist(),
        endpoints={role:dict(index=i,configuration=configs[i],discovery=objectives[i].tolist(),
            confirmation=co[i].tolist(),group_losses=cg[i].tolist()) for role,i in endpoints.items()},
        worst_reduction_ci={role:np.quantile(values,[.025,.975]).tolist() for role,values in boot.items()},
        frozen_selection_sha256=hashlib.sha256((ROOT/'frozen_selection.json').read_bytes()).hexdigest())
    dump(ROOT/'results.json',result)
    print(json.dumps(result['endpoints'],indent=2),flush=True)


if __name__=='__main__':main()
