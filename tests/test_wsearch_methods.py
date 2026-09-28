"""Regressions for the loss model, the composite-kernel surrogate and confirmation."""
import unittest

import numpy as np
import pandas as pd

from aiv.wsearch import (
    Candidate,
    distinct_by_overlap,
    max_overlap,
    FixedSupportConstructor,
    Standardizer,
    WeaknessEvaluator,
    resampling_covariance,
    pooled_within_variance,
    resampling_covariance,
    sample_candidates,
)
from aiv.wsearch.confirm import bca_confirm, benjamini_hochberg, holm
from aiv.wsearch.gp import (
    CompositeGP,
    cosine_kernel,
    expected_improvement,
    interval_coverage,
    matern52,
)
from aiv.wsearch.lossmodel import (
    AuxLossModel,
    b1_region_scores,
    error_tree_region,
    region_embeddings,
)


def planted(n=1500, p=3, seed=0, lift=3.0):
    """Covariates with a loss elevated inside one ball, so there is a truth."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    center = np.array([1.2, -0.8] + [0.0] * (p - 2))
    d = np.sqrt(((X - center) ** 2).sum(1))
    rate = 0.4 * (1.0 + lift * np.exp(-(d ** 2)))
    r = rng.gamma(2.0, rate / 2.0)
    return X, r, center, rng


class LossModelTests(unittest.TestCase):
    def test_ensemble_is_linear_in_its_leaf_features(self):
        """g(x) = base + phi(x).v, the identity the B1/S5b comparison rests on."""
        X, r, _, _ = planted()
        aux = AuxLossModel(depth=3, n_estimators=120).fit(X, r)
        self.assertLess(aux.reconstruction_error, 1e-4)
        direct = np.asarray(aux.phi(X) @ aux.leaf_values).ravel() + aux.base
        np.testing.assert_allclose(direct, aux.predict(X), atol=1e-4)

    def test_region_score_is_the_mean_prediction_over_members(self):
        X, r, _, rng = planted()
        aux = AuxLossModel(depth=3, n_estimators=120).fit(X, r)
        members = [rng.choice(len(X), 80, replace=False) for _ in range(5)]
        got = b1_region_scores(region_embeddings(aux.phi(X), members), aux)
        want = [aux.predict(X[idx]).mean() for idx in members]
        np.testing.assert_allclose(got, want, atol=1e-4)

    def test_region_average_beats_pointwise_ranking_on_the_objective(self):
        """Why B1 scores the region and not its center."""
        X, r, _, rng = planted(n=2000)
        std = Standardizer.fit(pd.DataFrame(X))
        Xs = std.transform(pd.DataFrame(X))
        aux = AuxLossModel(depth=3, n_estimators=150).fit(X, r)
        con = FixedSupportConstructor(Xs, m=120, metric="box")
        cands = sample_candidates(Xs, 400, np.random.default_rng(1))
        regions = con.build_batch(cands)
        ev = WeaknessEvaluator(r)
        truth = np.array([ev.contrast(g.member_idx) for g in regions])

        Phi = aux.phi(X)
        avg = b1_region_scores(region_embeddings(Phi, [g.member_idx for g in regions]), aux)
        pointwise = np.array([aux.predict(std.inverse(c.z)[None, :])[0] for c in cands])
        self.assertGreater(truth[np.argmax(avg)], truth[np.argmax(pointwise)])
        self.assertGreater(np.corrcoef(avg, truth)[0, 1], np.corrcoef(pointwise, truth)[0, 1])

    def test_error_tree_returns_a_supported_high_loss_leaf(self):
        X, r, _, _ = planted()
        idx, tree = error_tree_region(X, r, m=100)
        self.assertGreaterEqual(len(idx), 100)
        self.assertGreater(r[idx].mean(), r.mean())


class CompositeKernelTests(unittest.TestCase):
    def test_components_have_unit_diagonal_and_are_psd(self):
        rng = np.random.default_rng(0)
        X, E = rng.normal(size=(40, 3)), np.abs(rng.normal(size=(40, 15)))
        for K in (matern52(X, X, np.ones(3)), cosine_kernel(E)):
            np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
            self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-9)

    def test_cosine_normalization_removes_the_compactness_artifact(self):
        """A raw embedding inner product gives compact regions a larger prior."""
        rng = np.random.default_rng(3)
        E = np.abs(rng.normal(size=(60, 20)))
        E[:30] *= 6.0                                    # regions concentrated in few leaves
        raw_diag = (E * E).sum(1)
        self.assertGreater(raw_diag[:30].mean(), 5 * raw_diag[30:].mean())
        np.testing.assert_allclose(np.diag(cosine_kernel(E)), 1.0, atol=1e-10)

    def test_mixing_weight_selects_the_component(self):
        rng = np.random.default_rng(1)
        X, E = rng.normal(size=(30, 2)), np.abs(rng.normal(size=(30, 8)))
        gp = CompositeGP(lam=1.0, lengthscale=np.ones(2), sigma2=1.0)
        np.testing.assert_allclose(gp._k(X, E), matern52(X, X, np.ones(2)), atol=1e-12)
        gp = CompositeGP(lam=0.0, lengthscale=np.ones(2), sigma2=1.0)
        np.testing.assert_allclose(gp._k(X, E), cosine_kernel(E), atol=1e-12)

    def test_posterior_interpolates_a_noiseless_function(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 2))
        E = np.abs(rng.normal(size=(50, 6)))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        gp = CompositeGP(lam=1.0).fit(X, E, y)
        mu, sd = gp.predict(X, E)
        self.assertLess(float(np.abs(mu - y).max()), 0.05)
        self.assertLess(float(sd.max()), 0.3)

    def test_the_evaluator_is_deterministic_within_a_run(self):
        """Given fixed discovery losses there is no measurement noise to model."""
        X, r, center, _ = planted(n=1200, p=3, seed=4)
        Xdf = pd.DataFrame(X)
        Xs = Standardizer.fit(Xdf).transform(Xdf)
        con = FixedSupportConstructor(Xs, m=120, metric="box")
        ev = WeaknessEvaluator(r)
        zc = (center - Xdf.mean().to_numpy()) / Xdf.std().to_numpy()
        c = Candidate.cube(zc)
        self.assertEqual(ev.contrast(con.build(c).member_idx),
                         ev.contrast(con.build(c).member_idx))

    def test_resampling_covariance_is_collinear_with_the_signal_kernel(self):
        """Why it cannot serve as an observation-noise term.

        The covariance is proportional to the overlap-fraction matrix, which is
        itself a positive-semidefinite similarity kernel over candidates and is
        strongly correlated with the composite kernel. A likelihood carrying both
        has two nearly collinear covariance components, so the split between
        signal and noise is weakly determined; which way it goes depends on the
        data, and on the credit-default task the signal amplitude collapses.
        """
        X, r, center, rng = planted(n=2000, p=3, seed=5)
        Xdf = pd.DataFrame(X)
        Xs = Standardizer.fit(Xdf).transform(Xdf)
        con = FixedSupportConstructor(Xs, m=150, metric="box")
        ev = WeaknessEvaluator(r)
        aux = AuxLossModel(depth=3, n_estimators=120).fit(X, r)
        zc = (center - Xdf.mean().to_numpy()) / Xdf.std().to_numpy()
        base = Candidate.cube(zc)
        cands = [Candidate(zc + 0.6 * rng.normal(size=3), base.w) for _ in range(120)]
        members = [g.member_idx for g in con.build_batch(cands)]
        meas = [ev.measure(mi) for mi in members]
        y = np.array([mm.contrast for mm in meas])
        coords = np.stack([c.z for c in cands])
        E = region_embeddings(aux.phi(X), members)
        Sigma = resampling_covariance(members, len(X), pooled_within_variance(meas),
                                      float(np.mean([mm.mean_out for mm in meas])))
        self.assertGreater(np.linalg.eigvalsh(Sigma).min(), -1e-10)   # a valid kernel

        d = np.sqrt(np.diag(Sigma))
        Sigma_corr = Sigma / np.outer(d, d)
        np.testing.assert_allclose(np.diag(Sigma_corr), 1.0, atol=1e-10)

        gp = CompositeGP(lam=0.5).fit(coords, E, y)
        K = gp._k(coords, E) / gp.sigma2
        iu = np.triu_indices_from(K, k=1)
        self.assertGreater(float(np.corrcoef(K[iu], Sigma_corr[iu])[0, 1]), 0.5)

    def test_a_jitter_only_surrogate_is_overconfident_on_a_rough_response(self):
        """The response is deterministic but piecewise constant in the coordinates,
        so the nugget is model misspecification rather than measurement error."""
        X, r, center, rng = planted(n=2000, p=3, seed=6)
        Xdf = pd.DataFrame(X)
        Xs = Standardizer.fit(Xdf).transform(Xdf)
        con = FixedSupportConstructor(Xs, m=150, metric="box")
        ev = WeaknessEvaluator(r)
        aux = AuxLossModel(depth=3, n_estimators=120).fit(X, r)
        zc = (center - Xdf.mean().to_numpy()) / Xdf.std().to_numpy()
        base = Candidate.cube(zc)
        cands = [Candidate(zc + 0.8 * rng.normal(size=3), base.w) for _ in range(160)]
        members = [g.member_idx for g in con.build_batch(cands)]
        y = np.array([ev.contrast(mi) for mi in members])
        coords = np.stack([c.z for c in cands])
        E = region_embeddings(aux.phi(X), members)
        tr, te = np.arange(110), np.arange(110, 160)

        nug = CompositeGP(lam=0.5).fit(coords[tr], E[tr], y[tr])
        cov_nug = interval_coverage(*nug.predictive(coords[te], E[te]), y[te], 0.90)
        bare = CompositeGP(lam=0.5, jitter=1e-8)
        bare.fit(coords[tr], E[tr], y[tr])
        bare.noise_scale = 1e-8
        bare._factor(coords[tr], E[tr], y[tr])
        cov_bare = interval_coverage(*bare.predict(coords[te], E[te]), y[te], 0.90)
        self.assertLess(cov_bare, 0.7)
        self.assertGreater(cov_nug, cov_bare)

    def test_overlap_wastes_budget_and_the_constraint_recovers_it(self):
        """The operational cost of overlap, which the acquisition constrains."""
        X, r, center, rng = planted(n=2500, p=3, seed=8)
        Xdf = pd.DataFrame(X)
        Xs = Standardizer.fit(Xdf).transform(Xdf)
        con = FixedSupportConstructor(Xs, m=150, metric="box")
        zc = (center - Xdf.mean().to_numpy()) / Xdf.std().to_numpy()
        base = Candidate.cube(zc)
        cands = [Candidate(zc + 0.18 * rng.normal(size=3), base.w) for _ in range(120)]
        members = [g.member_idx for g in con.build_batch(cands)]
        dup = [max_overlap([members[i]], members[:i], len(X))[0] > 0.9
               for i in range(1, len(members))]
        self.assertGreater(np.mean(dup), 0.5)
        keep = distinct_by_overlap(members, len(X), threshold=0.5)
        self.assertLess(len(keep), len(members) / 3)

    def test_loss_kernel_ties_only_on_duplicate_member_sets(self):
        """The composite kernel was motivated by acquisition ties that a
        piecewise-constant leaf kernel would produce. Ties do occur, but only
        between proposals whose member sets are identical, which are the same
        region and should tie; the acquisition maximizer is unique."""
        X, r, center, rng = planted(n=2500, p=3, seed=12)
        Xdf = pd.DataFrame(X)
        Xs = Standardizer.fit(Xdf).transform(Xdf)
        con = FixedSupportConstructor(Xs, m=150, metric="box")
        ev = WeaknessEvaluator(r)
        aux = AuxLossModel(depth=3, n_estimators=120).fit(X, r)

        tr_c = sample_candidates(Xs, 80, np.random.default_rng(1), "box")
        tr_m = [g.member_idx for g in con.build_batch(tr_c)]
        y = np.array([ev.contrast(mi) for mi in tr_m])
        Phi = aux.phi(X)
        gp = CompositeGP(lam=0.0).fit(np.stack([c.as_tuple() for c in tr_c]),
                                      region_embeddings(Phi, tr_m), y)

        zc = (center - Xdf.mean().to_numpy()) / Xdf.std().to_numpy()
        base = Candidate.cube(zc)
        pool = [Candidate(zc + 0.05 * rng.normal(size=3), base.w) for _ in range(300)]
        pm = [g.member_idx for g in con.build_batch(pool)]
        mu, sd = gp.predict(np.stack([c.as_tuple() for c in pool]), region_embeddings(Phi, pm))
        ei = expected_improvement(mu, sd, y.max())

        n_distinct_members = len({tuple(a) for a in pm})
        self.assertEqual(len(np.unique(np.round(ei, 12))), n_distinct_members)
        self.assertEqual(int((ei >= ei.max() - 1e-9).sum()),
                         sum(1 for a in pm if tuple(a) == tuple(pm[int(np.argmax(ei))])))

    def test_expected_improvement_is_nonnegative_and_rewards_uncertainty(self):
        mu = np.array([0.0, 0.0, 1.0])
        sd = np.array([0.1, 1.0, 0.1])
        ei = expected_improvement(mu, sd, best=0.5)
        self.assertTrue(np.all(ei >= 0))
        self.assertGreater(ei[1], ei[0])
        self.assertGreater(ei[2], ei[0])


class ConfirmationTests(unittest.TestCase):
    def test_interval_covers_a_known_contrast(self):
        rng = np.random.default_rng(0)
        n = 3000
        inside = np.zeros(n, dtype=bool)
        inside[:200] = True
        loss = rng.gamma(2.0, 0.2, n)
        loss[inside] *= 2.0
        c = bca_confirm(loss, inside, margin=0.0, n_boot=2000)
        self.assertLess(c.ci_low, 1.0)
        self.assertGreater(c.ci_high, 1.0)
        self.assertLess(c.p_value, 0.01)

    def test_margin_is_what_stops_trivial_confirmation(self):
        """A weak but real inflation clears a zero margin and fails a 0.5 margin."""
        rng = np.random.default_rng(1)
        n = 4000
        inside = np.zeros(n, dtype=bool)
        inside[:400] = True
        loss = rng.gamma(2.0, 0.2, n)
        loss[inside] *= 1.15
        self.assertLess(bca_confirm(loss, inside, margin=0.0, n_boot=2000).p_value, 0.05)
        self.assertGreater(bca_confirm(loss, inside, margin=0.5, n_boot=2000).p_value, 0.5)

    def test_one_sided_test_is_calibrated_at_the_margin(self):
        """Rejection rate at the boundary of H0, which is what the margin controls."""
        rng = np.random.default_rng(7)
        n, m, margin, reject = 2000, 200, 0.5, 0
        trials = 60
        for t in range(trials):
            inside = np.zeros(n, dtype=bool)
            inside[rng.choice(n, m, replace=False)] = True
            loss = rng.gamma(2.0, 0.2, n)
            loss[inside] *= (1.0 + margin)          # exactly on the boundary
            if bca_confirm(loss, inside, margin=margin, n_boot=800, seed=t).p_value < 0.05:
                reject += 1
        self.assertLess(reject / trials, 0.20)      # loose bound at 60 trials

    def test_insufficient_support_is_inconclusive_rather_than_significant(self):
        rng = np.random.default_rng(2)
        loss = rng.gamma(2.0, 0.2, 500)
        inside = np.zeros(500, dtype=bool)
        inside[:5] = True
        c = bca_confirm(loss, inside, min_support=30)
        self.assertTrue(np.isnan(c.p_value))
        self.assertIn("inconclusive", c.detail)

    def test_frozen_region_confirms_through_its_own_rule(self):
        X, r, center, rng = planted(n=2000, p=3, seed=9)
        Xdf = pd.DataFrame(X, columns=list("abc"))
        from aiv.wsearch.grid import BinGrid

        std, grid = Standardizer.fit(Xdf), BinGrid.fit(Xdf, n_bins=10)
        Xs = std.transform(Xdf)
        con = FixedSupportConstructor(Xs, m=150, metric="ball")
        zc = (center - Xdf.mean().to_numpy()) / Xdf.std().to_numpy()
        frozen = con.freeze(con.build(Candidate.cube(zc)), std, grid, X)

        conf_X = rng.normal(size=(2000, 3))
        d = np.sqrt(((conf_X - center) ** 2).sum(1))
        conf_r = rng.gamma(2.0, 0.4 * (1.0 + 3.0 * np.exp(-(d ** 2))) / 2.0)
        conf_df = pd.DataFrame(conf_X, columns=list("abc"))
        c = bca_confirm(conf_r, frozen.mask(conf_df), margin=0.0, n_boot=1500)
        self.assertGreater(c.contrast, 0.0)
        self.assertGreater(c.ci_low, 0.0)

    def test_bh_rejects_at_least_as_much_as_holm(self):
        rng = np.random.default_rng(0)
        p = np.concatenate([rng.uniform(0, 0.01, 4), rng.uniform(0, 1, 16)])
        bh, _ = benjamini_hochberg(p, 0.05)
        hl, _ = holm(p, 0.05)
        self.assertGreaterEqual(bh.sum(), hl.sum())
        self.assertTrue(np.all(hl <= bh))

    def test_bh_controls_the_false_discovery_rate_under_the_null(self):
        rng = np.random.default_rng(1)
        fdr = []
        for _ in range(200):
            p = rng.uniform(size=20)
            rej, _ = benjamini_hochberg(p, 0.10)
            fdr.append(1.0 if rej.any() else 0.0)
        self.assertLess(np.mean(fdr), 0.15)


if __name__ == "__main__":
    unittest.main()


class KMedoidsTests(unittest.TestCase):
    def test_pam_lowers_cost_and_returns_real_observations(self):
        from aiv.wsearch.kmedoids import medoid_of, pam
        rng = np.random.default_rng(0)
        pts = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(4, 0.3, (60, 2)),
                         rng.normal(-4, 0.3, (60, 2))])
        D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        res = pam(D, k=3, seed=0)
        self.assertEqual(len(res.medoids), 3)
        self.assertTrue(set(res.medoids) <= set(range(len(pts))))   # medoids are observations
        self.assertGreater(res.silhouette, 0.8)                      # the three groups are found
        for c in range(3):
            members = np.flatnonzero(res.labels == c)
            self.assertIn(medoid_of(D, members), set(res.medoids))

    def test_build_swap_matches_scikit_learn_extra(self):
        # Reference medoids from scikit-learn-extra 0.3.0 KMedoids(method="pam",
        # init="build"); the integer grid produces many tied distances.
        from aiv.wsearch.kmedoids import pam_build_swap
        for kind, expected in [("ties", [8, 15, 107, 101, 105]), ("normal", [27, 62, 37, 101, 73])]:
            rng = np.random.default_rng(7)
            X = rng.integers(0, 4, (120, 3)).astype(float) if kind == "ties" else rng.normal(size=(120, 3))
            D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
            self.assertEqual(pam_build_swap(D, 5, block=17).tolist(), expected)

    def test_medoid_minimises_average_dissimilarity(self):
        from aiv.wsearch.kmedoids import medoid_of
        rng = np.random.default_rng(1)
        D = rng.random((40, 40)); D = (D + D.T) / 2; np.fill_diagonal(D, 0.0)
        members = np.arange(0, 40, 2)
        best = medoid_of(D, members)
        mine = D[np.ix_(members, members)].mean(1)
        self.assertEqual(best, members[int(np.argmin(mine))])


class FrontierTests(unittest.TestCase):
    def test_lift_and_coverage_behave_at_the_extremes(self):
        from aiv.wsearch import lift_coverage
        r = np.concatenate([np.full(90, 0.1), np.full(10, 5.0)])
        whole = np.ones(100, dtype=bool)
        lift, cov = lift_coverage(r, whole)
        self.assertAlmostEqual(lift, 1.0)          # the whole sample has lift one
        self.assertAlmostEqual(cov, 1.0)           # and holds all the excess
        hot = np.zeros(100, dtype=bool); hot[90:] = True
        lift, cov = lift_coverage(r, hot)
        self.assertGreater(lift, 5.0)
        self.assertAlmostEqual(cov, 1.0)           # the 10 hot rows hold all of it

    def test_contrast_runs_away_where_lift_does_not(self):
        """Why the axes are lift and coverage rather than contrast and support."""
        from aiv.wsearch import lift_coverage
        rng = np.random.default_rng(0)
        r = rng.gamma(2.0, 0.2, 1000)
        r[:20] *= 0.01                              # a small, unusually easy complement
        mask = np.ones(1000, dtype=bool); mask[:20] = False
        contrast = r[mask].mean() / r[~mask].mean() - 1
        lift, cov = lift_coverage(r, mask)
        self.assertGreater(contrast, 50)            # inflated by the easy complement
        self.assertLess(lift, 1.1)                  # lift stays bounded

    def test_frontier_keeps_only_nondominated_regions(self):
        from aiv.wsearch import weakness_frontier
        rng = np.random.default_rng(1)
        r = rng.gamma(2.0, 0.3, 600); r[:200] *= 3.0
        regions = {}
        for k in (50, 100, 200, 400):
            m = np.zeros(600, dtype=bool); m[:k] = True
            regions[f"top{k}"] = m
        bad = np.zeros(600, dtype=bool); bad[300:400] = True   # a weak-free region
        regions["elsewhere"] = bad
        front, allpts = weakness_frontier(r, regions, {"top50": 1, "top100": 2})
        labels = [p.label for p in front]
        self.assertNotIn("elsewhere", labels)
        self.assertEqual(len(front), len(set(labels)))
        lifts = [p.lift for p in front]; covs = [p.coverage for p in front]
        self.assertEqual(lifts, sorted(lifts, reverse=True))   # sorted by lift
        self.assertEqual(covs, sorted(covs))                   # coverage rises as lift falls
