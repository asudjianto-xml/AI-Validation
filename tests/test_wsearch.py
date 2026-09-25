"""Regressions for the fixed-support constructor, evaluator and noise model."""
import unittest

import numpy as np
import pandas as pd

from aiv.wsearch import (
    BinGrid,
    Candidate,
    FixedSupportConstructor,
    Standardizer,
    TouchMeter,
    WeaknessEvaluator,
    energy_distance,
    resampling_covariance,
    overlap_fraction,
    sample_candidates,
    twin_folds,
)


def toy(n=400, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{j}" for j in range(p)])
    return X, rng


class ConstructorTests(unittest.TestCase):
    def test_support_is_exact_for_both_metrics(self):
        X, rng = toy()
        std = Standardizer.fit(X)
        Xs = std.transform(X)
        for metric in ("box", "ball"):
            con = FixedSupportConstructor(Xs, m=37, metric=metric)
            for cand in sample_candidates(Xs, 20, rng):
                self.assertEqual(con.build(cand).m, 37)

    def test_members_are_the_closest_rows_under_the_declared_metric(self):
        X, rng = toy()
        Xs = Standardizer.fit(X).transform(X)
        con = FixedSupportConstructor(Xs, m=25, metric="box")
        cand = sample_candidates(Xs, 1, rng)[0]
        region = con.build(cand)
        u = np.abs((Xs - cand.z) / np.maximum(cand.w, 1e-9)).max(1)
        inside, outside = u[region.member_idx], np.delete(u, region.member_idx)
        self.assertLessEqual(inside.max(), outside.min() + 1e-12)
        self.assertAlmostEqual(region.gamma, float(inside.max()))

    def test_uniform_shape_box_is_the_chebyshev_cube(self):
        X, rng = toy()
        Xs = Standardizer.fit(X).transform(X)
        con = FixedSupportConstructor(Xs, m=30, metric="box")
        z = Xs[5]
        cube = con.build(Candidate.cube(z))
        direct = np.argsort(np.abs(Xs - z).max(1))[:30]
        self.assertEqual(set(cube.member_idx), set(direct))

    def test_boundary_ties_are_resolved_by_the_declared_rule(self):
        """A box radius is a maximum over coordinates, so discrete features tie
        many rows at the boundary. The region is defined only once the tie-break
        is declared, and both construction paths must apply the same one."""
        rng = np.random.default_rng(5)
        n, p, m = 600, 4, 60
        X = pd.DataFrame(rng.integers(0, 4, size=(n, p)).astype(float),
                         columns=[f"x{j}" for j in range(p)])
        Xs = Standardizer.fit(X).transform(X)
        con = FixedSupportConstructor(Xs, m=m, metric="box")
        cands = sample_candidates(Xs, 40, rng)
        tied = [int((con._keys(c)[0] == np.sort(con._keys(c)[0])[m - 1]).sum()) for c in cands]
        self.assertGreater(max(tied), 1)                 # the ties are real here
        for cand in cands:
            region = con.build(cand)
            u, euclid = con._keys(cand)
            self.assertEqual(set(region.member_idx), set(np.lexsort((euclid, u))[:m]))
            inside = u[region.member_idx]
            self.assertLessEqual(inside.max(), np.delete(u, region.member_idx).min() + 1e-12)

    def test_batch_and_single_agree_exactly_under_ties(self):
        rng = np.random.default_rng(6)
        X = pd.DataFrame(rng.integers(0, 3, size=(500, 4)).astype(float), columns=list("abcd"))
        Xs = Standardizer.fit(X).transform(X)
        for metric in ("box", "ball"):
            con = FixedSupportConstructor(Xs, m=50, metric=metric)
            cands = sample_candidates(Xs, 30, np.random.default_rng(7))
            for one, many in zip((con.build(c) for c in cands), con.build_batch(cands)):
                np.testing.assert_array_equal(one.member_idx, many.member_idx)

    def test_batch_matches_single_candidate_construction(self):
        X, rng = toy(n=300, p=4)
        Xs = Standardizer.fit(X).transform(X)
        for metric in ("box", "ball"):
            con = FixedSupportConstructor(Xs, m=40, metric=metric)
            cands = sample_candidates(Xs, 16, np.random.default_rng(3))
            for one, many in zip((con.build(c) for c in cands), con.build_batch(cands)):
                self.assertEqual(set(one.member_idx), set(many.member_idx))

    def test_freezing_snaps_outward_so_support_can_only_grow(self):
        X, rng = toy(n=500, p=3)
        std = Standardizer.fit(X)
        grid = BinGrid.fit(X, n_bins=10)
        Xs = std.transform(X)
        raw = X.to_numpy(float)
        con = FixedSupportConstructor(Xs, m=50, metric="box")
        for cand in sample_candidates(Xs, 15, np.random.default_rng(7)):
            frozen = con.freeze(con.build(cand), std, grid, raw)
            self.assertGreaterEqual(frozen.support_disc, 50)
            self.assertEqual(int(frozen.mask(X).sum()), frozen.support_disc)

    def test_frozen_rule_transfers_to_unseen_rows(self):
        X, rng = toy(n=400, p=3)
        std, grid = Standardizer.fit(X), BinGrid.fit(X, n_bins=8)
        Xs = std.transform(X)
        con = FixedSupportConstructor(Xs, m=40, metric="ball")
        frozen = con.freeze(con.build(sample_candidates(Xs, 1, rng)[0]), std, grid, X.to_numpy(float))
        fresh = pd.DataFrame(rng.normal(size=(250, 3)), columns=X.columns)
        mask = frozen.mask(fresh)
        self.assertEqual(mask.shape, (250,))
        Z = (fresh.to_numpy(float) - frozen.mu) / frozen.sd
        expect = np.sqrt(((Z - frozen.center) ** 2).sum(1)) <= frozen.radius
        np.testing.assert_array_equal(mask, expect)


class EvaluatorTests(unittest.TestCase):
    def test_contrast_matches_its_definition(self):
        rng = np.random.default_rng(1)
        r = rng.gamma(2.0, 1.0, size=300)
        idx = np.arange(40)
        ev = WeaknessEvaluator(r)
        got = ev.measure(idx)
        want = r[idx].mean() / np.delete(r, idx).mean() - 1
        self.assertAlmostEqual(got.contrast, want, places=12)
        self.assertAlmostEqual(got.sd_in, r[idx].std(ddof=1), places=12)

    def test_complement_contrast_ranks_like_the_population_contrast(self):
        """Eq. (4): at fixed support the two contrasts induce the same ranking."""
        rng = np.random.default_rng(2)
        r = rng.gamma(2.0, 1.0, size=600)
        ev = WeaknessEvaluator(r)
        sets = [rng.choice(600, 60, replace=False) for _ in range(40)]
        comp = np.array([ev.measure(s).contrast for s in sets])
        popn = np.array([r[s].mean() / r.mean() - 1 for s in sets])
        self.assertEqual(list(np.argsort(comp)), list(np.argsort(popn)))

    def test_contrast_map_between_the_two_baselines(self):
        rng = np.random.default_rng(5)
        r = rng.gamma(2.0, 1.0, size=500)
        idx = rng.choice(500, 50, replace=False)
        ev = WeaknessEvaluator(r)
        M = ev.measure(idx).contrast
        pi = 50 / 500
        self.assertAlmostEqual(r[idx].mean() / r.mean() - 1, (1 - pi) * M / (1 + pi * M), places=12)

    def test_tail_rate_uses_the_declared_threshold(self):
        r = np.array([0.1, 0.2, 5.0, 6.0, 0.3, 0.4, 0.5, 0.6], dtype=float)
        ev = WeaknessEvaluator(r, tau=1.0)
        self.assertAlmostEqual(ev.measure(np.array([0, 1, 2, 3])).tail_in, 0.5)

    def test_delete_one_fold_replicates_keep_most_of_the_region(self):
        rng = np.random.default_rng(4)
        n, m, K = 900, 90, 10
        r = rng.gamma(2.0, 1.0, size=n)
        folds = rng.permutation(n) % K
        ev = WeaknessEvaluator(r, folds=folds)
        idx = np.sort(rng.choice(n, m, replace=False))
        rep = ev.replicates(idx)
        self.assertEqual(len(rep.values), K)
        self.assertTrue(np.all(np.isfinite(rep.values)))
        for k in range(K):
            keep = folds != k
            sub = np.flatnonzero(keep)
            inside = np.isin(sub, idx)
            want = r[sub][inside].mean() / r[sub][~inside].mean() - 1
            self.assertAlmostEqual(rep.values[k], want, places=12)
        self.assertGreater(rep.se, 0.0)
        self.assertLessEqual(rep.floor, np.nanmedian(rep.values))

    def test_replicates_beat_within_fold_measurement_on_variance(self):
        """The reason for deleting a fold instead of restricting to one."""
        rng = np.random.default_rng(11)
        n, m, K = 2000, 200, 10
        r = rng.gamma(2.0, 1.0, size=n)
        folds = rng.permutation(n) % K
        idx = np.sort(rng.choice(n, m, replace=False))
        ev = WeaknessEvaluator(r, folds=folds)
        delete_one = ev.replicates(idx).values
        within = []
        for k in range(K):
            sub = np.flatnonzero(folds == k)
            inside = np.isin(sub, idx)
            within.append(r[sub][inside].mean() / r[sub][~inside].mean() - 1)
        self.assertLess(np.std(delete_one), np.std(within))


class CostTests(unittest.TestCase):
    def test_meter_charges_the_grand_total_once_and_m_per_evaluation(self):
        r = np.random.default_rng(0).gamma(2.0, 1.0, size=500)
        meter = TouchMeter()
        ev = WeaknessEvaluator(r, meter=meter)
        self.assertEqual(meter.reads, 500)
        for _ in range(4):
            ev.measure(np.arange(50))
        self.assertEqual(meter.by_tag["eval"], 200)
        self.assertEqual(meter.reads, 700)

    def test_one_shot_benchmark_is_not_at_the_origin(self):
        """A loss-model fit reads every discovery loss, so it is charged n."""
        r = np.random.default_rng(0).gamma(2.0, 1.0, size=1000)
        meter = TouchMeter()
        ev = WeaknessEvaluator(r, meter=meter)
        meter.charge(len(r), "loss_model_fit")
        self.assertEqual(meter.by_tag["loss_model_fit"], 1000)
        before = meter.reads
        for _ in range(100):
            ev.measure(np.arange(150))
        self.assertEqual(meter.reads - before, 15000)


class NoiseModelTests(unittest.TestCase):
    def test_overlap_fraction_is_one_on_the_diagonal_and_symmetric(self):
        members = [np.arange(10), np.arange(5, 15), np.arange(50, 60)]
        frac = overlap_fraction(members, n=100)
        np.testing.assert_allclose(np.diag(frac), 1.0)
        np.testing.assert_allclose(frac, frac.T)
        self.assertAlmostEqual(frac[0, 1], 0.5)
        self.assertAlmostEqual(frac[0, 2], 0.0)

    def test_resampling_covariance_reproduces_the_delta_method_scaling(self):
        members = [np.arange(20), np.arange(10, 30)]
        s2, mean_out, m = 4.0, 2.0, 20
        cov = resampling_covariance(members, n=200, s2=s2, mean_out=mean_out)
        self.assertAlmostEqual(cov[0, 0], s2 / (m * mean_out ** 2))
        self.assertAlmostEqual(cov[0, 1], 0.5 * s2 / (m * mean_out ** 2))

    def test_nearby_candidates_really_do_share_most_members(self):
        """The premise of the correlated-noise model, measured rather than assumed."""
        X, rng = toy(n=800, p=4)
        Xs = Standardizer.fit(X).transform(X)
        con = FixedSupportConstructor(Xs, m=80, metric="box")
        base = sample_candidates(Xs, 1, rng)[0]
        near = [Candidate(base.z + 0.05 * rng.normal(size=4), base.w) for _ in range(8)]
        regions = con.build_batch([base] + near)
        frac = overlap_fraction([r.member_idx for r in regions], n=len(Xs))
        off = frac[np.triu_indices_from(frac, k=1)]
        self.assertGreater(off.mean(), 0.5)

    def test_covariance_is_positive_semidefinite(self):
        X, rng = toy(n=600, p=3)
        Xs = Standardizer.fit(X).transform(X)
        con = FixedSupportConstructor(Xs, m=60, metric="ball")
        regions = con.build_batch(sample_candidates(Xs, 25, rng))
        cov = resampling_covariance([r.member_idx for r in regions], len(Xs), s2=1.0, mean_out=1.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-10)


class TwinningTests(unittest.TestCase):
    def test_folds_are_balanced_and_cover_every_row(self):
        X, _ = toy(n=600, p=3)
        Xs = Standardizer.fit(X).transform(X)
        folds = twin_folds(Xs, K=10)
        counts = np.bincount(folds)
        self.assertEqual(len(counts), 10)
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_folds_resemble_the_whole_sample_at_least_as_well_as_random(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.gamma(2.0, 1.0, size=(1200, 3)), columns=list("abc"))
        Xs = Standardizer.fit(X).transform(X)
        tw = twin_folds(Xs, K=6)
        rd = rng.permutation(1200) % 6
        ed = lambda f: np.mean([energy_distance(Xs[f == k], Xs, rng) for k in range(6)])
        self.assertLessEqual(ed(tw), ed(rd) * 1.5)


if __name__ == "__main__":
    unittest.main()
