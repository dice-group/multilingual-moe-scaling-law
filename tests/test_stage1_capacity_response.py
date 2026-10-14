"""Independent Result 1 fixtures; no fitted laws or cluster artifacts needed."""
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from result_analysis.capacity_response import matched_exposure, response, supporting
from result_analysis.capacity_response import tables as derive


class CapacityResponseTests(unittest.TestCase):
    def test_legacy_import_is_the_canonical_implementation(self):
        from result_analysis.capacity_response import tables as derive
        self.assertIs(derive.matched_exposure, matched_exposure)

    def test_response_curve_never_extrapolates(self):
        self.assertAlmostEqual(response.interpolate(10., [1., 100.], [0., 2.]), 1.)
        for x, xp in ((101., [1., 100.]), (0., [1., 100.]), (2., [1., 1.]),
                      (2., [100., 1.]), (np.nan, [1., 100.])):
            with self.subTest(x=x, xp=xp), self.assertRaisesRegex(ValueError, "support"):
                response.interpolate(x, xp, [0., 2.])

    def test_response_recomputes_loss_using_realized_exposure(self):
        records = []
        for language, share, gap in (("a", 1., .1), ("b", 2., .5)):
            for k, exposures, tokens in ((12, [1., 10.], [100., 1000.]),
                                         (200, [2., 20.], [10000., 100000.])):
                for exposure, total in zip(exposures, tokens):
                    loss = 5-2*np.log10(exposure*share)+(gap if k == 200 else 0)
                    records.append(dict(ladder="D", E=1, K=k, language=language, N_act=100.,
                        D_total=total, D_t=exposure*share, raw_nll=loss, adjusted_loss=loss-7.))
        queries = pd.DataFrame([dict(ladder="D", E=1, D200=20000., response=999.)])
        result = response.matched_response(pd.DataFrame(records), queries).iloc[0]
        self.assertAlmostEqual(result.response, .3)
        self.assertAlmostEqual(result.compute_ratio_min, 50.)
        self.assertAlmostEqual(result.compute_ratio_max, 50.)
        self.assertEqual(result.FLOPs, 6*100*20000)
        self.assertEqual(result.language_count, 2)

    def test_crossings_use_log_compute_and_require_unique_support(self):
        rows = pd.DataFrame(dict(ladder=["D"]*4, E=[1, 1, 32, 32],
                                 FLOPs=[1., 100., 4., 400.], response=[-1., 1., -1., 1.]))
        crossings = response.crossovers(rows)
        self.assertAlmostEqual(crossings["1"], 10.)
        self.assertAlmostEqual(crossings["32"], 40.)
        rows.loc[1, "response"] = -.5
        with self.assertRaisesRegex(ValueError, "one supported"):
            response.crossovers(rows)

    def test_composition_interpolation_and_support(self):
        observations = pd.DataFrame([
            dict(arm="compact", seed=1, E=1, language="x", horizon=1, D_t=1., raw_nll=3.),
            dict(arm="compact", seed=1, E=1, language="x", horizon=100, D_t=100., raw_nll=1.),
            dict(arm="K200", seed=1, E=1, language="x", horizon=10, D_t=10., raw_nll=2.5)])
        queries = pd.DataFrame([dict(arm="compact", seed=1, E=1, language="x", horizon=10, method=method)
                               for method in ("log_linear_D_t", "linear_D_t", "pchip_log_D_t")])
        gaps = supporting.composition_points(observations, queries).set_index("method").gap
        self.assertAlmostEqual(gaps.log_linear_D_t, .5)
        self.assertAlmostEqual(gaps.pchip_log_D_t, .5)
        self.assertAlmostEqual(gaps.linear_D_t, 2.5-np.interp(10., [1., 100.], [3., 1.]))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            supporting.composition_points(pd.concat([observations, observations.iloc[2:]]), queries)
        observations.loc[2, "D_t"] = 101.
        with self.assertRaisesRegex(ValueError, "outside"):
            supporting.composition_points(observations, queries)

    def test_resource_gains_balance_languages_then_sizes_and_keep_loss_blind_tiers(self):
        records = []
        for language, size, repeats, gain in (("a", 1, 3, 0.), ("b", 1, 1, 2.),
                ("a", 2, 1, 3.), ("b", 2, 1, 3.), ("c", 1, 1, .1),
                ("d", 1, 1, .2), ("e", 1, 1, .3)):
            for step in range(repeats):
                for expert in (1, 32):
                    records.append(dict(K=200, language=language, N_act=float(size),
                        D_total=float(step+1), D_t=(step+1)/100., E=expert,
                        adjusted_loss=5-(gain if expert == 32 else 0)))
        metadata = pd.DataFrame(dict(split=["train"]*4, subset=["a", "b", "c", "d"],
            resource_availability=["LOW", "LOW", "HIGH", None], words=[10., 100., 1e6, 1e6],
            documents=[1., 10., 1e5, 1e5], utf8_bytes=[100., 1000., 1e7, 1e7]))
        rows = pd.DataFrame(records)
        mapping, gains = supporting.resource_gains(rows, metadata)
        self.assertAlmostEqual(gains.loc["LOW", 32], 2.)
        self.assertEqual(mapping.set_index("language").loc["d", "tier"], "HIGH")
        self.assertEqual(mapping.set_index("language").loc["e", "tier"], "MEDIUM")
        changed, _ = supporting.resource_gains(rows.assign(adjusted_loss=-rows.adjusted_loss), metadata)
        pd.testing.assert_frame_equal(mapping, changed)


    def test_release_loader_checks_identity_hash_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / derive.RELEASE
            release.mkdir(parents=True)
            expected = pd.DataFrame({"loss": [1., 2.]})
            data = release / "data.parquet"
            expected.to_parquet(data, index=False)
            manifest = {"manifest_integrity_sha256": derive.RELEASE_INTEGRITY_ID,
                        "table_inventory": {"sample": {"path": "data.parquet", "rows": 2,
                                                       "sha256": derive.sha256_file(data)}}}
            path = release / "release_manifest.json"
            path.write_text(json.dumps(manifest))
            with patch.object(derive, "RELEASE_MANIFEST_SHA256", derive.sha256_file(path)):
                pd.testing.assert_frame_equal(derive.load_release_table(root, "sample", ["loss"]), expected)
                data.write_bytes(data.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    derive.load_release_table(root, "sample", ["loss"])
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "release manifest"):
                    derive.load_release_table(root, "sample", ["loss"])
            expected.to_parquet(data, index=False)
            manifest["table_inventory"]["sample"].update(rows=3, sha256=derive.sha256_file(data))
            path.write_text(json.dumps(manifest))
            with patch.object(derive, "RELEASE_MANIFEST_SHA256", derive.sha256_file(path)):
                with self.assertRaisesRegex(ValueError, "row-count"):
                    derive.load_release_table(root, "sample", ["loss"])

    def test_seed_loader_keeps_scientific_consistency_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "seed.csv"
            seed = pd.DataFrame(dict(seed=[1, 2, 3], E1_pressure=[1.]*3,
                E16_pressure=[.5]*3, E32_pressure=[.2]*3, relative_attenuation=[.8]*3))
            seed.to_csv(path, index=False)
            with patch.object(derive, "SEED_SIDECAR", path):
                with patch.object(derive, "SEED_SIDECAR_SHA256", derive.sha256_file(path)):
                    pd.testing.assert_frame_equal(derive.load_seed_sidecar(root), seed)
                    seed.loc[0, "relative_attenuation"] = .7
                    seed.to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, "unexpected seed sidecar"):
                        derive.load_seed_sidecar(root)
                with patch.object(derive, "SEED_SIDECAR_SHA256", derive.sha256_file(path)):
                    with self.assertRaisesRegex(ValueError, "inconsistent"):
                        derive.load_seed_sidecar(root)


if __name__ == "__main__":
    unittest.main()
