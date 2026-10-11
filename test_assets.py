import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load("benchmark")
rsna = load("rsna_prior")


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = benchmark.generate()

    def test_repeatable_generation(self):
        pd.testing.assert_frame_equal(self.data, benchmark.generate())

    def test_schema_and_ids(self):
        self.assertEqual(list(self.data), list(benchmark.DESCRIPTIONS))
        self.assertTrue(self.data.row_id.is_unique)
        self.assertEqual(len(self.data), 8928)

    def test_group_and_time_isolation(self):
        parts = [self.data[self.data.split.eq(x)] for x in ["train", "validation", "holdout"]]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(set(parts[i].device_id) & set(parts[j].device_id))
                self.assertLess(pd.to_datetime(parts[i].timestamp).max(), pd.to_datetime(parts[j].timestamp).min())

    def test_safe_feature_contract(self):
        self.assertFalse(set(benchmark.FEATURES) & {"row_id", "device_id", "failure_next_hour", "post_outcome_score", "split"})

    def test_target_and_missingness(self):
        self.assertEqual(set(self.data.failure_next_hour), {0, 1})
        self.assertGreater(self.data.signal_strength.isna().sum(), 0)
        self.assertFalse(self.data.drop(columns="signal_strength").isna().any().any())

    def test_pipeline_handles_unseen_groups_and_missing_values(self):
        train = self.data[self.data.split.eq("train")]
        test = self.data[self.data.split.eq("validation")]
        model = benchmark.pipeline()
        model.fit(train[benchmark.FEATURES], train.failure_next_hour)
        predicted = model.predict_proba(test[benchmark.FEATURES])[:, 1]
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue(((predicted >= 0) & (predicted <= 1)).all())


class RSNAContractTests(unittest.TestCase):
    def test_mount_discovery_never_recurses_into_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for nesting in ["competition", "competitions/competition"]:
                mount = root / nesting
                mount.mkdir(parents=True)
                (mount / "train.csv").touch()
                (mount / "sample_submission.csv").touch()
                with patch.object(Path, "rglob", side_effect=AssertionError("Recursive image traversal")):
                    self.assertEqual(rsna.resolve_data(root / "missing", root), mount)
                (mount / "train.csv").unlink()

    def labels(self):
        return pd.DataFrame({t: [0.0, 1.0, np.nan] for t in rsna.TARGETS})

    def template(self):
        return pd.DataFrame({rsna.ID: ["synthetic-a", "synthetic-b"], **{t: [0.5, 0.5] for t in rsna.TARGETS}})

    def test_missing_is_not_negative(self):
        self.assertTrue(rsna.priors(self.labels()).eq(0.5).all())

    def test_no_observed_label_is_neutral(self):
        labels = self.labels()
        labels["ACL"] = np.nan
        self.assertEqual(rsna.priors(labels)["ACL"], 0.5)

    def test_invalid_labels_rejected(self):
        labels = self.labels()
        labels.loc[0, "ACL"] = 2
        with self.assertRaises(ValueError):
            rsna.priors(labels)

    def test_template_identity_and_column_order(self):
        template = self.template()
        output = rsna.predict(template, rsna.priors(self.labels()))
        self.assertEqual(list(output), list(template))
        pd.testing.assert_series_equal(output[rsna.ID], template[rsna.ID])

    def test_duplicates_rejected(self):
        template = self.template()
        template[rsna.ID] = "same"
        with self.assertRaises(ValueError):
            rsna.predict(template, rsna.priors(self.labels()))

    def test_nonfinite_predictions_rejected(self):
        fitted = rsna.priors(self.labels())
        fitted["ACL"] = np.nan
        with self.assertRaises(ValueError):
            rsna.predict(self.template(), fitted)


if __name__ == "__main__":
    unittest.main()
