"""Original synthetic device benchmark for group/time leakage demonstrations.

Data are simulated, not collected from users, patients, hardware or competitions.
The generator and evaluation code are MIT licensed; generated tables are CC0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 731
NUMERIC = ["load", "temperature", "vibration", "signal_strength", "age_hours"]
CATEGORICAL = ["device_family"]
FEATURES = NUMERIC + CATEGORICAL
DESCRIPTIONS = {
    "row_id": "Unique synthetic observation ID; not a predictive feature.",
    "device_id": "Synthetic device identity. Use for grouping, not as a deployable signal.",
    "timestamp": "UTC observation timestamp; features are known at this instant.",
    "device_family": "Synthetic equipment family A, B or C, known at prediction time.",
    "load": "Simulated normalized workload, bounded from 0 to 1.",
    "temperature": "Simulated temperature in degrees Celsius; not a measurement.",
    "vibration": "Simulated nonnegative vibration index; arbitrary units.",
    "signal_strength": "Simulated radio signal strength in dBm; includes missing values.",
    "age_hours": "Synthetic equipment age in hours at observation time.",
    "failure_next_hour": "Simulated binary target: failure in the following hour.",
    "post_outcome_score": "INTENTIONAL LEAK: available only after the outcome. Never use in a valid model.",
    "split": "Frozen group-and-time partition: train, validation or holdout.",
}


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    for group in range(240):
        family = ["A", "B", "C"][group % 3]
        latent_risk = rng.normal(0, 1.3)
        initial_age = rng.uniform(500, 9000)
        times = range(48) if group < 168 else (range(48, 60) if group < 204 else range(60, 72))
        split = "train" if group < 168 else ("validation" if group < 204 else "holdout")
        for hour in times:
            load = float(rng.beta(2, 2))
            temperature = 35 + 24 * load + rng.normal(0, 5)
            vibration = max(0.0, 0.8 * load + 0.15 * latent_risk + rng.normal(0, 0.25))
            signal = -60 - 20 * load + rng.normal(0, 5)
            age = initial_age + hour
            drift = 0.25 if hour >= 48 else 0.0
            logit = -2.2 + 2.1 * load + 1.5 * vibration + latent_risk + drift
            probability = 1 / (1 + np.exp(-logit))
            target = int(rng.random() < probability)
            rows.append({
                "row_id": f"g{group:03d}-h{hour:03d}",
                "device_id": f"device-{group:03d}",
                "timestamp": (origin + pd.Timedelta(hours=hour)).isoformat(),
                "device_family": family,
                "load": round(load, 6),
                "temperature": round(float(temperature), 6),
                "vibration": round(float(vibration), 6),
                "signal_strength": round(float(signal), 6) if rng.random() > 0.05 else np.nan,
                "age_hours": round(float(age), 6),
                "failure_next_hour": target,
                "post_outcome_score": round(target + float(rng.uniform(-0.1, 0.1)), 6),
                "split": split,
            })
    return pd.DataFrame(rows)


def pipeline(extra_numeric: list[str] | None = None, with_identity: bool = False) -> Pipeline:
    numeric = NUMERIC + (extra_numeric or [])
    categorical = CATEGORICAL + (["device_id"] if with_identity else [])
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocessing),
                     ("model", LogisticRegression(max_iter=500, random_state=SEED))])


def metrics(y: pd.Series, probability: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(y, probability)), "rows": len(y)}


def evaluate(frame: pd.DataFrame) -> dict:
    """Predeclared demonstrations. No hyperparameter search on frozen holdout."""
    train = frame.loc[frame.split.eq("train")]
    validation = frame.loc[frame.split.eq("validation")]
    holdout = frame.loc[frame.split.eq("holdout")]
    honest = pipeline()
    honest.fit(train[FEATURES], train.failure_next_hour)
    honest_validation = metrics(validation.failure_next_hour, honest.predict_proba(validation[FEATURES])[:, 1])
    honest_holdout = metrics(holdout.failure_next_hour, honest.predict_proba(holdout[FEATURES])[:, 1])
    shuffled_train, shuffled_test = train_test_split(frame, test_size=0.25, random_state=SEED,
                                                    stratify=frame.failure_next_hour)
    identity = pipeline(with_identity=True)
    identity.fit(shuffled_train[FEATURES + ["device_id"]], shuffled_train.failure_next_hour)
    leaky = pipeline(extra_numeric=["post_outcome_score"])
    leaky.fit(train[FEATURES + ["post_outcome_score"]], train.failure_next_hour)
    return {
        "seed": SEED,
        "purpose": "Teaching benchmark, not evidence of real-world predictive quality",
        "frozen_validation": honest_validation,
        "frozen_holdout_once": honest_holdout,
        "invalid_random_rows_with_identity": metrics(
            shuffled_test.failure_next_hour,
            identity.predict_proba(shuffled_test[FEATURES + ["device_id"]])[:, 1]),
        "invalid_post_outcome_feature": metrics(
            holdout.failure_next_hour,
            leaky.predict_proba(holdout[FEATURES + ["post_outcome_score"]])[:, 1]),
        "guardrails": {"group_overlap": 0, "temporal_overlap": False,
                       "target_in_safe_features": False, "post_outcome_in_safe_features": False},
    }


def make_notebook() -> dict:
    def cell(kind: str, text: str) -> dict:
        result = {"cell_type": kind, "metadata": {}, "source": text}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "cells": [
                cell("markdown", "# Grouped time-series leakage lab\n\nOriginal, fully synthetic data. "
                     "Compare a frozen group/time split against two deliberately invalid evaluations. "
                     "This is an educational example, not a device-failure model.\n\n"
                     "Data: CC0. Code: MIT. Author account: irynavs. No real device, medical or competition data.\n\n"
                     "[Generator and tests](https://github.com/siriushex/kaggle-validation-lab). "
                     "The public holdout is for demonstration only; it is not a secret competition test set."),
                cell("code", "from pathlib import Path\nimport importlib.util, json, hashlib, platform\n"
                     "import numpy as np\nimport pandas as pd\nimport sklearn\n"
                     "candidates = list(Path('/kaggle/input').rglob('benchmark.py')) if Path('/kaggle/input').exists() else []\n"
                     "generator = candidates[0] if candidates else Path('benchmark.py')\n"
                     "spec = importlib.util.spec_from_file_location('benchmark', generator)\n"
                     "benchmark = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(benchmark)\n"
                     "csv_path = generator.parent / 'observations.csv'\n"
                     "data = pd.read_csv(csv_path) if csv_path.exists() else benchmark.generate()\n"
                     "print({'rows': len(data), 'python': platform.python_version(), 'numpy': np.__version__, "
                     "'pandas': pd.__version__, 'sklearn': sklearn.__version__})\ndata.groupby('split').size()"),
                cell("markdown", "## 1. Validate the deployment contract\n\n"
                     "New devices at later times are the intended deployment setting. "
                     "Fit imputation, scaling and encoding on training rows only. "
                     "Identifiers and post-outcome values are excluded from deployable features."),
                cell("code", "parts = {s: data[data['split'].eq(s)] for s in ['train', 'validation', 'holdout']}\n"
                     "assert data.row_id.is_unique\n"
                     "assert not set(parts['train'].device_id) & set(parts['validation'].device_id)\n"
                     "assert not set(parts['train'].device_id) & set(parts['holdout'].device_id)\n"
                     "assert pd.to_datetime(parts['train'].timestamp).max() < pd.to_datetime(parts['validation'].timestamp).min()\n"
                     "assert pd.to_datetime(parts['validation'].timestamp).max() < pd.to_datetime(parts['holdout'].timestamp).min()\n"
                     "assert 'post_outcome_score' not in benchmark.FEATURES\nprint('Contract checks passed')"),
                cell("markdown", "## 2. Run predeclared comparisons\n\n"
                     "The honest pipeline uses the same fixed configuration for validation and one holdout evaluation. "
                     "The intentionally leaky experiments demonstrate why good-looking numbers may be invalid. "
                     "Do not tune configuration by rerunning the holdout."),
                cell("code", "results = benchmark.evaluate(data)\n"
                     "table = pd.DataFrame({k: v for k, v in results.items() if isinstance(v, dict) and 'roc_auc' in v}).T\n"
                     "print(table.round(4).to_string())\n"
                     "Path('validation-results.json').write_text(json.dumps(results, indent=2))\n"
                     "repro = {'source_sha256': hashlib.sha256(generator.read_bytes()).hexdigest(), "
                     "'seed': benchmark.SEED, 'python': platform.python_version(), 'numpy': np.__version__, "
                     "'pandas': pd.__version__, 'sklearn': sklearn.__version__}\n"
                     "Path('environment.json').write_text(json.dumps(repro, indent=2))"),
                cell("markdown", "## 3. Interpret before optimising\n\n"
                     "A near-perfect score from the post-outcome feature is a known bug, not a breakthrough. "
                     "Random row splitting shares device identities and ignores the deployment horizon. "
                     "Random and frozen evaluations also cover different populations, so their difference "
                     "is not a causal estimate of leakage.\n\n"
                     "Suggested exercises: remove one valid feature; measure calibration; plot missingness; "
                     "add a predeclared temporal embargo. Use new independent data for model selection. "
                     "Synthetic results do not establish performance on real hardware or Kaggle competitions."),
            ]}


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(output / "observations.csv", index=False)
    pd.DataFrame([{"column": k, "description": v} for k, v in DESCRIPTIONS.items()]).to_csv(
        output / "data_dictionary.csv", index=False)
    (output / "benchmark.py").write_bytes(Path(__file__).read_bytes())
    (output / "starter.ipynb").write_text(json.dumps(make_notebook(), indent=2))
    description = """# Grouped Time-Series Leakage Benchmark

## What this dataset is
An original, fully synthetic benchmark for learning reliable validation on repeated device observations. It contains 8,928 simulated rows, 240 fictitious devices, a binary next-hour outcome, missing sensor values, and a mild future distribution shift. No real devices, people, patients, competition data or third-party records are included.

## Why use it
Practise group-aware and time-aware splitting, train-only preprocessing, calibration, and detection of post-outcome leakage. The `post_outcome_score` column is deliberately invalid and clearly labelled. It must never be used in an honest predictive pipeline.

## Files and columns
`observations.csv` is the main table. `data_dictionary.csv` describes every column. `benchmark.py` recreates the data with NumPy seed 731 and runs fixed baseline comparisons. `starter.ipynb` is a step-by-step executable walkthrough. Data are CC0; generator and notebook code are MIT.

## Frozen evaluation protocol
Training: 168 devices at hours 0-47. Validation: 36 different devices at hours 48-59. Holdout: another 36 devices at hours 60-71. Groups and time intervals do not overlap. Fit all transformations only on training. Use validation for decisions and evaluate a frozen candidate once on holdout. This public educational holdout is not secret and must not be treated as an unbiased final test after repeated tuning.

## Provenance and reproducibility
Created specifically for this validation lab using an original probabilistic simulator; no scraped, clinical or personal data. Source and tests: https://github.com/siriushex/kaggle-validation-lab . Fixed snapshot, not a live feed. Tested locally with Python 3.12.12, NumPy 2.5.4, pandas 3.0.6 and scikit-learn 1.9.1; record actual versions and hashes when rerunning. Floating-point/library changes can produce small numerical differences. The standalone notebook records its actual Kaggle environment too.

## Limitations
The failure mechanism, measurement units and missingness are artificial. Scores are demonstrations, not evidence of real-world reliability or competition performance. Random-row and frozen splits cover different populations, so their score difference is not a causal measure of leakage. Do not use this dataset for operational or clinical decisions.

## Suggested work
Compare AUC, Brier score and log loss; analyse errors; test an embargo; compare missing-value strategies. Never publish misleading near-perfect scores obtained from post-outcome features.
"""
    (output / "README.md").write_text(description)
    fields = [{"name": k, "description": v,
               "type": "numeric" if k in NUMERIC + ["post_outcome_score", "failure_next_hour"] else
                       ("datetime" if k == "timestamp" else "string")} for k, v in DESCRIPTIONS.items()]
    metadata = {"id": "irynavs/grouped-time-series-leakage-benchmark",
                "title": "Grouped Time-Series Leakage Benchmark",
                "subtitle": "Synthetic devices for honest group/time validation and leakage detection",
                "description": description, "licenses": [{"name": "CC0-1.0"}],
                "keywords": ["education", "classification", "tabular"],
                "expectedUpdateFrequency": "never",
                "userSpecifiedSources": "Original synthetic simulator, seed 731. No external data. "
                    "https://github.com/siriushex/kaggle-validation-lab",
                "resources": [
                    {"path": "observations.csv", "description": "Synthetic observations and frozen group/time partitions.",
                     "schema": {"fields": fields}},
                    {"path": "data_dictionary.csv", "description": "Complete data dictionary.",
                     "schema": {"fields": [{"name": "column", "description": "Field name in observations.csv", "type": "string"},
                                           {"name": "description", "description": "Meaning and leakage constraints", "type": "string"}]}},
                    {"path": "benchmark.py", "description": "MIT-licensed original generator and evaluation."},
                    {"path": "starter.ipynb", "description": "MIT-licensed reproducible educational walkthrough."},
                    {"path": "README.md", "description": "Data card, provenance, protocol and limitations."},
                    {"path": "LICENSE.txt", "description": "License terms for data and code."},
                ]}
    metadata["data"] = [
        {"name": resource["path"], "description": resource["description"],
         "totalBytes": (output / resource["path"]).stat().st_size if (output / resource["path"]).exists() else 0,
         "columns": resource.get("schema", {}).get("fields", [])}
        for resource in metadata["resources"]
    ]
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"rows": len(data), "devices": data.device_id.nunique(),
            "split_rows": data.groupby("split").size().to_dict(),
            "sha256": hashlib.sha256((output / "observations.csv").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
