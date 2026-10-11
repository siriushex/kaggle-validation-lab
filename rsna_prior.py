"""RSNA submission-contract baseline; NOT an image model or clinical tool.

Never read the Report column. Missing target labels are unknown, not negatives.
Constant scores cannot beat chance AUC; this tests the code-competition plumbing.
MIT licensed original implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def priors(labels: pd.DataFrame) -> pd.Series:
    if labels.empty or not set(TARGETS).issubset(labels):
        raise ValueError("Training labels missing or empty")
    result = {}
    for target in TARGETS:
        values = labels[target].dropna()
        if not values.isin([0, 1]).all():
            raise ValueError("Non-binary target encountered")
        result[target] = float((values.sum() + 1) / (len(values) + 2))
    return pd.Series(result)


def predict(template: pd.DataFrame, fitted: pd.Series) -> pd.DataFrame:
    if list(template) != [ID] + TARGETS or template.empty:
        raise ValueError("Unexpected submission template")
    if template[ID].isna().any() or not template[ID].is_unique:
        raise ValueError("Invalid study identifiers")
    if not fitted.index.equals(pd.Index(TARGETS)) or not fitted.between(0, 1).all():
        raise ValueError("Invalid probability vector")
    output = template.copy()
    for target in TARGETS:
        output[target] = float(fitted[target])
    return output


def run(data: Path, output: Path) -> dict:
    # Explicit usecols prevents clinical reports being loaded or logged.
    train_path = data / "train.csv"
    labels = pd.read_csv(train_path, usecols=[ID] + TARGETS)
    if labels[ID].isna().any() or not labels[ID].is_unique:
        raise ValueError("Duplicate/missing study identities")
    template = pd.read_csv(data / "sample_submission.csv")
    # Stable identity hash split; no row IDs or medical data are written to logs.
    validation_mask = labels[ID].map(
        lambda value: int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5 == 0)
    development = labels.loc[~validation_mask]
    validation = labels.loc[validation_mask]
    fitted = priors(development)
    aucs = {}
    counts = {}
    for target in TARGETS:
        observed = validation[target].dropna()
        counts[target] = len(observed)
        aucs[target] = float(roc_auc_score(observed, np.full(len(observed), fitted[target]))) if observed.nunique() == 2 else None
    usable = [x for x in aucs.values() if x is not None]
    submission = predict(template, priors(labels))
    if not submission[ID].equals(template[ID]):
        raise ValueError("Submission row order changed")
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)
    report = {
        "purpose": "Submission contract smoke baseline only, not a competitive or clinical model",
        "metric": "macro ROC AUC; incomplete local coverage is not official metric",
        "local_auc": float(np.mean(usable)) if len(usable) == len(TARGETS) else None,
        "covered_targets": len(usable), "auc_by_target": aucs,
        "validation_observed_counts": counts,
        "training_rows": len(labels), "missing_labels_left_unknown": int(labels[TARGETS].isna().sum().sum()),
        "submission_rows": len(submission), "clinical_reports_read": False,
        "images_read": False, "internet_required": False, "gpu_required": False,
        "submitted": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256((output / "submission.csv").read_bytes()).hexdigest(),
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
