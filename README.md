# Grouped time-series validation lab

Original synthetic device data for learning group/time splits and detecting target leakage.
No medical, personal, third-party or competition data are distributed here.

Data: https://www.kaggle.com/datasets/irynavs/grouped-time-series-leakage-benchmark
Run: `python benchmark.py --output dataset`, then open `starter.ipynb`.
Tests: `python -m unittest discover -p test_assets.py -v`.
`rsna_prior.py` is an original, data-free submission-contract example, not an image model or clinical tool.
Its constant probabilities cannot beat chance AUC. Do not publish restricted RSNA inputs/outputs.

The notebook fits imputation/encoding/scaling on training only and compares a fixed
honest baseline with two explicitly invalid demonstrations. A public teaching
holdout cannot remain unbiased after repeated tuning. Synthetic scores do not
establish real-world or competition performance.

Generated CSV: CC0. Original code: MIT. See LICENSE.txt.
Agent-assisted preparation is disclosed; it is not evidence the account holder completed a course personally.
