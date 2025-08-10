#!/usr/bin/env python3
"""Aggregate results for real LOPOCV runs (experiment: Fear_Classification_LOPOCV_Real).

Collects all runs, extracts metrics (test_accuracy, macro/weighted scores if present),
parses metrics.json artifacts (confusion matrix + supports), and produces:
  aggregation_real/summary_runs.csv
  aggregation_real/aggregate_confusion_matrix.csv
  aggregation_real/aggregate_metrics.json
  aggregation_real/README_aggregation.txt

Usage:
  python experiments/aggregate_lopocv_real.py

Optional environment:
  EXP_NAME  -> override experiment name

Requires mlflow >= 2.x.
"""
from __future__ import annotations
import os, json, csv, statistics, math
from pathlib import Path
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

EXP_NAME = os.getenv("EXP_NAME", "Fear_Classification_LOPOCV_Real")
OUT_DIR = Path("aggregation_real")
OUT_DIR.mkdir(exist_ok=True)

client = MlflowClient()
exp = mlflow.get_experiment_by_name(EXP_NAME)
if exp is None:
    raise SystemExit(f"Experiment '{EXP_NAME}' not found. Run lopocv_real first.")

runs = client.search_runs([exp.experiment_id], max_results=10000, order_by=["attributes.start_time ASC"])
if not runs:
    raise SystemExit("No runs found to aggregate.")

rows = []
conf_mats = []
missing_cm = 0
for r in runs:
    rid = r.info.run_id
    params = r.data.params
    metrics = r.data.metrics
    test_subj = params.get("test_subject", "unknown")
    test_acc = metrics.get("test_accuracy")
    # Try artifact metrics.json first
    cm = None
    try:
        local_metrics_json = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="metrics.json")
        with open(local_metrics_json, "r") as f:
            mdata = json.load(f)
        if isinstance(mdata.get("confusion_matrix"), list):
            cm_arr = np.array(mdata["confusion_matrix"], dtype=int)
            if cm_arr.shape == (2,2):
                cm = cm_arr
    except Exception:
        pass
    if cm is None:
        # fallback confusion_matrix.csv
        try:
            cm_path = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="confusion_matrix.csv")
            cm = np.loadtxt(cm_path, delimiter=",", dtype=int)
        except Exception:
            missing_cm += 1
            cm = None
    if cm is not None:
        conf_mats.append(cm)
    row = {
        "run_id": rid,
        "test_subject": test_subj,
        "test_accuracy": test_acc,
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "precision_weighted": metrics.get("precision_weighted"),
        "recall_weighted": metrics.get("recall_weighted"),
        "f1_weighted": metrics.get("f1_weighted")
    }
    rows.append(row)

# Aggregate confusion matrix
if conf_mats:
    agg_cm = np.sum(conf_mats, axis=0)
else:
    agg_cm = np.zeros((2,2), dtype=int)

# Compute summary stats
accuracies = [r["test_accuracy"] for r in rows if r["test_accuracy"] is not None]
mean_acc = float(statistics.fmean(accuracies)) if accuracies else math.nan
std_acc = float(statistics.pstdev(accuracies)) if len(accuracies) > 1 else 0.0

macro_f1s = [r["f1_macro"] for r in rows if r["f1_macro"] is not None]
mean_f1_macro = float(statistics.fmean(macro_f1s)) if macro_f1s else None

weighted_f1s = [r["f1_weighted"] for r in rows if r["f1_weighted"] is not None]
mean_f1_weighted = float(statistics.fmean(weighted_f1s)) if weighted_f1s else None

summary = {
    "experiment": EXP_NAME,
    "num_runs": len(rows),
    "mean_test_accuracy": mean_acc,
    "std_test_accuracy": std_acc,
    "mean_f1_macro": mean_f1_macro,
    "mean_f1_weighted": mean_f1_weighted,
    "confusion_matrix_aggregate": agg_cm.tolist(),
    "confusion_matrix_missing_runs": missing_cm
}

# Write CSV of runs
csv_path = OUT_DIR / "summary_runs.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Write aggregate confusion matrix
cm_path = OUT_DIR / "aggregate_confusion_matrix.csv"
np.savetxt(cm_path, agg_cm, fmt="%d", delimiter=",")

# Write metrics JSON
json_path = OUT_DIR / "aggregate_metrics.json"
json_path.write_text(json.dumps(summary, indent=2))

# Human readable README
readme_path = OUT_DIR / "README_aggregation.txt"
readme_path.write_text(f"""Aggregation Report for {EXP_NAME}\n\nRuns: {len(rows)}\nMean test accuracy: {mean_acc:.4f}\nStd test accuracy: {std_acc:.4f}\nMean F1 macro: {mean_f1_macro}\nMean F1 weighted: {mean_f1_weighted}\nAggregate confusion matrix (rows=true [Low,High], cols=pred):\n{agg_cm}\nMissing confusion matrices: {missing_cm}\nArtifacts generated:\n- summary_runs.csv\n- aggregate_confusion_matrix.csv\n- aggregate_metrics.json\n""")

print("Aggregation complete:")
print("  ", csv_path)
print("  ", cm_path)
print("  ", json_path)
print("  ", readme_path)
