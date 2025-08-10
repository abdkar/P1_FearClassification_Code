#!/usr/bin/env python3
"""Synthetic demo run for Fear Classification.
Generates random metrics and comprehensive reproducibility artifacts under MLflow.
"""
import os, time, random, json, yaml, subprocess, sys, platform, hashlib
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Enable MLflow system metrics BEFORE importing mlflow
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")
os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "1")  # seconds

import mlflow, pandas as pd, matplotlib.pyplot as plt, numpy as np

def _safe_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        if dirty:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"

def _log_repro_artifacts(out_dir: Path, params, history_df, final_metrics, cm):
    """Create and log all reproducibility artifacts to MLflow.
    Keeps main() concise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"; config_path.write_text(yaml.dump(params, sort_keys=False)); mlflow.log_artifact(str(config_path))
    metrics_path = out_dir / "metrics.csv"; history_df.to_csv(metrics_path, index=False); mlflow.log_artifact(str(metrics_path))
    plt.figure(figsize=(6,4)); plt.plot(history_df.epoch, history_df.train_loss, label="train_loss"); plt.plot(history_df.epoch, history_df.val_loss, label="val_loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend(); plt.tight_layout(); loss_fig = out_dir / "loss_curve.png"; plt.savefig(loss_fig, dpi=120); plt.close(); mlflow.log_artifact(str(loss_fig))
    plt.figure(figsize=(6,4)); plt.plot(history_df.epoch, history_df.train_acc, label="train_acc"); plt.plot(history_df.epoch, history_df.val_acc, label="val_acc"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend(); plt.tight_layout(); acc_fig = out_dir / "accuracy_curve.png"; plt.savefig(acc_fig, dpi=120); plt.close(); mlflow.log_artifact(str(acc_fig))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d"); mlflow.log_artifact(str(out_dir / "confusion_matrix.csv"))
    plt.figure(figsize=(3,3)); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.colorbar();
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
    plt.tight_layout(); cm_fig = out_dir / "confusion_matrix.png"; plt.savefig(cm_fig, dpi=140); plt.close(); mlflow.log_artifact(str(cm_fig))
    miscls_rows = []
    for i in range(15):
        t = random.choice([0,1]); p = 1-t if random.random()<0.7 else t; miscls_rows.append({"id": i, "true": t, "pred": p, "prob": round(random.uniform(0.4,0.9),3)})
    pd.DataFrame(miscls_rows).to_csv(out_dir / "misclassified_examples.csv", index=False); mlflow.log_artifact(str(out_dir / "misclassified_examples.csv"))
    summary = {"final": final_metrics, "best_val_acc": float(history_df.val_acc.max()), "best_val_loss": float(history_df.val_loss.min())}; (out_dir / "summary.json").write_text(json.dumps(summary, indent=2)); mlflow.log_artifact(str(out_dir / "summary.json"))
    dataset_meta = {"dataset_name": "DemoDataset", "data_version": "v1_demo", "num_samples_train": 1000, "num_samples_val": 200, "num_samples_test": 300, "class_mapping": {"0": "Negative", "1": "Positive"}, "split_seed": params["seed"], "preprocessing": ["normalize", "window=128", "augment=no"]}
    (out_dir / "dataset_metadata.json").write_text(json.dumps(dataset_meta, indent=2)); mlflow.log_artifact(str(out_dir / "dataset_metadata.json"))
    splits = {"train_ids": list(range(0,50)), "val_ids": list(range(50,60)), "test_ids": list(range(60,80))}; (out_dir / "data_splits.json").write_text(json.dumps(splits, indent=2)); mlflow.log_artifact(str(out_dir / "data_splits.json"))
    (out_dir / "preprocessing.md").write_text("# Preprocessing Steps\n- Normalization\n- Fixed window length = 128\n- No augmentation applied (demo)\n"); mlflow.log_artifact(str(out_dir / "preprocessing.md"))
    try: req_text = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30).decode()
    except Exception as e: req_text = f"failed: {e}"
    (out_dir / "requirements_frozen.txt").write_text(req_text); mlflow.log_artifact(str(out_dir / "requirements_frozen.txt"))
    architecture = "DummyModel(\n  input_dim=128,\n  layers=[Conv1D(32), ReLU, MaxPool, LSTM(64), Dense(2)]\n)"; (out_dir / "model_architecture.txt").write_text(architecture); mlflow.log_artifact(str(out_dir / "model_architecture.txt"))
    weights = {"W1": np.random.randn(32,3).tolist(), "b1": np.random.randn(32).tolist()}; (out_dir / "model_weights.json").write_text(json.dumps(weights)[:5000]); mlflow.log_artifact(str(out_dir / "model_weights.json"))
    metrics_hash = hashlib.sha256((out_dir / "metrics.csv").read_bytes()).hexdigest(); (out_dir / "integrity.json").write_text(json.dumps({"metrics_csv_sha256": metrics_hash}, indent=2)); mlflow.log_artifact(str(out_dir / "integrity.json"))

def main():
    ARTIFACT_ROOT = Path("artifacts_local/demo"); ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_stamp_dir = ARTIFACT_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
    mlflow.set_experiment("Real_Time_Demo")
    with mlflow.start_run(run_name="Baseline_Training_Run"):
        params = {"model_type": "CNN_LSTM", "optimizer": "adam", "batch_size": 32, "epochs": 20, "learning_rate": 1e-3, "seed": 42}
        random.seed(params["seed"]); np.random.seed(params["seed"]); mlflow.log_params(params)
        mlflow.set_tags({"git_commit": _safe_git_commit(), "data_version": "v1_demo", "docker_image": os.environ.get("DOCKER_IMAGE", "not-set"), "mlflow_version": mlflow.__version__, "python_version": sys.version.split()[0], "platform": platform.platform()})
        history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}; loss = 1.0; acc = 0.4
        for epoch in range(1, params["epochs"] + 1):
            loss *= 0.90 + random.uniform(-0.02, 0.02); acc = min(0.95, acc + random.uniform(0.01, 0.04))
            val_loss = loss * (1.0 + random.uniform(-0.05, 0.10)); val_acc = min(0.95, acc + random.uniform(-0.03, 0.03))
            mlflow.log_metric("train_loss", loss, step=epoch); mlflow.log_metric("train_acc", acc, step=epoch); mlflow.log_metric("val_loss", val_loss, step=epoch); mlflow.log_metric("val_acc", val_acc, step=epoch)
            history["epoch"].append(epoch); history["train_loss"].append(loss); history["train_acc"].append(acc); history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
            _ = sum(i*i for i in range(20000)); time.sleep(0.05)
        final_metrics = {"test_accuracy": val_acc + random.uniform(-0.01, 0.02), "test_loss": val_loss * random.uniform(0.9, 1.05)}; mlflow.log_metrics(final_metrics)
        df_history = pd.DataFrame(history); cm = np.array([[random.randint(40,60), random.randint(0,10)], [random.randint(0,10), random.randint(40,60)]])
        _log_repro_artifacts(run_stamp_dir, params, df_history, final_metrics, cm)
        time.sleep(1.0); print("Demo run finished. Reproducibility artifacts logged at", run_stamp_dir)

if __name__ == "__main__":
    main()
