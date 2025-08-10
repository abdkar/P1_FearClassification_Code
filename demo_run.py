#!/usr/bin/env python3
import os, time, random, json, yaml, subprocess, sys, platform, hashlib
# Enable MLflow system metrics BEFORE importing mlflow
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"  # seconds
from pathlib import Path
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _safe_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        if dirty:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"

def main():
    # Base organized artifact root
    ARTIFACT_ROOT = Path("artifacts_local/demo")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_stamp_dir = ARTIFACT_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
    run_stamp_dir.mkdir(exist_ok=True)
    mlflow.set_experiment("Real_Time_Demo")
    with mlflow.start_run(run_name="Baseline_Training_Run"):
        params = {
            "model_type": "CNN_LSTM",
            "optimizer": "adam",
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 1e-3,
            "seed": 42
        }
        random.seed(params["seed"])
        np.random.seed(params["seed"])
        mlflow.log_params(params)
        # Tags for traceability
        mlflow.set_tags({
            "git_commit": _safe_git_commit(),
            "data_version": "v1_demo",
            "docker_image": os.environ.get("DOCKER_IMAGE", "not-set"),
            "mlflow_version": mlflow.__version__,
            "python_version": sys.version.split()[0]
        })
        history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}
        loss = 1.0
        acc = 0.4
        for epoch in range(1, 21):
            loss *= 0.90 + random.uniform(-0.02, 0.02)
            acc = min(0.95, acc + random.uniform(0.01, 0.04))
            val_loss = loss * (1.0 + random.uniform(-0.05, 0.10))
            val_acc = min(0.95, acc + random.uniform(-0.03, 0.03))
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            history["epoch"].append(epoch)
            history["train_loss"].append(loss)
            history["train_acc"].append(acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            _ = sum(i*i for i in range(20000))
            time.sleep(0.1)
        final_metrics = {
            "test_accuracy": val_acc + random.uniform(-0.01, 0.02),
            "test_loss": val_loss * random.uniform(0.9, 1.05)
        }
        mlflow.log_metrics(final_metrics)
        # -------- Artifact logging --------
        # Place artifacts inside organized folder
        out_dir = run_stamp_dir
        out_dir.mkdir(exist_ok=True)
        # Params YAML / Config
        config_path = out_dir / "config.yaml"
        config_path.write_text(yaml.dump(params, sort_keys=False))
        mlflow.log_artifact(str(config_path))
        # Metrics CSV
        df = pd.DataFrame(history)
        metrics_path = out_dir / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(str(metrics_path))
        # Learning curves
        plt.figure(figsize=(6,4))
        plt.plot(df.epoch, df.train_loss, label="train_loss")
        plt.plot(df.epoch, df.val_loss, label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend(); plt.tight_layout()
        loss_fig = out_dir / "loss_curve.png"; plt.savefig(loss_fig, dpi=120); plt.close(); mlflow.log_artifact(str(loss_fig))
        plt.figure(figsize=(6,4))
        plt.plot(df.epoch, df.train_acc, label="train_acc")
        plt.plot(df.epoch, df.val_acc, label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend(); plt.tight_layout()
        acc_fig = out_dir / "accuracy_curve.png"; plt.savefig(acc_fig, dpi=120); plt.close(); mlflow.log_artifact(str(acc_fig))
        # Dummy confusion matrix
        cm = np.array([[random.randint(40,60), random.randint(0,10)], [random.randint(0,10), random.randint(40,60)]])
        np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        mlflow.log_artifact(str(out_dir / "confusion_matrix.csv"))
        # Misclassified examples (synthetic)
        miscls_rows = []
        for i in range(15):
            true_label = random.choice([0,1])
            pred_label = 1-true_label if random.random()<0.7 else true_label  # force many miscls
            prob = round(random.uniform(0.4,0.9),3)
            miscls_rows.append({"id": i, "true": true_label, "pred": pred_label, "prob": prob})
        miscls_df = pd.DataFrame(miscls_rows)
        miscls_path = out_dir / "misclassified_examples.csv"
        miscls_df.to_csv(miscls_path, index=False)
        mlflow.log_artifact(str(miscls_path))
        # Visual confusion matrix
        plt.figure(figsize=(3,3))
        plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
        plt.tight_layout()
        cm_fig = out_dir / "confusion_matrix.png"; plt.savefig(cm_fig, dpi=140); plt.close(); mlflow.log_artifact(str(cm_fig))
        # Summary JSON
        summary = {"final": final_metrics, "best_val_acc": max(history["val_acc"]), "best_val_loss": min(history["val_loss"])}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(out_dir / "summary.json"))
        # Dataset + preprocessing metadata
        dataset_meta = {
            "dataset_name": "DemoDataset",
            "data_version": "v1_demo",
            "num_samples_train": 1000,
            "num_samples_val": 200,
            "num_samples_test": 300,
            "class_mapping": {"0": "Negative", "1": "Positive"},
            "split_seed": params["seed"],
            "preprocessing": ["normalize", "window=128", "augment=no"],
        }
        (out_dir / "dataset_metadata.json").write_text(json.dumps(dataset_meta, indent=2))
        mlflow.log_artifact(str(out_dir / "dataset_metadata.json"))
        # Data splits (synthetic ids)
        splits = {"train_ids": list(range(0,50)), "val_ids": list(range(50,60)), "test_ids": list(range(60,80))}
        (out_dir / "data_splits.json").write_text(json.dumps(splits, indent=2))
        mlflow.log_artifact(str(out_dir / "data_splits.json"))
        # Preprocessing description
        (out_dir / "preprocessing.md").write_text("""# Preprocessing Steps\n- Normalization\n- Fixed window length = 128\n- No augmentation applied (demo)\n""")
        mlflow.log_artifact(str(out_dir / "preprocessing.md"))
        # Environment / requirements snapshot
        try:
            req_text = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30).decode()
            (out_dir / "requirements_frozen.txt").write_text(req_text)
            mlflow.log_artifact(str(out_dir / "requirements_frozen.txt"))
        except Exception as e:
            (out_dir / "requirements_frozen.txt").write_text(f"failed: {e}")
            mlflow.log_artifact(str(out_dir / "requirements_frozen.txt"))
        # Minimal model architecture (dummy) + weights
        architecture = """DummyModel(\n  input_dim=128,\n  layers=[Conv1D(32), ReLU, MaxPool, LSTM(64), Dense(2)]\n)"""
        (out_dir / "model_architecture.txt").write_text(architecture)
        mlflow.log_artifact(str(out_dir / "model_architecture.txt"))
        weights = {"W1": np.random.randn(32,3).tolist(), "b1": np.random.randn(32).tolist()}
        (out_dir / "model_weights.json").write_text(json.dumps(weights)[:5000])  # truncate for demo
        mlflow.log_artifact(str(out_dir / "model_weights.json"))
        # Hash of metrics file for integrity
        metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        (out_dir / "integrity.json").write_text(json.dumps({"metrics_csv_sha256": metrics_hash}, indent=2))
        mlflow.log_artifact(str(out_dir / "integrity.json"))
        # allow system metrics thread to flush final samples
        time.sleep(2)
        print("Demo run finished. Reproducibility artifacts logged at", out_dir)

if __name__ == "__main__":
    main()
