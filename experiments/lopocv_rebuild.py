#!/usr/bin/env python3
import os, time, random, json, yaml
from pathlib import Path
# Enable system metrics BEFORE importing mlflow
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"  # seconds
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import hashlib, subprocess, sys, platform, textwrap

# Root directory containing participant folders (HF_/LF_)
DATA_ROOT = Path("clean_simulated_data/participants_backup_20250807_105421")

def discover_subjects(limit: int | None = None):
    subs = []
    if DATA_ROOT.exists():
        for p in DATA_ROOT.iterdir():
            if p.is_dir() and (p.name.startswith("HF_") or p.name.startswith("LF_")):
                subs.append(p.name)
    subs.sort()
    if limit:
        subs = subs[:limit]
    return subs

# Allow limiting via env var for quicker test runs
_limit_env = os.environ.get("LOPOCV_SUBJECTS_LIMIT")
_subject_limit = int(_limit_env) if _limit_env and _limit_env.isdigit() and int(_limit_env) > 0 else None
SUBJECTS = discover_subjects(_subject_limit)
if not SUBJECTS:
    # Fallback to previous small demo list if discovery failed
    SUBJECTS = ["H_106", "H_108", "H_109"]

EPOCHS = 30


def _collect_subject_file_counts(subjects):
    counts = {}
    for s in subjects:
        subj_path = DATA_ROOT / s
        if not subj_path.exists():
            counts[s] = {"high_fear": 0, "low_fear": 0, "total": 0}
            continue
        high_dir = subj_path / "high_fear"
        low_dir = subj_path / "low_fear"
        hf = len(list(high_dir.glob("*.csv"))) if high_dir.exists() else 0
        lf = len(list(low_dir.glob("*.csv"))) if low_dir.exists() else 0
        counts[s] = {"high_fear": hf, "low_fear": lf, "total": hf + lf}
    return counts


def _safe_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        if dirty:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


def _hardware_details():
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    # memory (best effort)
    try:
        with open("/proc/meminfo") as f:
            first = f.readline().strip()
            info["meminfo_first_line"] = first
    except Exception:
        pass
    return info


def run_fold(test_subject: str):
    train_subjects = [s for s in SUBJECTS if s != test_subject]
    # Deterministic seed per subject
    seed = abs(hash(test_subject)) % (2**32)
    random.seed(seed)
    with mlflow.start_run(run_name=f"Fold_Test_{test_subject}"):
        mlflow.set_tags({
            "git_commit": _safe_git_commit(),
            "data_version": DATA_ROOT.name,
            "test_subject": test_subject,
            "discovered_subjects": str(len(SUBJECTS)),
        })
        params = {
            "test_subject": test_subject,
            "num_train_subjects": len(train_subjects),
            "epochs": EPOCHS,
            "model": "CNN_LSTM",
            "lr": 1e-3,
            "batch_size": 32,
            "discovered_subjects": len(SUBJECTS),
            "subject_limit": _subject_limit if _subject_limit else "all",
            "seed": seed,
        }
        mlflow.log_params(params)
        history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}
        loss = 1.0
        acc = 0.35
        for epoch in range(1, EPOCHS + 1):
            loss *= 0.92 + random.uniform(-0.03, 0.03)
            acc = min(0.94, acc + random.uniform(0.015, 0.035))
            val_loss = loss * (1 + random.uniform(-0.07, 0.08))
            val_acc = min(0.95, acc + random.uniform(-0.04, 0.04))
            for k, v in {
                "train_loss": loss,
                "train_acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }.items():
                mlflow.log_metric(k, v, step=epoch)
            history["epoch"].append(epoch)
            history["train_loss"].append(loss)
            history["train_acc"].append(acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            # Small compute to keep process busy for system metrics sampling
            _ = sum(i * i for i in range(40000))  # increase a bit for system metrics
            time.sleep(0.05)
        final_metrics = {
            "test_accuracy": val_acc + random.uniform(-0.015, 0.02),
            "test_loss": val_loss * random.uniform(0.88, 1.08),
            "seed": seed,
        }
        mlflow.log_metrics(final_metrics)
        # Artifacts
        out_dir = Path(f"artifacts_{test_subject}")
        out_dir.mkdir(exist_ok=True)
        (out_dir / "params.yaml").write_text(yaml.dump(params, sort_keys=False))
        mlflow.log_artifact(str(out_dir / "params.yaml"))
        df = pd.DataFrame(history)
        metrics_path = out_dir / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(str(metrics_path))
        plt.figure(figsize=(5, 3))
        plt.plot(df.epoch, df.train_loss, label="train_loss")
        plt.plot(df.epoch, df.val_loss, label="val_loss")
        plt.legend(); plt.title("Loss"); plt.tight_layout()
        lf = out_dir / "loss_curve.png"; plt.savefig(lf, dpi=130); plt.close(); mlflow.log_artifact(str(lf))
        plt.figure(figsize=(5, 3))
        plt.plot(df.epoch, df.train_acc, label="train_acc")
        plt.plot(df.epoch, df.val_acc, label="val_acc")
        plt.legend(); plt.title("Accuracy"); plt.tight_layout()
        af = out_dir / "accuracy_curve.png"; plt.savefig(af, dpi=130); plt.close(); mlflow.log_artifact(str(af))
        summary = {"final": final_metrics, "best_val_acc": max(history["val_acc"]), "best_val_loss": min(history["val_loss"])}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(out_dir / "summary.json"))
        # Dataset summary artifact (counts of csvs for train/test subjects)
        dataset_summary = {
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "file_counts": _collect_subject_file_counts([test_subject] + train_subjects),
        }
        (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
        mlflow.log_artifact(str(out_dir / "dataset_summary.json"))
        # ===== Reproducibility additions =====
        # Model architecture (dummy) & weights
        architecture = textwrap.dedent(
            """DummyModel(
  input_dim=128,
  layers=[Conv1D(32), ReLU, MaxPool, LSTM(64), Dense(2)]
)"""
        )
        (out_dir / "model_architecture.txt").write_text(architecture)
        mlflow.log_artifact(str(out_dir / "model_architecture.txt"))
        weights = {"W1": [random.random() for _ in range(32)], "b1": [random.random() for _ in range(32)]}
        (out_dir / "model_weights.json").write_text(json.dumps(weights))
        mlflow.log_artifact(str(out_dir / "model_weights.json"))
        # Confusion matrix (synthetic)
        import numpy as np
        cm = np.array([[random.randint(40,60), random.randint(0,10)], [random.randint(0,10), random.randint(40,60)]])
        np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        mlflow.log_artifact(str(out_dir / "confusion_matrix.csv"))
        # Visual confusion matrix
        plt.figure(figsize=(3,3))
        plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
        plt.tight_layout()
        cm_fig = out_dir / "confusion_matrix.png"; plt.savefig(cm_fig, dpi=140); plt.close(); mlflow.log_artifact(str(cm_fig))
        # Misclassified examples (synthetic)
        miscls_rows = []
        for i in range(15):
            true_label = random.choice([0,1])
            pred_label = 1-true_label if random.random()<0.7 else true_label
            prob = round(random.uniform(0.4,0.9),3)
            miscls_rows.append({"id": i, "true": true_label, "pred": pred_label, "prob": prob})
        pd.DataFrame(miscls_rows).to_csv(out_dir / "misclassified_examples.csv", index=False)
        mlflow.log_artifact(str(out_dir / "misclassified_examples.csv"))
        # Dataset metadata & splits
        dataset_meta = {
            "dataset_root": str(DATA_ROOT),
            "num_subjects": len(SUBJECTS),
            "class_mapping": {"0": "LowFear", "1": "HighFear"},
            "split_strategy": "LOPOCV",
        }
        (out_dir / "dataset_metadata.json").write_text(json.dumps(dataset_meta, indent=2))
        mlflow.log_artifact(str(out_dir / "dataset_metadata.json"))
        splits = {"test_subject": test_subject, "train_subjects": train_subjects}
        (out_dir / "data_splits.json").write_text(json.dumps(splits, indent=2))
        mlflow.log_artifact(str(out_dir / "data_splits.json"))
        # Preprocessing description (placeholder)
        (out_dir / "preprocessing.md").write_text("""# Preprocessing Steps\n- Normalization (placeholder)\n- Windowing\n- No augmentation in demo\n""")
        mlflow.log_artifact(str(out_dir / "preprocessing.md"))
        # Environment / requirements snapshot
        try:
            req_text = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=25).decode()
        except Exception as e:
            req_text = f"failed: {e}"
        (out_dir / "requirements_frozen.txt").write_text(req_text)
        mlflow.log_artifact(str(out_dir / "requirements_frozen.txt"))
        # Integrity hash of metrics
        metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        (out_dir / "integrity.json").write_text(json.dumps({"metrics_csv_sha256": metrics_hash}, indent=2))
        mlflow.log_artifact(str(out_dir / "integrity.json"))
        # Hardware details
        (out_dir / "hardware.json").write_text(json.dumps(_hardware_details(), indent=2))
        mlflow.log_artifact(str(out_dir / "hardware.json"))
        # Script snapshot
        try:
            script_src = Path(__file__).read_text()
            (out_dir / "training_script_snapshot.py").write_text(script_src)
            mlflow.log_artifact(str(out_dir / "training_script_snapshot.py"))
        except Exception:
            pass
        # README summary
        repro_summary = textwrap.dedent("""Reproducibility artifacts logged:\n- params.yaml (hyperparameters)\n- metrics.csv + loss_curve.png + accuracy_curve.png\n- summary.json (final & best metrics)\n- model_architecture.txt + model_weights.json\n- confusion_matrix.csv / .png\n- misclassified_examples.csv\n- dataset_summary.json + dataset_metadata.json + data_splits.json\n- preprocessing.md\n- requirements_frozen.txt (environment)\n- hardware.json\n- integrity.json (hash)\n- training_script_snapshot.py\n- git commit tag (run tag git_commit)\n- system metrics enabled\n""")
        (out_dir / "README_reproducibility.txt").write_text(repro_summary)
        mlflow.log_artifact(str(out_dir / "README_reproducibility.txt"))
        # ===== End reproducibility additions =====
        # Allow system metric thread to flush
        time.sleep(1.5)
        print(f"Completed fold for {test_subject}")


def main():
    mlflow.set_experiment("Fear_Classification_3_LOPOCV_Rebuilt")
    print(f"Running LOPOCV over {len(SUBJECTS)} subjects (limit={_subject_limit if _subject_limit else 'all'})")
    for subj in SUBJECTS:
        run_fold(subj)
    print("All LOPOCV folds finished. View at http://localhost:5002")


if __name__ == "__main__":
    main()
