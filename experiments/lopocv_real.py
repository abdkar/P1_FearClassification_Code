#!/usr/bin/env python3
"""Real LOPOCV over clean_simulated_data participants (HF_/LF_ only).

This script trains a lightweight CNN per fold using only participant data
(HF_* & LF_*) with optional control augmentation (disabled by default).

Environment variables:
  LOPOCV_SUBJECTS_LIMIT   -> int, limit number of subjects (e.g. 5) for quick test
  INCLUDE_CONTROLS        -> 1|0 include control_data/ as extra low_fear (default 0)
  EPOCHS                  -> int, training epochs (default 350 legacy)
  BATCH_SIZE              -> int, batch size (default 128 legacy)
  LEARNING_RATE           -> float, optimizer LR (default 0.0001 legacy)
  EARLY_STOPPING_PATIENCE -> int, EarlyStopping patience (default 52 legacy)
  REDUCE_LR_PATIENCE      -> int, ReduceLROnPlateau patience (default 9 legacy)
  REDUCE_LR_FACTOR        -> float, LR reduction factor (default 0.1 legacy)
  REDUCE_LR_MIN_LR        -> float, minimum LR (default 1e-7 legacy)
  MANUAL_CLASS_WEIGHTS    -> json or "0.59,3.28" for {0:0.59,1:3.28} (default legacy weights)
  SEED                    -> int, global seed (default 42)
  IG_ENABLE               -> 1|0 compute Integrated Gradients (default 1)
  IG_STEPS                -> int, m_steps for IG path integration (default 100)
  IG_TARGET_CLASS         -> 0|1 target class index for IG (default 1 = HighFear)
  IG_MAX_TEST_SAMPLES     -> int, cap #test samples for IG (0 means all, default 0)

Artifacts per fold:
  metrics.json
  confusion_matrix.csv / .png
  dataset_summary.json
  params.yaml
  + Full reproducibility bundle (added)
  + integrated_gradients_sum.csv (if IG enabled)

Run:
  python experiments/lopocv_real.py

Keep synthetic script (lopocv_rebuild.py) for fast demo.
"""
from __future__ import annotations
import os, json, random, yaml, time, sys, hashlib, io, platform, subprocess, textwrap
from pathlib import Path
import numpy as np
# Enable MLflow system metrics BEFORE importing mlflow
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")
os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "1")
import mlflow
import matplotlib.pyplot as plt

# Add src to path
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from data.clean_data_loader import CleanDataLoader
import tensorflow as tf
from tensorflow.keras import layers, models

# Integrated Gradients (legacy-equivalent) optional import
try:  # pragma: no cover
    from Core_Files.integrated_gradients import IntegratedGradients
except Exception:  # If path not available, IG can be disabled
    IntegratedGradients = None

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:  # pragma: no cover
    confusion_matrix = None
    classification_report = None

# ---------------- Helper functions for reproducibility ---------------- #

def _safe_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"

def _hardware_details():
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/meminfo") as f:
            info["meminfo_first_line"] = f.readline().strip()
    except Exception:
        pass
    return info

def _model_architecture_txt(model):
    buff = io.StringIO()
    def _printer(text, line_break=True):  # Keras may pass line_break kw
        if line_break:
            buff.write(text + "\n")
        else:
            buff.write(text)
    model.summary(print_fn=_printer)
    return buff.getvalue()

def _weights_snapshot(model, limit_layers=4, limit_values=12):
    snap = {}
    for i, w in enumerate(model.weights[:limit_layers]):
        arr = w.numpy().ravel()
        snap[w.name] = {
            "shape": list(w.shape),
            "values_head": [float(x) for x in arr[:limit_values]]
        }
    return snap

def _plot_history(history, out_dir: Path):
    import pandas as pd
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    csv_path = out_dir / "metrics.csv"
    hist_df.to_csv(csv_path, index=False)
    # Loss
    plt.figure(figsize=(5,3))
    plt.plot(hist_df.epoch, hist_df.loss, label="train_loss")
    if 'val_loss' in hist_df: plt.plot(hist_df.epoch, hist_df.val_loss, label="val_loss")
    plt.legend(); plt.title("Loss"); plt.tight_layout()
    loss_png = out_dir / "loss_curve.png"; plt.savefig(loss_png, dpi=130); plt.close()
    # Accuracy
    if 'accuracy' in hist_df:
        plt.figure(figsize=(5,3))
        plt.plot(hist_df.epoch, hist_df.accuracy, label="train_acc")
        if 'val_accuracy' in hist_df: plt.plot(hist_df.epoch, hist_df.val_accuracy, label="val_acc")
        plt.legend(); plt.title("Accuracy"); plt.tight_layout()
        acc_png = out_dir / "accuracy_curve.png"; plt.savefig(acc_png, dpi=130); plt.close()
    else:
        acc_png = None
    return csv_path, loss_png, acc_png

def _log_requirements(out_dir: Path):
    try:
        req = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30).decode()
    except Exception as e:
        req = f"FAILED: {e}"
    p = out_dir / "requirements_frozen.txt"; p.write_text(req)
    return p

def _integrity_hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

# ---------------------------------------------------------------------- #
# Configuration from env (updated to legacy defaults)
SUBJECT_LIMIT = os.getenv("LOPOCV_SUBJECTS_LIMIT")
SUBJECT_LIMIT = int(SUBJECT_LIMIT) if SUBJECT_LIMIT and SUBJECT_LIMIT.isdigit() else None
INCLUDE_CONTROLS = os.getenv("INCLUDE_CONTROLS", "1").lower() in ("1", "true", "yes")
EPOCHS = int(os.getenv("EPOCHS", "350"))  # legacy 350
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))  # legacy 128
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))  # legacy 1e-4
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "52"))  # legacy 52
REDUCE_LR_PATIENCE = int(os.getenv("REDUCE_LR_PATIENCE", "9"))  # legacy 9
REDUCE_LR_FACTOR = float(os.getenv("REDUCE_LR_FACTOR", "0.1"))  # legacy 0.1
REDUCE_LR_MIN_LR = float(os.getenv("REDUCE_LR_MIN_LR", "1e-7"))  # legacy 1e-7
MANUAL_CLASS_WEIGHTS_ENV = os.getenv("MANUAL_CLASS_WEIGHTS", "0.59,3.28")  # legacy weights
GLOBAL_SEED = int(os.getenv("SEED", "42"))
# New: verbosity and target subject selection
TRAIN_VERBOSE = int(os.getenv("VERBOSE", "1"))  # 0 silent, 1 progress bar (per-batch), 2 one line per epoch
TARGET_SUBJECT = os.getenv("TARGET_SUBJECT")

# Parse manual class weights
manual_class_weights = {0: 0.59, 1: 3.28}
try:
    if "," in MANUAL_CLASS_WEIGHTS_ENV and "{" not in MANUAL_CLASS_WEIGHTS_ENV:
        parts = [p.strip() for p in MANUAL_CLASS_WEIGHTS_ENV.split(",")]
        if len(parts) == 2:
            manual_class_weights = {0: float(parts[0]), 1: float(parts[1])}
    else:
        manual_class_weights = {int(k): float(v) for k, v in json.loads(MANUAL_CLASS_WEIGHTS_ENV).items()}
except Exception:
    pass  # fallback to default if parsing fails

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
import tensorflow as tf
tf.random.set_seed(GLOBAL_SEED)

DATA_ROOT = Path("clean_simulated_data")
PARTICIPANTS_DIR = DATA_ROOT / "participants"
if not PARTICIPANTS_DIR.is_dir():
    raise SystemExit(f"Participants directory not found: {PARTICIPANTS_DIR}")

SUBJECTS = sorted([p.name for p in PARTICIPANTS_DIR.iterdir() if p.is_dir() and (p.name.startswith("HF_") or p.name.startswith("LF_"))])
if SUBJECT_LIMIT:
    SUBJECTS = SUBJECTS[:SUBJECT_LIMIT]
if TARGET_SUBJECT:
    SUBJECTS = [s for s in SUBJECTS if s == TARGET_SUBJECT]
    if not SUBJECTS:
        raise SystemExit(f"TARGET_SUBJECT {TARGET_SUBJECT} not found.")

# MLflow experiment
mlflow.set_experiment("Fear_Classification_LOPOCV_Real")


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# IG environment config (unchanged defaults already legacy aligned for m_steps=100)
IG_ENABLE = os.getenv("IG_ENABLE", "1").lower() in ("1","true","yes")
IG_STEPS = int(os.getenv("IG_STEPS", "100"))
IG_TARGET_CLASS = int(os.getenv("IG_TARGET_CLASS", "1"))
IG_MAX_TEST_SAMPLES = int(os.getenv("IG_MAX_TEST_SAMPLES", "0"))


def run_fold(test_subject: str, loader: CleanDataLoader):
    X_train, y_train, X_test, y_test = loader.get_lopocv_split(test_subject)
    if X_test.size == 0 or X_train.size == 0:
        print(f"Skipping {test_subject}: insufficient data (train {X_train.shape}, test {X_test.shape})")
        return
    train_subjects = [s for s in SUBJECTS if s != test_subject]

    # Shuffle training data
    idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[idx], y_train[idx]

    input_shape = X_train.shape[1:]
    model = build_model(input_shape)

    # Use manual legacy class weights instead of dynamic computation
    class_weights = manual_class_weights

    params = {
        'test_subject': test_subject,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'reduce_lr_patience': REDUCE_LR_PATIENCE,
        'reduce_lr_factor': REDUCE_LR_FACTOR,
        'reduce_lr_min_lr': REDUCE_LR_MIN_LR,
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'input_shape': list(input_shape),
        'include_controls': INCLUDE_CONTROLS,
        'class_weights': class_weights,
        'subjects_total': len(SUBJECTS),
        'seed': GLOBAL_SEED,
        'ig_enabled': IG_ENABLE,
        'ig_steps': IG_STEPS if IG_ENABLE else None,
        'ig_target_class': IG_TARGET_CLASS if IG_ENABLE else None,
        'ig_max_test_samples': IG_MAX_TEST_SAMPLES if IG_ENABLE else None,
    }

    with mlflow.start_run(run_name=f"Fold_{test_subject}"):
        # Tag commit & data
        mlflow.set_tags({
            'git_commit': _safe_git_commit(),
            'data_root': str(DATA_ROOT),
            'include_controls': str(INCLUDE_CONTROLS),
            'control_aug_samples': str(loader._last_control_count),
            'fold_type': 'LOPOCV'
        })
        mlflow.log_params({k: v for k,v in params.items() if k not in ('class_weights','input_shape')})
        mlflow.log_param('input_shape', json.dumps(params['input_shape']))
        mlflow.log_param('class_weights', json.dumps(class_weights))

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR)
        ]
        history = model.fit(
            X_train, y_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            verbose=TRAIN_VERBOSE,
            class_weight=class_weights,
            callbacks=callbacks_list
        )

        # Log training curves
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']), start=1):
            mlflow.log_metric('train_loss', float(loss), step=epoch)
            mlflow.log_metric('train_acc', float(acc), step=epoch)
            mlflow.log_metric('val_loss', float(val_loss), step=epoch)
            mlflow.log_metric('val_acc', float(val_acc), step=epoch)

        # Evaluate
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        test_acc = float((y_pred == y_test).mean())
        mlflow.log_metric('test_accuracy', test_acc)

        # Confusion matrix
        if confusion_matrix:
            cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        else:
            cm = np.zeros((2,2), dtype=int)
            for t,p in zip(y_test, y_pred):
                cm[int(t), int(p)] += 1

        # Output directory
        out_dir = Path(f"artifacts_real_{test_subject}")
        out_dir.mkdir(exist_ok=True)

        # Training history & plots
        metrics_csv, loss_png, acc_png = _plot_history(history, out_dir)
        mlflow.log_artifact(str(metrics_csv))
        mlflow.log_artifact(str(loss_png))
        if acc_png: mlflow.log_artifact(str(acc_png))

        # Confusion matrix artifacts
        np.savetxt(out_dir / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
        mlflow.log_artifact(str(out_dir / 'confusion_matrix.csv'))
        plt.figure(figsize=(3.5,3.5))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(cm[i,j]), ha='center', va='center', color='black')
        plt.tight_layout()
        fig_path = out_dir / 'confusion_matrix.png'
        plt.savefig(fig_path, dpi=140); plt.close()
        mlflow.log_artifact(str(fig_path))

        # Misclassified examples
        mis_rows = []
        for i,(t,p,probs) in enumerate(zip(y_test, y_pred, y_prob)):
            if t != p:
                mis_rows.append({
                    'index': i,
                    'true': int(t),
                    'pred': int(p),
                    'prob_pred_class': float(probs[p]),
                    'prob_high_fear': float(probs[1])
                })
        import pandas as pd
        mis_path = out_dir / 'misclassified_examples.csv'
        pd.DataFrame(mis_rows).to_csv(mis_path, index=False)
        mlflow.log_artifact(str(mis_path))

        # Metrics summary JSON
        metrics = {
            'test_accuracy': test_acc,
            'support_low': int((y_test==0).sum()),
            'support_high': int((y_test==1).sum()),
            'confusion_matrix': cm.tolist()
        }
        if classification_report:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics.update({
                'precision_macro': report['macro avg']['precision'],
                'recall_macro': report['macro avg']['recall'],
                'f1_macro': report['macro avg']['f1-score'],
                'precision_weighted': report['weighted avg']['precision'],
                'recall_weighted': report['weighted avg']['recall'],
                'f1_weighted': report['weighted avg']['f1-score']
            })
            mlflow.log_metrics({k: float(v) for k,v in metrics.items() if isinstance(v,(int,float)) and k!='confusion_matrix'})
        (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(out_dir / 'metrics.json'))

        # Dataset metadata
        dataset_meta = {
            'dataset_root': str(DATA_ROOT),
            'num_subjects': len(SUBJECTS),
            'include_controls': INCLUDE_CONTROLS,
            'class_mapping': {0: 'LowFear', 1: 'HighFear'},
            'split_strategy': 'LOPOCV'
        }
        (out_dir / 'dataset_metadata.json').write_text(json.dumps(dataset_meta, indent=2))
        mlflow.log_artifact(str(out_dir / 'dataset_metadata.json'))

        # Data splits
        splits = {'test_subject': test_subject, 'train_subjects': train_subjects}
        (out_dir / 'data_splits.json').write_text(json.dumps(splits, indent=2))
        mlflow.log_artifact(str(out_dir / 'data_splits.json'))

        # Dataset summary (per-subject counts)
        summary_rows = []
        for s in SUBJECTS:
            Xs, ys = loader._load_participant(s)  # noqa: SLF001 (internal ok for summary)
            summary_rows.append({
                'participant': s,
                'total': int(len(ys)),
                'high_fear': int((ys==1).sum()),
                'low_fear': int((ys==0).sum())
            })
        (out_dir / 'dataset_summary.json').write_text(json.dumps(summary_rows, indent=2))
        mlflow.log_artifact(str(out_dir / 'dataset_summary.json'))

        # Model architecture & lightweight weights snapshot
        arch_txt = _model_architecture_txt(model)
        (out_dir / 'model_architecture.txt').write_text(arch_txt)
        mlflow.log_artifact(str(out_dir / 'model_architecture.txt'))
        weights_snap = _weights_snapshot(model)
        (out_dir / 'model_weights.json').write_text(json.dumps(weights_snap, indent=2))
        mlflow.log_artifact(str(out_dir / 'model_weights.json'))

        # Preprocessing description (placeholder)
        (out_dir / 'preprocessing.md').write_text("""# Preprocessing Steps\n- Raw CSV loaded\n- Shape filtered to consistent trials per participant\n- No scaling applied (placeholder)\n- Class weights computed per fold\n""")
        mlflow.log_artifact(str(out_dir / 'preprocessing.md'))

        # Requirements snapshot
        req_path = _log_requirements(out_dir); mlflow.log_artifact(str(req_path))

        # Integrity hash of metrics.csv
        integrity = {'metrics_csv_sha256': _integrity_hash(metrics_csv)}
        (out_dir / 'integrity.json').write_text(json.dumps(integrity, indent=2))
        mlflow.log_artifact(str(out_dir / 'integrity.json'))

        # Hardware details
        (out_dir / 'hardware.json').write_text(json.dumps(_hardware_details(), indent=2))
        mlflow.log_artifact(str(out_dir / 'hardware.json'))

        # Params & config dump
        params_path = out_dir / 'params.yaml'
        params_path.write_text(yaml.dump(params, sort_keys=False))
        mlflow.log_artifact(str(params_path))
        (out_dir / 'config.yaml').write_text(yaml.dump({k:v for k,v in params.items() if k!='class_weights'}, sort_keys=False))
        mlflow.log_artifact(str(out_dir / 'config.yaml'))

        # Training script snapshot
        try:
            script_src = Path(__file__).read_text()
            (out_dir / 'training_script_snapshot.py').write_text(script_src)
            mlflow.log_artifact(str(out_dir / 'training_script_snapshot.py'))
        except Exception:
            pass

        # Repro README summary
        repro_summary = textwrap.dedent("""Logged for reproducibility:\n\n- params.yaml + config.yaml\n- model_architecture.txt\n- model_weights.json (truncated values)\n- metrics.csv + loss_curve.png + accuracy_curve.png\n- metrics.json (final & aggregate)\n- confusion_matrix.csv / .png\n- misclassified_examples.csv\n- dataset_metadata.json + dataset_summary.json\n- data_splits.json\n- preprocessing.md\n- requirements_frozen.txt\n- integrity.json (SHA256)\n- hardware.json\n- training_script_snapshot.py\n- git commit tag\n- system metrics enabled\n""")
        (out_dir / 'README_reproducibility.txt').write_text(repro_summary)
        mlflow.log_artifact(str(out_dir / 'README_reproducibility.txt'))

        # Integrated Gradients computation and artifact logging
        if IG_ENABLE and IntegratedGradients is not None and X_test.size:
            try:
                # Optionally subsample test set for IG if cap provided
                X_ig = X_test
                if IG_MAX_TEST_SAMPLES > 0 and X_ig.shape[0] > IG_MAX_TEST_SAMPLES:
                    X_ig = X_ig[:IG_MAX_TEST_SAMPLES]
                # Prepare config-like object
                class _IGCfg:
                    def __init__(self, m_steps, parameters):
                        self.m_steps = m_steps
                        self.parameters = parameters
                # Feature name placeholders feat_0..feat_{F-1}
                feature_names = [f"feat_{i}" for i in range(input_shape[-1])]
                ig_cfg = _IGCfg(IG_STEPS, feature_names)
                ig = IntegratedGradients(model, ig_cfg)
                ig_sum = ig.compute_feature_importance(X_ig, target_class_idx=IG_TARGET_CLASS)
                # Save CSV (first row headers). Convert to string array for uniform save.
                ig_path = out_dir / 'integrated_gradients_sum.csv'
                # ig_sum is header + values; ensure dtype str for mixed content
                np.savetxt(ig_path, ig_sum, fmt='%s', delimiter=',')
                mlflow.log_artifact(str(ig_path))
            except Exception as e:  # pragma: no cover
                err_path = out_dir / 'integrated_gradients_error.txt'
                err_path.write_text(str(e))
                mlflow.log_artifact(str(err_path))
        elif IG_ENABLE and IntegratedGradients is None:
            # Log that IG module not found
            mlflow.log_param('ig_warning', 'IntegratedGradients module not available')

        # Small delay to allow system metrics thread flush
        time.sleep(0.5)
        print(f"Fold {test_subject}: test_accuracy={test_acc:.3f} cm={cm.tolist()}")


def main():
    print(f"Running REAL LOPOCV on {len(SUBJECTS)} participants (limit={SUBJECT_LIMIT or 'all'}), include_controls={INCLUDE_CONTROLS}")
    loader = CleanDataLoader(root=str(DATA_ROOT), include_controls=INCLUDE_CONTROLS)
    start_all = time.time()
    for subj in SUBJECTS:
        run_fold(subj, loader)
    print(f"Done. Total time {time.time()-start_all:.1f}s. View in MLflow experiment 'Fear_Classification_LOPOCV_Real'.")


if __name__ == "__main__":
    main()
