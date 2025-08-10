# Fear Classification Platform

A reproducible research codebase for automated fear state classification from physiological / behavioral time‑series data. The project emphasizes:

- Rigorous leave‑one‑participant‑out cross‑validation (LOPOCV)
- Transparent experiment tracking (MLflow) + full reproducibility bundle
- Explainability via Integrated Gradients (per‑feature importance)
- Configurable legacy / modern training hyperparameters through environment variables
- Containerized (GPU & CPU) execution for portable deployment

> DISCLAIMER: This repository is for research and prototyping only and is **NOT** a validated clinical decision support system. Do not use outputs for medical diagnosis or treatment without independent clinical validation and regulatory clearance.

---
## 1. Key Capabilities

| Area | Features |
|------|----------|
| Data Handling | Clean participant dataset + optional control augmentation |
| Validation | Leave‑One‑Participant‑Out (LOPOCV) real and rebuilt pipelines |
| Training | Legacy hyperparameter profile (350 epochs, batch 128, early stopping, LR plateau scheduling) or override via env vars |
| Class Imbalance | Manual class weights (default `{0:0.59, 1:3.28}`) or configurable |
| Explainability | Integrated Gradients (selectable target class, steps, sampling cap) |
| Tracking | MLflow metrics, parameters, artifacts, system metrics, reproducibility snapshots |
| Repro Artifacts | Model summary, partial weights snapshot, dataset splits, misclassifications, requirements freeze, integrity hash |
| Containerization | GPU (CUDA/cuDNN) + CPU fallback Dockerfiles, docker‑compose with MLflow server |
| Sample Data & Generator | Structured synthetic subset generator (Sim_<ID> / Train/Test / Med|Lat) |

---
## 2. Repository Structure (Condensed)
```
P1_FearClassification_Code/
├── Core_Files/                # Transition legacy core (to be merged into src/)
├── src/                       # Primary future package modules
├── experiments/               # Entry scripts (lopocv_real.py, etc.)
├── clean_simulated_data/      # Participant & control data (real/sim/sample)
├── mlruns/                    # Local MLflow tracking store
├── docker/                    # Dockerfiles (GPU, CPU) & docs
├── docker-compose.yml         # Optional MLflow + app stack
├── Tests/                     # Test utilities
├── docs/ / Documentation/     # Extended technical summaries
└── README.md
```
See `PROJECT_STRUCTURE.md` for a full tree.

---
## 3. Installation (Local, Non‑Docker)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
(Optional) For GPU metrics:
```bash
pip install pynvml
```

### Structured Synthetic Sample Generator
A lightweight, reproducible synthetic subset can be created that mirrors the full data hierarchy (subject‑scoped folders, fear category, Train/Test split, hop‑based file naming, optional Med/Lat land types, optional minimal controls).

Structure example (Med only, 2 HF + 2 LF subjects, 2 test replicates):
```
clean_simulated_data/sample_sim/
  Sim_HF_201/
    HF/Train/Med_Train/HF_201_1.csv ...
    HF/Test/Med_Test/HF_201_9.csv ...
  Sim_LF_202/
    LF/Train/Med_Train/LF_202_5.csv ...
    LF/Test/Med_Test/LF_202_11.csv ...
  ... (optional control folders if --controls)
```

Generate a small sample (Med only):
```bash
python Scripts/generate_sample_data.py \
  --out clean_simulated_data/sample_sim \
  --hf 2 --lf 2 --tests 2 --land Med
```
Add Lat data too:
```bash
python Scripts/generate_sample_data.py --out clean_simulated_data/sample_sim --hf 2 --lf 2 --tests 2 --land Med Lat
```
Include minimal control examples:
```bash
python Scripts/generate_sample_data.py --controls
```
Key arguments:
- `--out`: root output directory (default `clean_simulated_data/sample_sim`)
- `--hf`, `--lf`: number of synthetic HF / LF subjects (IDs auto‑assigned HF_201, HF_203, … / LF_202, LF_204 …)
- `--tests`: number of independent Train/Test splits per subject (replicates)
- `--land`: one or both of `Med`, `Lat` (creates `<land>_Train` & `<land>_Test`)
- `--controls`: adds minimal healthy/athlete control folders with a few files
- `--seed`: RNG seed

Notes:
- Hop sets: Med = [1,3,5,7,9,11,13,15,17,19]; Lat = [0,2,4,6,8,10,12,14,16,18]
- Train/Test split is deterministic per subject & replicate (70/30) via seeded shuffle.
- Feature columns use biomechanical‑style names (e.g., `ankle_mom_X_Med`).
- Synthetic signals are non‑physiologic approximations (sinusoid + harmonics + noise) intended only for code execution examples.

---
## 4. Quick Start
### 4.1 Demo
```bash
python experiments/demo_run.py
```
### 4.2 Full LOPOCV (Rebuild)
```bash
python experiments/lopocv_rebuild.py
```
### 4.3 Real LOPOCV (Legacy‑Aligned Hyperparameters + Repro Artifacts)
```bash
# Example: single subject HF_203 with verbose per‑batch logs
VERBOSE=1 TARGET_SUBJECT=HF_203 python experiments/lopocv_real.py
```
Open MLflow UI in another terminal:
```bash
mlflow ui --port 5000
```
Navigate to: http://localhost:5000

---
## 5. Configurable Environment Variables (`lopocv_real.py`)
| Variable | Default | Description |
|----------|---------|-------------|
| EPOCHS | 350 | Max training epochs |
| BATCH_SIZE | 128 | Batch size |
| LEARNING_RATE | 0.0001 | Adam LR |
| EARLY_STOPPING_PATIENCE | 52 | Patience (restore best weights) |
| REDUCE_LR_PATIENCE | 9 | Plateau patience (val_loss) |
| REDUCE_LR_FACTOR | 0.1 | LR reduction factor |
| REDUCE_LR_MIN_LR | 1e-7 | Minimum LR |
| MANUAL_CLASS_WEIGHTS | 0.59,3.28 | Comma list or JSON `{\"0\":0.59,\"1\":3.28}` |
| VERBOSE | 2 | Keras verbosity (0,1,2) |
| TARGET_SUBJECT | (unset) | Restrict run to a single participant (e.g. HF_203) |
| LOPOCV_SUBJECTS_LIMIT | (unset) | Limit # subjects for smoke tests |
| INCLUDE_CONTROLS | 1 | Include control augmentation (1/0) |
| SEED | 42 | Global seed |
| IG_ENABLE | 1 | Enable Integrated Gradients |
| IG_STEPS | 100 | Path interpolation steps (m_steps) |
| IG_TARGET_CLASS | 1 | Target class index (e.g., 1 = HighFear) |
| IG_MAX_TEST_SAMPLES | 0 | Cap IG test samples (0 = all) |

Example multi‑var run:
```bash
EPOCHS=200 BATCH_SIZE=64 VERBOSE=1 IG_ENABLE=1 TARGET_SUBJECT=HF_201 \
python experiments/lopocv_real.py
```

---
## 6. Explainability (Integrated Gradients)
When enabled, each fold run computes feature attribution:
- `integrated_gradients_sum.csv` (header + aggregated importance)
- Top feature indices & importance metrics logged to MLflow parameters/metrics
- Adjustable via IG_* environment variables

Use cases:
- Inspect stability of top features across folds
- Support model interpretation & scientific reporting

---
## 7. MLflow Artifact Bundle (per Fold)
| File | Purpose |
|------|---------|
| metrics.csv / loss_curve.png / accuracy_curve.png | Epoch trajectories |
| metrics.json | Final evaluation summary |
| confusion_matrix.* | Confusion matrix counts + visualization |
| misclassified_examples.csv | Error analysis |
| dataset_metadata.json / dataset_summary.json | Dataset snapshot |
| data_splits.json | Train vs test subject listing |
| model_architecture.txt | Full Keras model summary |
| model_weights.json | Truncated weight snapshot (sanity check) |
| params.yaml / config.yaml | Captured hyperparameters & config |
| requirements_frozen.txt | Environment lock snapshot |
| preprocessing.md | Data preprocessing narrative |
| integrity.json | SHA256 of metrics.csv for tamper detection |
| hardware.json | Host hardware context |
| training_script_snapshot.py | Exact source of executed script |
| integrated_gradients_sum.csv (if IG enabled) | Explainability artifact |

---
## 8. Docker
### 8.1 GPU Image
```bash
docker build -t fear-classification:gpu -f docker/Dockerfile .
docker run --gpus all --rm -v $(pwd):/app -w /app \
  -e TARGET_SUBJECT=HF_203 -e VERBOSE=1 fear-classification:gpu
```
### 8.2 CPU Image
```bash
docker build -t fear-classification:cpu -f docker/Dockerfile.cpu .
docker run --rm -v $(pwd):/app -w /app fear-classification:cpu
```
### 8.3 Compose (MLflow + App)
```bash
docker compose up --build
# MLflow at http://localhost:5000
```
Override environment in `docker-compose.yml` or with `docker compose run -e VAR=...`.

---
## 9. Development Guidelines
1. Prefer adding new logic under `src/`; migrate items from `Core_Files/` progressively.
2. Maintain deterministic seeds (NumPy, Python `random`, TensorFlow) when modifying training loops.
3. Log every new critical metric to MLflow for comparability.
4. Keep Integrated Gradients interface consistent (avoid changing array format).
5. Add/adjust tests in `Tests/` for new behaviors.
6. Document new env variables in this README.

### Code Style
- PEP8 / black formatting
- Type hints encouraged
- Minimal side effects in module import scope

---
## 10. Roadmap
- Consolidate legacy & src pipelines
- Add cross‑fold aggregation + statistical summary script
- Add calibration & ROC plotting utilities
- Package release (pip installable) with CLI interface

---
## 11. Citation (Placeholder)
If you use this repository, please cite:
```
@misc{fear_classification_platform,
  title  = {Fear Classification Platform},
  author = {Your Team},
  year   = {2025},
  url    = {https://github.com/your_org/fear-classification}
}
```

---
## 12. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## 13. Contact
For questions / collaboration: [your_email@example.com]

---
## 14. Clinical & Ethical Note
Model interpretations are probabilistic and subject to dataset biases. Always corroborate automated outputs with expert assessment. Ensure compliance with data protection and ethical review requirements.

---
**End of README**
