# Project Structure

```
P1_FearClassification_Code/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── src/                        # Main source code package
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── clean_data_loader.py
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/               # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── eval/                   # Evaluation logic
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── utils.py
│
├── experiments/                # Experiment scripts
│   ├── demo_run.py            # Demo experiment
│   └── lopocv_rebuild.py      # LOPOCV experiment
│
├── Tests/                      # Test files
│   ├── test_mlflow_visualization.py
│   ├── test_standard_mlflow.py
│   └── export_db_to_mlflow.py
│
├── docs/                       # Documentation
│   ├── CLEAN_DATA_IMPLEMENTATION_SUMMARY.md
│   ├── DATA_LEAKAGE_FIX_SUMMARY.md
│   ├── MLFLOW_IMPLEMENTATION_SUMMARY.md
│   └── [other documentation files]
│
├── clean_simulated_data/       # Clean dataset
│   ├── participants/           # Participant data (5-10 trials each)
│   └── control_data/          # Control group data
│
├── Old_ver/                    # Legacy code versions
│   ├── main.py
│   ├── clean_main.py
│   └── [other legacy files]
│
├── artifacts/                  # Experiment artifacts
│   └── experiments/
│
├── mlruns/                     # MLflow tracking
├── results/                    # Result outputs
├── notebooks/                  # Jupyter notebooks
├── docker/                     # Docker configuration
├── Core_Files/                 # Core reference files
├── Demos/                      # Demo files
├── Documentation/              # Additional documentation
└── Scripts/                    # Utility scripts
```

## Key Components

### Source Code (`src/`)
- **data/**: CleanDataLoader with control data integration
- **models/**: Neural network model definitions
- **training/**: Training logic with LOPOCV
- **eval/**: Evaluation and metrics
- **utils/**: Utility functions
- **config.py**: Centralized configuration

### Experiments (`experiments/`)
- **demo_run.py**: Quick demonstration script
- **lopocv_rebuild.py**: Full LOPOCV experiment

### Data (`clean_simulated_data/`)
- **participants/**: 62 participants with 5-10 trials each
- **control_data/**: 111 control samples for augmentation

### Testing (`Tests/`)
- Unit tests and integration tests
- MLflow testing utilities

## Usage

1. Run quick demo: `python experiments/demo_run.py`
2. Run full LOPOCV: `python experiments/lopocv_rebuild.py`
3. View results: Check `mlruns/` or use MLflow UI
