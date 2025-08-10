# Fear Classification using CNNs with Integrated Gradients

This project implements a machine learning pipeline for fear classification using Convolutional Neural Networks (CNNs) with model interpretability through Integrated Gradients.

## Project Structure

The codebase has been refactored into modular components for better maintainability and reusability:

```
├── main.py                    # Main execution script
├── config.py                  # Configuration settings and hyperparameters
├── data_loader.py            # Data loading and preprocessing utilities
├── model.py                  # CNN model definition
├── trainer.py                # Model training logic
├── evaluator.py              # Model evaluation and results generation
├── integrated_gradients.py   # Integrated gradients implementation for interpretability
├── runtime_tracker.py        # Runtime tracking and performance monitoring
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── Orginal_ode.py           # Original monolithic script (for reference)
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for each functionality
- **CNN Architecture**: Custom 1D CNN model for time-series fear classification
- **Model Interpretability**: Integrated Gradients implementation for feature importance analysis
- **Multiple Evaluation Metrics**: Comprehensive evaluation with different thresholds and metrics
- **Automated Experiments**: Support for running multiple experiments across subjects and test cases
- **GPU Support**: Automatic GPU detection and mixed precision training
- **Comprehensive Logging**: Detailed results saving and visualization
- **Runtime Tracking**: Comprehensive performance monitoring with detailed timing statistics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd P1_FearClassification_Code
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run all experiments with default configuration:
```bash
python main.py
```

### Custom Configuration

Modify the `config.py` file to adjust:
- Hyperparameters (learning rate, batch size, epochs)
- Data paths
- Model architecture parameters
- Class weights
- Evaluation thresholds

### Running Single Experiments

You can also run experiments for specific subjects by modifying the main script:

```python
from config import Config
from main import run_single_experiment

config = Config()
name = "H_106"  # Subject name
test_n = 1      # Test number

run_single_experiment(config, name, test_n)
```

## Configuration

The `config.py` file contains all configurable parameters:

- **Data Settings**: Normalization, standardization options
- **Model Hyperparameters**: Learning rate, batch size, epochs, patience values
- **Architecture**: Model layers, dropout rates, regularization
- **Evaluation**: Different probability thresholds for classification
- **Paths**: Data directories and output folders

## Model Architecture

The CNN model consists of:
1. Three 1D Convolutional blocks with batch normalization and dropout
2. Max pooling layer
3. Global average pooling
4. Dense layers with regularization
5. Softmax output for binary classification

## Interpretability

The project includes Integrated Gradients implementation for model interpretability:
- Computes feature importance scores
- Provides insights into which input features contribute most to predictions
- Saves results for further analysis

## Output

The pipeline generates:
- **Model files**: Trained models saved in .h5 format
- **Results tables**: Excel files with predictions and metrics
- **Feature importance**: CSV files with integrated gradients results
- **Visualizations**: Training plots and learning curves
- **Comprehensive metrics**: Precision, recall, F1-score, confusion matrices
- **Runtime Summary**: Detailed timing statistics for performance analysis

## Runtime Tracking

The project includes comprehensive runtime tracking that monitors:

### Timing Metrics
- **Total Runtime**: Complete execution time for all cross-validation folds
- **Average Time per Fold**: Mean execution time per fold
- **Average Training Time**: Mean training time per fold
- **Average Inference Time**: Mean prediction time per test set

### Runtime Summary Output
```
============================================================
1D-CNN Cross-Validation Runtime Summary
============================================================
Metric                                   Time (seconds)  Time (minutes) 
============================================================
Total Runtime for 60 Folds               1012.05         16.87
Average Time per Fold                    16.86           0.28
Average Training Time per Fold           15.56           0.26
Average Inference Time per Test Set      0.0827          -
============================================================
```

### Features
- Automatic timing of training and inference phases
- Cross-validation performance statistics
- Runtime summary saved to `runtime_summary.txt`
- Console output with formatted timing table
- Performance monitoring for optimization

## Data Structure

Expected data structure:
```
SH_Wrok_Data/
├── Leav_one_24_0811_1/
│   ├── Leav_{Subject}/
│   │   ├── HF/Train/{Land}_Train/
│   │   ├── LF/Train/{Land}_Train/
│   │   ├── HF/Test/{Land}_Test/
│   │   └── LF/Test/{Land}_Test/
├── Control_KK_Healthy_24_1/C5/
└── Control_KK_Athletes_24_1/C5/
```

Where:
- `{Subject}`: Subject identifier (e.g., H_106, L_103)
- `{Land}`: Landmark type (Med or Lat)
- HF: High Fear data
- LF: Low Fear data

## Evaluation Metrics

The model is evaluated using:
- Accuracy at different probability thresholds (0.35, 0.43, 0.5)
- Precision, Recall, F1-score (weighted and macro averages)
- Confusion matrices
- Feature importance through Integrated Gradients

## GPU Support

The code automatically detects and configures GPU usage:
- Enables mixed precision training for compatible GPUs
- Falls back to CPU if no GPU is available
- Provides helpful messages for cloud platforms (Colab, Kaggle)

## Contributing

When contributing to this project:
1. Maintain the modular structure
2. Add comprehensive docstrings
3. Update configuration options in `config.py`
4. Test with both GPU and CPU environments
5. Document any new features in this README

## Original Code

The original monolithic script is preserved as `Orginal_ode.py` for reference and comparison purposes.

## License

[Add your license information here]

## Contact

[Add your contact information here]
