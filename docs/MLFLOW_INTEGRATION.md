# MLflow Integration for Fear Classification Project

## Overview

The Fear Classification project now includes comprehensive MLflow integration for experiment tracking and machine learning lifecycle management. MLflow provides professional-grade capabilities for tracking experiments, logging metrics, managing models, and ensuring reproducibility.

## Features

### 🔬 Experiment Tracking
- **Automatic Run Management**: Each experiment automatically creates an MLflow run with unique identification
- **Parameter Logging**: All hyperparameters, configuration settings, and model parameters are tracked
- **Metric Logging**: Training metrics, validation metrics, and evaluation metrics are captured in real-time
- **Dataset Information**: Data shapes, class distributions, and dataset statistics are recorded

### 📊 Comprehensive Metrics
- **Training Metrics**: Loss, accuracy, validation loss, validation accuracy per epoch
- **Best Performance**: Best validation accuracy and loss with corresponding epochs
- **Evaluation Metrics**: Test accuracy, precision, recall, F1-score, AUC
- **Multiple Thresholds**: Performance metrics for different classification thresholds (0.5, 0.43, 0.35)
- **Confusion Matrices**: Detailed confusion matrices for each threshold

### 🤖 Model Management
- **Model Versioning**: Automatic model versioning with MLflow Model Registry
- **Model Artifacts**: Complete model files with architecture summaries
- **Model Metadata**: Training time, parameters, and performance metrics linked to each model

### ⏱️ Performance Tracking
- **Training Time**: Detailed timing information for training phases
- **Inference Time**: Precise inference timing for performance optimization
- **Runtime Summary**: Comprehensive performance statistics integration

### 📁 Artifact Management
- **Results DataFrames**: Complete results tables as CSV artifacts
- **Feature Importance**: Integrated Gradients feature importance scores
- **Runtime Reports**: JSON artifacts with timing and performance data
- **Model Summaries**: Detailed model architecture descriptions

## Usage

### Basic Integration

The MLflow integration is seamlessly integrated into the existing pipeline. No changes are required to existing code - MLflow tracking happens automatically:

```python
from main import main

# Run experiments with automatic MLflow tracking
main()
```

### Manual MLflow Operations

For advanced usage, you can interact with the MLflow tracker directly:

```python
from mlflow_tracker import mlflow_tracker

# Start a custom run
mlflow_tracker.start_run("Custom_Experiment", "Subject_01", 1)

# Log custom parameters
mlflow_tracker.log_parameters(config)

# Log custom metrics
mlflow_tracker.log_evaluation_metrics(results, inference_time)

# End the run
mlflow_tracker.end_run()
```

## MLflow UI

### Starting the UI

1. Open a terminal in the project directory
2. Run: `mlflow ui`
3. Open your browser to: `http://localhost:5000`
4. Navigate to the 'Fear_Classification_Experiments' experiment

### UI Features

- **Experiment Overview**: Compare multiple runs side-by-side
- **Metric Visualization**: Interactive plots for training curves and metrics
- **Parameter Comparison**: Easy comparison of hyperparameters across runs
- **Model Registry**: Central model management and versioning
- **Artifact Browsing**: Download and examine all logged artifacts

## Directory Structure

```
project_root/
├── mlflow_tracking/           # MLflow backend storage
│   ├── experiments/           # Experiment metadata
│   ├── models/               # Registered models
│   └── artifacts/            # Run artifacts
├── mlflow_tracker.py         # MLflow integration module
└── test_mlflow_integration.py # MLflow test script
```

## Testing the Integration

Run the MLflow integration test to verify everything is working:

```bash
python test_mlflow_integration.py
```

This test will:
- ✅ Verify MLflow installation
- ✅ Test experiment creation
- ✅ Test parameter logging
- ✅ Test metric logging
- ✅ Test artifact logging
- ✅ Provide UI access instructions

## Configuration

### MLflow Settings

The MLflow tracker is configured with the following defaults:
- **Tracking URI**: `./mlflow_tracking` (local directory)
- **Experiment Name**: `Fear_Classification_Experiments`
- **Auto-logging**: Enabled for TensorFlow/Keras models

### Customization

You can customize MLflow behavior by modifying `mlflow_tracker.py`:

```python
# Change experiment name
mlflow_tracker = MLflowTracker("Custom_Experiment_Name")

# Use remote tracking server
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
```

## Best Practices

### 🎯 Experiment Organization
- Each subject/test combination gets a unique run
- Consistent naming convention: `{run_name}_{subject}_{test_number}`
- Proper tagging for easy filtering and searching

### 📈 Metric Strategy
- Log metrics at multiple levels (epoch, run, experiment)
- Include both training and validation metrics
- Track multiple evaluation thresholds for robust analysis

### 🔄 Reproducibility
- All hyperparameters and configuration settings are logged
- Random seeds and environment information tracked
- Complete dataset information for reproducibility

### 🚀 Performance Optimization
- Efficient logging with error handling
- Non-blocking logging that doesn't affect training performance
- Configurable logging levels for different use cases

## Error Handling

The MLflow integration includes comprehensive error handling:
- **Graceful Degradation**: If MLflow fails, experiments continue without tracking
- **Warning Messages**: Clear warnings when MLflow operations fail
- **Fallback Modes**: Automatic fallback to local logging when needed

## Benefits

### 🔬 Research Benefits
- **Experiment Comparison**: Easy comparison of different approaches
- **Parameter Optimization**: Track what works and what doesn't
- **Reproducibility**: Ensure experiments can be replicated exactly
- **Collaboration**: Share experiments and results with team members

### 🏭 Production Benefits
- **Model Registry**: Centralized model management
- **Version Control**: Track model evolution over time
- **Deployment Tracking**: Monitor model performance in production
- **Audit Trail**: Complete history of model development

### 📊 Analysis Benefits
- **Performance Trends**: Track improvements over time
- **Statistical Analysis**: Export data for deeper statistical analysis
- **Visualization**: Rich plots and charts for presentations
- **Reporting**: Automated report generation for stakeholders

## Integration with Existing Features

The MLflow integration works seamlessly with all existing features:
- ✅ **Runtime Tracking**: MLflow logs complement the existing runtime summary
- ✅ **Simulated Data**: Works with both real and simulated datasets
- ✅ **Feature Importance**: Integrated Gradients results are logged as artifacts
- ✅ **Model Evaluation**: All evaluation metrics are tracked and versioned
- ✅ **Results Export**: Traditional CSV exports continue alongside MLflow tracking

## Next Steps

Future enhancements could include:
- Remote MLflow server deployment
- Advanced model staging and promotion workflows
- Automated hyperparameter optimization with MLflow
- Integration with cloud storage for artifacts
- Custom MLflow plugins for domain-specific tracking
