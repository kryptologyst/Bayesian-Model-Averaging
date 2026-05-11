# Bayesian Model Averaging

A comprehensive implementation of Bayesian Model Averaging (BMA) methods for combining multiple machine learning models with uncertainty quantification.

**Author:** kryptologyst  
**GitHub:** [https://github.com/kryptologyst](https://github.com/kryptologyst)

## ⚠️ Disclaimer

**This is a research and educational demonstration only.**

- **Not intended for production decisions or control systems**
- **Results should not be used for critical decision-making**
- **Models may not generalize to real-world scenarios**
- **Always validate results with domain experts**

## Overview

Bayesian Model Averaging (BMA) is a method for combining multiple models based on their posterior probabilities. Instead of picking a single "best" model, BMA takes the weighted average of the predictions of all candidate models, where the weight for each model is proportional to its posterior probability. This approach improves predictive performance by accounting for model uncertainty.

### Key Features

- **Multiple BMA Methods**: Cross-validation, BIC, and Laplace approximation
- **Comprehensive Evaluation**: Classification and regression metrics with uncertainty quantification
- **Interactive Demo**: Streamlit web interface for exploring BMA methods
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Modern Stack**: Python 3.10+, scikit-learn, PyTorch-compatible, with optional Bayesian libraries

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Bayesian-Model-Averaging.git
cd Bayesian-Model-Averaging

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Optional Dependencies

For advanced Bayesian methods, install additional dependencies:

```bash
# For MCMC-based BMA (Pyro)
pip install pyro-ppl

# For NumPyro-based BMA
pip install numpyro

# For PyMC-based BMA
pip install pymc

# For experiment tracking
pip install wandb mlflow
```

## Quick Start

### Command Line Interface

Run a classification experiment:

```bash
python scripts/run_experiment.py --task classification --dataset breast_cancer
```

Run a regression experiment:

```bash
python scripts/run_experiment.py --task regression --dataset diabetes
```

Run a synthetic experiment:

```bash
python scripts/run_experiment.py --task synthetic
```

### Python API

```python
from src.bma.train.pipeline import BMAPipeline
from src.bma.data.loader import DataLoader

# Initialize pipeline
pipeline = BMAPipeline(random_state=42)

# Run classification experiment
results = pipeline.run_classification_experiment(
    dataset_name="breast_cancer",
    bma_methods=["cv", "bic", "laplace"]
)

# Print results
print(f"Best BMA method: {max(results['bma_results'].items(), key=lambda x: x[1]['accuracy'])}")
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

Then open your browser to `http://localhost:8501` to explore the interactive interface.

## BMA Methods

### 1. Cross-Validation Based BMA

Uses cross-validation scores as posterior probabilities:

```python
from src.bma.models.bma import CrossValidationBMA

bma_model = CrossValidationBMA(models, cv=5)
bma_model.fit(X_train, y_train)
predictions = bma_model.predict(X_test)
```

### 2. Bayesian Information Criterion (BIC) BMA

Uses BIC to compute model weights:

```python
from src.bma.models.bma import BICBMA

bma_model = BICBMA(models)
bma_model.fit(X_train, y_train)
predictions = bma_model.predict(X_test)
```

### 3. Laplace Approximation BMA

Uses Laplace approximation for log evidence:

```python
from src.bma.models.bma import LaplaceApproximationBMA

bma_model = LaplaceApproximationBMA(models)
bma_model.fit(X_train, y_train)
predictions = bma_model.predict(X_test)
```

## Datasets

### Built-in Datasets

**Classification:**
- `breast_cancer`: Wisconsin Breast Cancer dataset
- `wine`: Wine quality dataset
- `iris`: Iris flower dataset
- `synthetic`: Generated synthetic data

**Regression:**
- `diabetes`: Diabetes dataset
- `boston`: Boston housing dataset (deprecated, falls back to synthetic)
- `synthetic`: Generated synthetic data

### Custom Data

Load your own CSV data:

```python
from src.bma.data.loader import DataLoader

loader = DataLoader(random_state=42)
X_train, X_test, y_train, y_test, metadata = loader.load_csv_data(
    file_path="your_data.csv",
    target_column="target"
)
```

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **ROC-AUC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve
- **Calibration Metrics**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score

### Regression Metrics

- **RMSE/MAE**: Root Mean Square Error, Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE/SMAPE**: Mean Absolute Percentage Error, Symmetric MAPE
- **MASE**: Mean Absolute Scaled Error
- **Uncertainty Metrics**: Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW)

### Uncertainty Quantification

- **Ensemble Uncertainty**: Mean variance, epistemic uncertainty
- **Model Agreement**: Pairwise agreement, consensus metrics
- **Calibration**: Reliability diagrams, calibration curves

## Configuration

Use YAML configuration files for reproducible experiments:

```yaml
# configs/my_experiment.yaml
experiment:
  task: "classification"
  dataset: "breast_cancer"
  random_state: 42
  test_size: 0.2
  cv_folds: 5

bma_methods:
  - "cv"
  - "bic"
  - "laplace"

data:
  scale_features: true
```

Run with custom configuration:

```bash
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

## Project Structure

```
bayesian-model-averaging/
├── src/bma/                    # Main package
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                 # BMA model implementations
│   ├── metrics/                # Evaluation metrics
│   ├── train/                  # Training pipelines
│   └── viz/                    # Visualization utilities
├── configs/                    # Configuration files
├── data/                       # Data directory
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── assets/                     # Output directory
│   ├── results/               # Experiment results
│   └── figures/               # Generated plots
├── demo/                       # Demo applications
├── scripts/                    # Command-line scripts
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
└── README.md                   # This file
```

## Examples

### Basic Classification Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.bma.models.bma import CrossValidationBMA, create_default_models

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create BMA model
models = create_default_models(random_state=42)
bma_model = CrossValidationBMA(models, cv=5, random_state=42)

# Train and predict
bma_model.fit(X_train, y_train)
y_pred = bma_model.predict(X_test)
y_proba = bma_model.predict_proba(X_test)

# Get model weights
weights = bma_model.get_model_weights()
print("Model weights:", weights)
```

### Advanced Regression Example

```python
from src.bma.train.pipeline import BMAPipeline
from src.bma.viz.plots import BMAVisualizer

# Initialize pipeline
pipeline = BMAPipeline(random_state=42)

# Run experiment
results = pipeline.run_regression_experiment(
    dataset_name="diabetes",
    bma_methods=["cv", "bic", "laplace"]
)

# Create visualizations
visualizer = BMAVisualizer()
fig = visualizer.plot_model_comparison(results, metric="r2")
fig.savefig("regression_comparison.png")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/
ruff check src/
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_bma.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bayesian_model_averaging,
  title={Bayesian Model Averaging: A Comprehensive Implementation},
  author={kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Bayesian-Model-Averaging}
}
```

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Visualization with [matplotlib](https://matplotlib.org/) and [plotly](https://plotly.com/)
- Interactive demo with [Streamlit](https://streamlit.io/)
- Configuration management with [OmegaConf](https://omegaconf.readthedocs.io/)
# Bayesian-Model-Averaging
