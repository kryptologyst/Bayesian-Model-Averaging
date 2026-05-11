"""
Bayesian Model Averaging package.

This package provides implementations of Bayesian Model Averaging methods
for combining multiple machine learning models with uncertainty quantification.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .models.bma import (
    BayesianModelAveraging,
    CrossValidationBMA,
    BICBMA,
    LaplaceApproximationBMA,
    create_default_models,
)

from .data.loader import DataLoader

from .metrics.evaluation import (
    ClassificationMetrics,
    RegressionMetrics,
    UncertaintyMetrics,
    ModelComparison,
)

from .train.pipeline import BMAPipeline

from .viz.plots import BMAVisualizer

__all__ = [
    "BayesianModelAveraging",
    "CrossValidationBMA", 
    "BICBMA",
    "LaplaceApproximationBMA",
    "create_default_models",
    "DataLoader",
    "ClassificationMetrics",
    "RegressionMetrics", 
    "UncertaintyMetrics",
    "ModelComparison",
    "BMAPipeline",
    "BMAVisualizer",
]
