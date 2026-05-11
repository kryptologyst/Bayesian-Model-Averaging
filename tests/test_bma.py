"""
Tests for Bayesian Model Averaging implementation.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from src.bma.models.bma import (
    CrossValidationBMA,
    BICBMA,
    LaplaceApproximationBMA,
    create_default_models,
)
from src.bma.data.loader import DataLoader
from src.bma.metrics.evaluation import ClassificationMetrics, RegressionMetrics


class TestBMA:
    """Test cases for BMA models."""

    def setup_method(self):
        """Setup test data."""
        # Classification data
        X_clf, y_clf = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=42
        )
        self.X_clf_train, self.X_clf_test, self.y_clf_train, self.y_clf_test = train_test_split(
            X_clf, y_clf, test_size=0.3, random_state=42
        )

        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )

        # Create models
        self.clf_models = create_default_models(random_state=42)
        self.reg_models = [
            ("linear", LinearRegression()),
            ("ridge", Ridge(random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]

    def test_cross_validation_bma_classification(self):
        """Test CrossValidationBMA for classification."""
        bma = CrossValidationBMA(self.clf_models, cv=3, random_state=42)
        bma.fit(self.X_clf_train, self.y_clf_train)
        
        # Test predictions
        y_pred = bma.predict(self.X_clf_test)
        y_proba = bma.predict_proba(self.X_clf_test)
        
        assert len(y_pred) == len(self.y_clf_test)
        assert y_proba.shape == (len(self.y_clf_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        
        # Test model weights
        weights = bma.get_model_weights()
        assert len(weights) == len(self.clf_models)
        assert np.allclose(sum(weights.values()), 1.0)

    def test_bic_bma_classification(self):
        """Test BICBMA for classification."""
        bma = BICBMA(self.clf_models, random_state=42)
        bma.fit(self.X_clf_train, self.y_clf_train)
        
        # Test predictions
        y_pred = bma.predict(self.X_clf_test)
        y_proba = bma.predict_proba(self.X_clf_test)
        
        assert len(y_pred) == len(self.y_clf_test)
        assert y_proba.shape == (len(self.y_clf_test), 2)
        
        # Test model weights
        weights = bma.get_model_weights()
        assert len(weights) == len(self.clf_models)
        assert np.allclose(sum(weights.values()), 1.0)

    def test_laplace_bma_classification(self):
        """Test LaplaceApproximationBMA for classification."""
        bma = LaplaceApproximationBMA(self.clf_models, random_state=42)
        bma.fit(self.X_clf_train, self.y_clf_train)
        
        # Test predictions
        y_pred = bma.predict(self.X_clf_test)
        y_proba = bma.predict_proba(self.X_clf_test)
        
        assert len(y_pred) == len(self.y_clf_test)
        assert y_proba.shape == (len(self.y_clf_test), 2)
        
        # Test model weights
        weights = bma.get_model_weights()
        assert len(weights) == len(self.clf_models)
        assert np.allclose(sum(weights.values()), 1.0)

    def test_data_loader(self):
        """Test DataLoader functionality."""
        loader = DataLoader(random_state=42)
        
        # Test synthetic classification data
        X_train, X_test, y_train, y_test, metadata = loader.generate_synthetic_classification(
            n_samples=100, n_features=5, test_size=0.2
        )
        
        assert X_train.shape[0] == 80  # 80% of 100
        assert X_test.shape[0] == 20   # 20% of 100
        assert X_train.shape[1] == 5
        assert metadata["n_samples"] == 100
        assert metadata["n_features"] == 5

    def test_classification_metrics(self):
        """Test classification metrics."""
        # Create dummy predictions
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
        
        metrics = ClassificationMetrics.compute_all_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        
        # Test calibration metrics
        calib_metrics = ClassificationMetrics.compute_calibration_metrics(y_true, y_proba)
        assert "expected_calibration_error" in calib_metrics
        assert "brier_score" in calib_metrics

    def test_regression_metrics(self):
        """Test regression metrics."""
        # Create dummy predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = RegressionMetrics.compute_all_metrics(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
