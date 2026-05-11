"""
Evaluation metrics and utilities for Bayesian Model Averaging.

This module provides comprehensive evaluation metrics for BMA models including
classification, regression, and uncertainty quantification metrics.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from scipy import stats

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Classification evaluation metrics."""

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics for binary classification
        if len(np.unique(y_true)) == 2:
            metrics["precision_binary"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall_binary"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1_score_binary"] = f1_score(y_true, y_pred, zero_division=0)

        # Probability-based metrics
        if y_proba is not None:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_proba.ndim == 1:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                    metrics["average_precision"] = average_precision_score(y_true, y_proba)
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics["average_precision"] = average_precision_score(y_true, y_proba[:, 1])
            else:
                # Multi-class classification
                metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo")

        return metrics

    @staticmethod
    def compute_calibration_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            Dictionary of calibration metrics
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Use positive class probability

        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        # Brier Score
        brier_score = np.mean((y_proba - y_true) ** 2)

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "brier_score": brier_score,
        }


class RegressionMetrics:
    """Regression evaluation metrics."""

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_std: Predicted standard deviations (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic regression metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)

        # Percentage-based metrics
        try:
            metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        except ValueError:
            metrics["mape"] = np.nan

        # Symmetric MAPE
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        metrics["smape"] = smape

        # Mean Absolute Scaled Error (MASE)
        if len(y_true) > 1:
            naive_error = np.mean(np.abs(np.diff(y_true)))
            if naive_error > 0:
                metrics["mase"] = metrics["mae"] / naive_error
            else:
                metrics["mase"] = np.nan
        else:
            metrics["mase"] = np.nan

        # Uncertainty quantification metrics
        if y_pred_std is not None:
            metrics.update(RegressionMetrics._compute_uncertainty_metrics(y_true, y_pred, y_pred_std))

        return metrics

    @staticmethod
    def _compute_uncertainty_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_std: np.ndarray,
    ) -> Dict[str, float]:
        """Compute uncertainty quantification metrics."""
        metrics = {}

        # Prediction Interval Coverage Probability (PICP)
        alpha = 0.05  # 95% confidence interval
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower_bound = y_pred - z_score * y_pred_std
        upper_bound = y_pred + z_score * y_pred_std
        
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        metrics["picp"] = coverage

        # Mean Prediction Interval Width (MPIW)
        interval_width = upper_bound - lower_bound
        metrics["mpiw"] = np.mean(interval_width)

        # Prediction Interval Normalized Root Mean Square Error (PINAW)
        y_range = np.max(y_true) - np.min(y_true)
        metrics["pinaw"] = metrics["mpiw"] / y_range if y_range > 0 else np.nan

        # Coverage Width-based Criterion (CWC)
        n = len(y_true)
        penalty = 2 * np.exp(-2 * (coverage - (1 - alpha)) * n)
        metrics["cwc"] = metrics["pinaw"] * (1 + penalty)

        return metrics


class UncertaintyMetrics:
    """Uncertainty quantification metrics."""

    @staticmethod
    def compute_ensemble_uncertainty(
        predictions: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute ensemble uncertainty metrics.

        Args:
            predictions: Array of shape (n_samples, n_models) or (n_samples, n_models, n_classes)
            weights: Model weights (optional)

        Returns:
            Dictionary of uncertainty metrics
        """
        if weights is None:
            weights = np.ones(predictions.shape[1]) / predictions.shape[1]

        # Ensure weights sum to 1
        weights = weights / np.sum(weights)

        metrics = {}

        if predictions.ndim == 2:
            # Regression case
            weighted_mean = np.average(predictions, axis=1, weights=weights)
            weighted_var = np.average((predictions - weighted_mean[:, np.newaxis]) ** 2, axis=1, weights=weights)
            
            metrics["mean_prediction"] = np.mean(weighted_mean)
            metrics["mean_variance"] = np.mean(weighted_var)
            metrics["mean_std"] = np.mean(np.sqrt(weighted_var))
            
            # Epistemic uncertainty (disagreement between models)
            epistemic_var = np.var(predictions, axis=1)
            metrics["mean_epistemic_uncertainty"] = np.mean(epistemic_var)
            
        else:
            # Classification case
            weighted_mean = np.average(predictions, axis=1, weights=weights)
            
            # Entropy of the ensemble prediction
            entropy = -np.sum(weighted_mean * np.log(weighted_mean + 1e-10), axis=1)
            metrics["mean_entropy"] = np.mean(entropy)
            
            # Maximum probability
            max_prob = np.max(weighted_mean, axis=1)
            metrics["mean_max_probability"] = np.mean(max_prob)
            
            # Epistemic uncertainty (disagreement between models)
            epistemic_var = np.var(predictions, axis=1)
            metrics["mean_epistemic_uncertainty"] = np.mean(epistemic_var)

        return metrics

    @staticmethod
    def compute_model_agreement(
        predictions: np.ndarray,
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute model agreement metrics.

        Args:
            predictions: Array of shape (n_samples, n_models)
            threshold: Agreement threshold

        Returns:
            Dictionary of agreement metrics
        """
        metrics = {}

        # Pairwise agreement
        n_models = predictions.shape[1]
        agreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if predictions.ndim == 2:
                    # Regression case
                    agreement = np.mean(np.abs(predictions[:, i] - predictions[:, j]) < threshold)
                else:
                    # Classification case
                    agreement = np.mean(predictions[:, i] == predictions[:, j])
                agreements.append(agreement)

        metrics["mean_pairwise_agreement"] = np.mean(agreements)
        metrics["min_pairwise_agreement"] = np.min(agreements)
        metrics["max_pairwise_agreement"] = np.max(agreements)

        # Consensus agreement (all models agree)
        if predictions.ndim == 2:
            # Regression case
            consensus = np.mean(np.all(np.abs(predictions - np.mean(predictions, axis=1, keepdims=True)) < threshold, axis=1))
        else:
            # Classification case
            consensus = np.mean(np.all(predictions == predictions[:, 0:1], axis=1))
        
        metrics["consensus_agreement"] = consensus

        return metrics


class ModelComparison:
    """Model comparison utilities."""

    @staticmethod
    def compare_models(
        results: Dict[str, Dict[str, float]],
        metric: str = "accuracy",
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Compare multiple models using statistical tests.

        Args:
            results: Dictionary of model results
            metric: Metric to compare
            significance_level: Significance level for tests

        Returns:
            Dictionary of comparison results
        """
        model_names = list(results.keys())
        metric_values = [results[model][metric] for model in model_names]

        comparison = {
            "model_names": model_names,
            "metric_values": metric_values,
            "best_model": model_names[np.argmax(metric_values)],
            "worst_model": model_names[np.argmin(metric_values)],
            "metric_range": np.max(metric_values) - np.min(metric_values),
        }

        # Statistical tests
        if len(model_names) > 2:
            # ANOVA test
            try:
                f_stat, p_value = stats.f_oneway(*[results[model][metric] for model in model_names])
                comparison["anova_f_statistic"] = f_stat
                comparison["anova_p_value"] = p_value
                comparison["anova_significant"] = p_value < significance_level
            except Exception as e:
                logger.warning(f"ANOVA test failed: {e}")
                comparison["anova_f_statistic"] = np.nan
                comparison["anova_p_value"] = np.nan
                comparison["anova_significant"] = False

        # Pairwise comparisons
        pairwise_results = []
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    try:
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(
                            [results[model1][metric]],
                            [results[model2][metric]]
                        )
                        pairwise_results.append({
                            "model1": model1,
                            "model2": model2,
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < significance_level,
                        })
                    except Exception as e:
                        logger.warning(f"T-test failed for {model1} vs {model2}: {e}")

        comparison["pairwise_comparisons"] = pairwise_results

        return comparison
