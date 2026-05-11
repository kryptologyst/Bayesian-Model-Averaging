"""
Training and evaluation pipeline for Bayesian Model Averaging.

This module provides comprehensive training and evaluation pipelines for BMA models
with support for multiple datasets, models, and evaluation metrics.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from ..data.loader import DataLoader
from ..models.bma import (
    BayesianModelAveraging,
    CrossValidationBMA,
    BICBMA,
    LaplaceApproximationBMA,
    create_default_models,
)
from ..metrics.evaluation import (
    ClassificationMetrics,
    RegressionMetrics,
    UncertaintyMetrics,
    ModelComparison,
)

logger = logging.getLogger(__name__)


class BMAPipeline:
    """Pipeline for training and evaluating Bayesian Model Averaging models."""

    def __init__(
        self,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize BMA Pipeline.

        Args:
            random_state: Random state for reproducibility
            verbose: Whether to print progress information
        """
        self.random_state = random_state
        self.verbose = verbose
        self.data_loader = DataLoader(random_state=random_state)
        self.results: Dict[str, Any] = {}

        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def run_classification_experiment(
        self,
        dataset_name: str,
        models: Optional[List[Tuple[str, BaseEstimator]]] = None,
        bma_methods: Optional[List[str]] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a complete classification experiment.

        Args:
            dataset_name: Name of the dataset to use
            models: List of models to use (if None, uses default models)
            bma_methods: List of BMA methods to use (if None, uses all methods)
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of results
        """
        logger.info(f"Starting classification experiment on {dataset_name}")

        # Load data
        X_train, X_test, y_train, y_test, metadata = self.data_loader.load_classification_dataset(
            dataset_name, test_size=test_size
        )

        # Use default models if none provided
        if models is None:
            models = create_default_models(random_state=self.random_state)

        # Use all BMA methods if none provided
        if bma_methods is None:
            bma_methods = ["cv", "bic", "laplace"]

        results = {
            "dataset": dataset_name,
            "metadata": metadata,
            "models": [name for name, _ in models],
            "bma_methods": bma_methods,
            "individual_model_results": {},
            "bma_results": {},
        }

        # Evaluate individual models
        logger.info("Evaluating individual models...")
        for name, model in models:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            # Compute metrics
            metrics = ClassificationMetrics.compute_all_metrics(y_test, y_pred, y_proba)
            if y_proba is not None:
                calibration_metrics = ClassificationMetrics.compute_calibration_metrics(y_test, y_proba)
                metrics.update(calibration_metrics)
            
            training_time = time.time() - start_time
            metrics["training_time"] = training_time
            
            results["individual_model_results"][name] = metrics
            
            if self.verbose:
                logger.info(f"Model {name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")

        # Evaluate BMA methods
        logger.info("Evaluating BMA methods...")
        for method in bma_methods:
            logger.info(f"Evaluating BMA method: {method}")
            
            # Create BMA model
            if method == "cv":
                bma_model = CrossValidationBMA(models, cv=cv_folds, random_state=self.random_state)
            elif method == "bic":
                bma_model = BICBMA(models, random_state=self.random_state)
            elif method == "laplace":
                bma_model = LaplaceApproximationBMA(models, random_state=self.random_state)
            else:
                logger.warning(f"Unknown BMA method: {method}")
                continue
            
            start_time = time.time()
            
            # Train BMA model
            bma_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = bma_model.predict(X_test)
            y_proba = bma_model.predict_proba(X_test)
            
            # Compute metrics
            metrics = ClassificationMetrics.compute_all_metrics(y_test, y_pred, y_proba)
            calibration_metrics = ClassificationMetrics.compute_calibration_metrics(y_test, y_proba)
            metrics.update(calibration_metrics)
            
            # Get model weights
            model_weights = bma_model.get_model_weights()
            metrics["model_weights"] = model_weights
            
            training_time = time.time() - start_time
            metrics["training_time"] = training_time
            
            results["bma_results"][method] = metrics
            
            if self.verbose:
                logger.info(f"BMA {method}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
                logger.info(f"Model weights: {model_weights}")

        # Compare models
        logger.info("Comparing models...")
        all_results = {**results["individual_model_results"], **results["bma_results"]}
        comparison = ModelComparison.compare_models(all_results, metric="accuracy")
        results["comparison"] = comparison

        self.results[dataset_name] = results
        return results

    def run_regression_experiment(
        self,
        dataset_name: str,
        models: Optional[List[Tuple[str, BaseEstimator]]] = None,
        bma_methods: Optional[List[str]] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a complete regression experiment.

        Args:
            dataset_name: Name of the dataset to use
            models: List of models to use (if None, uses default models)
            bma_methods: List of BMA methods to use (if None, uses all methods)
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of results
        """
        logger.info(f"Starting regression experiment on {dataset_name}")

        # Load data
        X_train, X_test, y_train, y_test, metadata = self.data_loader.load_regression_dataset(
            dataset_name, test_size=test_size
        )

        # Use default models if none provided
        if models is None:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.svm import SVR
            
            models = [
                ("random_forest", RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ("linear_regression", LinearRegression()),
                ("ridge", Ridge(random_state=self.random_state)),
                ("svr", SVR()),
            ]

        # Use all BMA methods if none provided
        if bma_methods is None:
            bma_methods = ["cv", "bic", "laplace"]

        results = {
            "dataset": dataset_name,
            "metadata": metadata,
            "models": [name for name, _ in models],
            "bma_methods": bma_methods,
            "individual_model_results": {},
            "bma_results": {},
        }

        # Evaluate individual models
        logger.info("Evaluating individual models...")
        for name, model in models:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Compute metrics
            metrics = RegressionMetrics.compute_all_metrics(y_test, y_pred)
            
            training_time = time.time() - start_time
            metrics["training_time"] = training_time
            
            results["individual_model_results"][name] = metrics
            
            if self.verbose:
                logger.info(f"Model {name}: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}")

        # Evaluate BMA methods
        logger.info("Evaluating BMA methods...")
        for method in bma_methods:
            logger.info(f"Evaluating BMA method: {method}")
            
            # Create BMA model
            if method == "cv":
                bma_model = CrossValidationBMA(models, cv=cv_folds, random_state=self.random_state)
            elif method == "bic":
                bma_model = BICBMA(models, random_state=self.random_state)
            elif method == "laplace":
                bma_model = LaplaceApproximationBMA(models, random_state=self.random_state)
            else:
                logger.warning(f"Unknown BMA method: {method}")
                continue
            
            start_time = time.time()
            
            # Train BMA model
            bma_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = bma_model.predict(X_test)
            
            # Compute metrics
            metrics = RegressionMetrics.compute_all_metrics(y_test, y_pred)
            
            # Get model weights
            model_weights = bma_model.get_model_weights()
            metrics["model_weights"] = model_weights
            
            training_time = time.time() - start_time
            metrics["training_time"] = training_time
            
            results["bma_results"][method] = metrics
            
            if self.verbose:
                logger.info(f"BMA {method}: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}")
                logger.info(f"Model weights: {model_weights}")

        # Compare models
        logger.info("Comparing models...")
        all_results = {**results["individual_model_results"], **results["bma_results"]}
        comparison = ModelComparison.compare_models(all_results, metric="r2")
        results["comparison"] = comparison

        self.results[dataset_name] = results
        return results

    def run_synthetic_experiment(
        self,
        task_type: str = "classification",
        n_samples: int = 1000,
        n_features: int = 20,
        noise_level: float = 0.1,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Run experiment on synthetic data.

        Args:
            task_type: Type of task ("classification" or "regression")
            n_samples: Number of samples
            n_features: Number of features
            noise_level: Amount of noise
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary of results
        """
        logger.info(f"Starting synthetic {task_type} experiment")

        if task_type == "classification":
            X_train, X_test, y_train, y_test, metadata = self.data_loader.generate_synthetic_classification(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise_level,
                test_size=test_size,
            )
            return self.run_classification_experiment("synthetic", test_size=test_size)
        else:
            X_train, X_test, y_train, y_test, metadata = self.data_loader.generate_synthetic_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise_level,
                test_size=test_size,
            )
            return self.run_regression_experiment("synthetic", test_size=test_size)

    def create_results_summary(self) -> pd.DataFrame:
        """
        Create a summary of all results.

        Returns:
            DataFrame with results summary
        """
        summary_data = []

        for dataset_name, results in self.results.items():
            # Individual models
            for model_name, metrics in results["individual_model_results"].items():
                summary_data.append({
                    "dataset": dataset_name,
                    "method": "individual",
                    "model": model_name,
                    "accuracy": metrics.get("accuracy", np.nan),
                    "f1_score": metrics.get("f1_score", np.nan),
                    "rmse": metrics.get("rmse", np.nan),
                    "r2": metrics.get("r2", np.nan),
                    "training_time": metrics.get("training_time", np.nan),
                })

            # BMA methods
            for method_name, metrics in results["bma_results"].items():
                summary_data.append({
                    "dataset": dataset_name,
                    "method": "bma",
                    "model": method_name,
                    "accuracy": metrics.get("accuracy", np.nan),
                    "f1_score": metrics.get("f1_score", np.nan),
                    "rmse": metrics.get("rmse", np.nan),
                    "r2": metrics.get("r2", np.nan),
                    "training_time": metrics.get("training_time", np.nan),
                })

        return pd.DataFrame(summary_data)

    def save_results(self, filepath: str) -> None:
        """
        Save results to file.

        Args:
            filepath: Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
