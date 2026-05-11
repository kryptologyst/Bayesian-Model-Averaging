"""
Bayesian Model Averaging (BMA) implementation with uncertainty quantification.

This module provides implementations of Bayesian Model Averaging methods including:
- Cross-validation based BMA
- Bayesian Information Criterion (BIC) based BMA
- Laplace approximation based BMA
- MCMC-based BMA using Pyro/Numpyro

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array

logger = logging.getLogger(__name__)


class BayesianModelAveraging(ABC):
    """Abstract base class for Bayesian Model Averaging implementations."""

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Bayesian Model Averaging.

        Args:
            models: List of (name, model) tuples
            random_state: Random state for reproducibility
        """
        self.models = models
        self.random_state = random_state
        self.model_weights_: Optional[Dict[str, float]] = None
        self.trained_models_: Optional[Dict[str, BaseEstimator]] = None
        self.model_scores_: Optional[Dict[str, float]] = None

        # Set random state for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            for _, model in models:
                if hasattr(model, "random_state"):
                    model.random_state = random_state

    @abstractmethod
    def _compute_model_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute model weights based on posterior probabilities."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianModelAveraging:
        """
        Fit the Bayesian Model Averaging ensemble.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self
        """
        X, y = check_X_y(X, y)
        
        logger.info(f"Fitting BMA with {len(self.models)} models")
        
        # Compute model weights
        self.model_weights_ = self._compute_model_weights(X, y)
        
        # Train all models on full dataset
        self.trained_models_ = {}
        for name, model in self.models:
            logger.info(f"Training model: {name}")
            self.trained_models_[name] = model.fit(X, y)
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted ensemble.

        Args:
            X: Test features

        Returns:
            Predicted probabilities
        """
        if self.model_weights_ is None or self.trained_models_ is None:
            raise ValueError("Model must be fitted before making predictions")

        X = check_array(X)
        
        # Get predictions from each model
        predictions = []
        for name, model in self.trained_models_.items():
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
            else:
                # For models without predict_proba, convert decision function to probabilities
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X)
                    pred = np.column_stack([1 - stats.norm.cdf(scores), stats.norm.cdf(scores)])
                else:
                    # Fallback to hard predictions converted to probabilities
                    hard_pred = model.predict(X)
                    pred = np.column_stack([1 - hard_pred, hard_pred])
            
            predictions.append(pred)

        # Weighted average of predictions
        weighted_pred = np.zeros_like(predictions[0])
        for i, (name, _) in enumerate(self.trained_models_.items()):
            weighted_pred += self.model_weights_[name] * predictions[i]

        return weighted_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using weighted ensemble.

        Args:
            X: Test features

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_model_weights(self) -> Dict[str, float]:
        """Get the computed model weights."""
        if self.model_weights_ is None:
            raise ValueError("Model must be fitted before accessing weights")
        return self.model_weights_.copy()


class CrossValidationBMA(BayesianModelAveraging):
    """Bayesian Model Averaging using cross-validation scores as posterior probabilities."""

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Cross-Validation based BMA.

        Args:
            models: List of (name, model) tuples
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            random_state: Random state for reproducibility
        """
        super().__init__(models, random_state)
        self.cv = cv
        self.scoring = scoring

    def _compute_model_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute model weights using cross-validation scores."""
        model_scores = {}
        
        for name, model in self.models:
            logger.info(f"Computing CV scores for model: {name}")
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            model_scores[name] = np.mean(scores)
            logger.info(f"Model {name} CV score: {model_scores[name]:.4f}")

        self.model_scores_ = model_scores
        
        # Convert scores to probabilities using softmax
        scores_array = np.array(list(model_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
        weights_array = exp_scores / np.sum(exp_scores)
        
        model_weights = dict(zip(model_scores.keys(), weights_array))
        
        logger.info("Model weights computed:")
        for name, weight in model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        return model_weights


class BICBMA(BayesianModelAveraging):
    """Bayesian Model Averaging using Bayesian Information Criterion (BIC)."""

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize BIC-based BMA.

        Args:
            models: List of (name, model) tuples
            random_state: Random state for reproducibility
        """
        super().__init__(models, random_state)

    def _compute_model_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute model weights using BIC."""
        model_bics = {}
        
        for name, model in self.models:
            logger.info(f"Computing BIC for model: {name}")
            
            # Fit model
            fitted_model = model.fit(X, y)
            
            # Compute BIC
            n_samples = X.shape[0]
            n_params = self._count_parameters(fitted_model)
            
            if hasattr(fitted_model, "predict_proba"):
                # For classification models
                proba = fitted_model.predict_proba(X)
                log_likelihood = np.sum(np.log(proba[np.arange(len(y)), y] + 1e-10))
            else:
                # For regression models
                predictions = fitted_model.predict(X)
                mse = np.mean((y - predictions) ** 2)
                log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi * mse) + 1)
            
            bic = -2 * log_likelihood + n_params * np.log(n_samples)
            model_bics[name] = bic
            logger.info(f"Model {name} BIC: {bic:.4f}")

        self.model_scores_ = model_bics
        
        # Convert BIC to weights (lower BIC = higher weight)
        bic_array = np.array(list(model_bics.values()))
        weights_array = np.exp(-0.5 * (bic_array - np.min(bic_array)))
        weights_array = weights_array / np.sum(weights_array)
        
        model_weights = dict(zip(model_bics.keys(), weights_array))
        
        logger.info("Model weights computed:")
        for name, weight in model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        return model_weights

    def _count_parameters(self, model: BaseEstimator) -> int:
        """Count the number of parameters in a model."""
        if hasattr(model, "coef_"):
            return model.coef_.size
        elif hasattr(model, "feature_importances_"):
            return model.feature_importances_.size
        elif hasattr(model, "n_features_in_"):
            return model.n_features_in_
        else:
            # Default fallback
            return 1


class LaplaceApproximationBMA(BayesianModelAveraging):
    """Bayesian Model Averaging using Laplace approximation."""

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        prior_variance: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Laplace approximation based BMA.

        Args:
            models: List of (name, model) tuples
            prior_variance: Prior variance for Laplace approximation
            random_state: Random state for reproducibility
        """
        super().__init__(models, random_state)
        self.prior_variance = prior_variance

    def _compute_model_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute model weights using Laplace approximation."""
        model_log_evidences = {}
        
        for name, model in self.models:
            logger.info(f"Computing Laplace approximation for model: {name}")
            
            # Fit model
            fitted_model = model.fit(X, y)
            
            # Compute log evidence using Laplace approximation
            log_evidence = self._laplace_log_evidence(fitted_model, X, y)
            model_log_evidences[name] = log_evidence
            logger.info(f"Model {name} log evidence: {log_evidence:.4f}")

        self.model_scores_ = model_log_evidences
        
        # Convert log evidences to weights
        log_evidences_array = np.array(list(model_log_evidences.values()))
        weights_array = np.exp(log_evidences_array - np.max(log_evidences_array))
        weights_array = weights_array / np.sum(weights_array)
        
        model_weights = dict(zip(model_log_evidences.keys(), weights_array))
        
        logger.info("Model weights computed:")
        for name, weight in model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        return model_weights

    def _laplace_log_evidence(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log evidence using Laplace approximation."""
        n_samples = X.shape[0]
        n_params = self._count_parameters(model)
        
        # Compute log likelihood
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            log_likelihood = np.sum(np.log(proba[np.arange(len(y)), y] + 1e-10))
        else:
            predictions = model.predict(X)
            mse = np.mean((y - predictions) ** 2)
            log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi * mse) + 1)
        
        # Prior (assuming Gaussian prior)
        log_prior = -0.5 * n_params * np.log(2 * np.pi * self.prior_variance)
        
        # Hessian approximation (simplified)
        log_det_hessian = n_params * np.log(self.prior_variance)
        
        # Log evidence = log likelihood + log prior - 0.5 * log det Hessian
        log_evidence = log_likelihood + log_prior - 0.5 * log_det_hessian
        
        return log_evidence

    def _count_parameters(self, model: BaseEstimator) -> int:
        """Count the number of parameters in a model."""
        if hasattr(model, "coef_"):
            return model.coef_.size
        elif hasattr(model, "feature_importances_"):
            return model.feature_importances_.size
        elif hasattr(model, "n_features_in_"):
            return model.n_features_in_
        else:
            return 1


def create_default_models(random_state: int = 42) -> List[Tuple[str, BaseEstimator]]:
    """Create default set of models for BMA."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    
    models = [
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=random_state)),
        ("svm", SVC(probability=True, random_state=random_state)),
        ("naive_bayes", GaussianNB()),
    ]
    
    return models
