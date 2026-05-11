"""
Data loading and preprocessing utilities for Bayesian Model Averaging.

This module provides utilities for loading datasets, generating synthetic data,
and preprocessing data for BMA experiments.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_iris,
    make_classification,
    make_regression,
    load_diabetes,
    load_boston,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preprocessing utilities."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize DataLoader.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None

    def load_classification_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load a classification dataset.

        Args:
            dataset_name: Name of the dataset to load
            test_size: Fraction of data to use for testing
            scale: Whether to standardize features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info(f"Loading classification dataset: {dataset_name}")

        if dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            metadata = {
                "feature_names": data.feature_names,
                "target_names": data.target_names,
                "n_classes": len(np.unique(y)),
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        elif dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            metadata = {
                "feature_names": data.feature_names,
                "target_names": data.target_names,
                "n_classes": len(np.unique(y)),
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        elif dataset_name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            metadata = {
                "feature_names": data.feature_names,
                "target_names": data.target_names,
                "n_classes": len(np.unique(y)),
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        elif dataset_name == "synthetic":
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=self.random_state,
            )
            metadata = {
                "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                "target_names": ["class_0", "class_1"],
                "n_classes": len(np.unique(y)),
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        else:
            raise ValueError(f"Unknown classification dataset: {dataset_name}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test, metadata

    def load_regression_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load a regression dataset.

        Args:
            dataset_name: Name of the dataset to load
            test_size: Fraction of data to use for testing
            scale: Whether to standardize features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info(f"Loading regression dataset: {dataset_name}")

        if dataset_name == "diabetes":
            data = load_diabetes()
            X, y = data.data, data.target
            metadata = {
                "feature_names": data.feature_names,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        elif dataset_name == "boston":
            try:
                data = load_boston()
                X, y = data.data, data.target
                metadata = {
                    "feature_names": data.feature_names,
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                }
            except ImportError:
                warnings.warn("Boston dataset is deprecated, using synthetic data instead")
                X, y = make_regression(
                    n_samples=506,
                    n_features=13,
                    noise=10.0,
                    random_state=self.random_state,
                )
                metadata = {
                    "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                }
        elif dataset_name == "synthetic":
            X, y = make_regression(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                noise=0.1,
                random_state=self.random_state,
            )
            metadata = {
                "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        else:
            raise ValueError(f"Unknown regression dataset: {dataset_name}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test, metadata

    def generate_synthetic_classification(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 2,
        n_informative: Optional[int] = None,
        noise: float = 0.1,
        test_size: float = 0.2,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic classification data.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            n_informative: Number of informative features
            noise: Amount of noise
            test_size: Fraction of data to use for testing
            scale: Whether to standardize features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        if n_informative is None:
            n_informative = min(n_features, 15)

        logger.info(f"Generating synthetic classification data: {n_samples} samples, {n_features} features")

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=n_classes,
            class_sep=1.0,
            flip_y=noise,
            random_state=self.random_state,
        )

        metadata = {
            "feature_names": [f"feature_{i}" for i in range(n_features)],
            "target_names": [f"class_{i}" for i in range(n_classes)],
            "n_classes": n_classes,
            "n_features": n_features,
            "n_samples": n_samples,
            "n_informative": n_informative,
            "noise": noise,
        }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, metadata

    def generate_synthetic_regression(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: Optional[int] = None,
        noise: float = 0.1,
        test_size: float = 0.2,
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic regression data.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_informative: Number of informative features
            noise: Amount of noise
            test_size: Fraction of data to use for testing
            scale: Whether to standardize features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        if n_informative is None:
            n_informative = min(n_features, 15)

        logger.info(f"Generating synthetic regression data: {n_samples} samples, {n_features} features")

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=self.random_state,
        )

        metadata = {
            "feature_names": [f"feature_{i}" for i in range(n_features)],
            "n_features": n_features,
            "n_samples": n_samples,
            "n_informative": n_informative,
            "noise": noise,
        }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, metadata

    def load_csv_data(
        self,
        file_path: str,
        target_column: str,
        test_size: float = 0.2,
        scale: bool = True,
        drop_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            test_size: Fraction of data to use for testing
            scale: Whether to standardize features
            drop_columns: Columns to drop

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info(f"Loading CSV data from: {file_path}")

        df = pd.read_csv(file_path)

        if drop_columns:
            df = df.drop(columns=drop_columns)

        # Separate features and target
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values

        # Handle categorical variables
        categorical_mask = df.drop(columns=[target_column]).select_dtypes(include=['object']).columns
        if len(categorical_mask) > 0:
            logger.info(f"Found categorical columns: {list(categorical_mask)}")
            # Simple label encoding for categorical variables
            for col in categorical_mask:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            X = df.drop(columns=[target_column]).values

        # Determine if classification or regression
        unique_targets = len(np.unique(y))
        is_classification = unique_targets < 20 and np.all(y == y.astype(int))

        metadata = {
            "feature_names": list(df.drop(columns=[target_column]).columns),
            "target_name": target_column,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "is_classification": is_classification,
            "n_classes": unique_targets if is_classification else None,
        }

        # Split data
        if is_classification:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"CSV data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test, metadata
