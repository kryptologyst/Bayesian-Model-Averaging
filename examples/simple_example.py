#!/usr/bin/env python3
"""
Simple example script demonstrating Bayesian Model Averaging.

This script shows how to use the BMA package for a basic classification task.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.bma.models.bma import CrossValidationBMA, create_default_models
from src.bma.train.pipeline import BMAPipeline


def main():
    """Run a simple BMA example."""
    print("🧠 Bayesian Model Averaging Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create models
    print("\nCreating models...")
    models = create_default_models(random_state=42)
    
    for name, model in models:
        print(f"- {name}: {type(model).__name__}")
    
    # Train individual models
    print("\nTraining individual models...")
    individual_results = {}
    
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        individual_results[name] = accuracy
        print(f"{name}: Accuracy = {accuracy:.4f}")
    
    # Train BMA model
    print("\nTraining BMA model...")
    bma_model = CrossValidationBMA(models, cv=5, random_state=42)
    bma_model.fit(X_train, y_train)
    
    # Make BMA predictions
    y_pred_bma = bma_model.predict(X_test)
    accuracy_bma = accuracy_score(y_test, y_pred_bma)
    
    print(f"BMA: Accuracy = {accuracy_bma:.4f}")
    
    # Show model weights
    weights = bma_model.get_model_weights()
    print("\nModel weights:")
    for model_name, weight in weights.items():
        print(f"  {model_name}: {weight:.4f}")
    
    # Find best individual model
    best_individual = max(individual_results.items(), key=lambda x: x[1])
    
    print(f"\nBest individual model: {best_individual[0]} (Accuracy: {best_individual[1]:.4f})")
    print(f"BMA accuracy: {accuracy_bma:.4f}")
    print(f"Improvement: {accuracy_bma - best_individual[1]:+.4f}")
    
    # Run comprehensive experiment
    print("\n" + "=" * 50)
    print("Running comprehensive experiment...")
    
    pipeline = BMAPipeline(random_state=42, verbose=False)
    results = pipeline.run_classification_experiment(
        dataset_name="breast_cancer",
        bma_methods=["cv", "bic", "laplace"],
        test_size=0.2,
        cv_folds=5
    )
    
    # Print summary
    print("\nComprehensive Results Summary:")
    print("-" * 30)
    
    # Individual models
    print("Individual Models:")
    for name, metrics in results["individual_model_results"].items():
        print(f"  {name}: Accuracy = {metrics['accuracy']:.4f}")
    
    # BMA methods
    print("\nBMA Methods:")
    for method, metrics in results["bma_results"].items():
        print(f"  {method}: Accuracy = {metrics['accuracy']:.4f}")
    
    # Best results
    best_individual = max(results["individual_model_results"].items(), 
                         key=lambda x: x[1]["accuracy"])
    best_bma = max(results["bma_results"].items(), 
                   key=lambda x: x[1]["accuracy"])
    
    print(f"\nBest Individual: {best_individual[0]} ({best_individual[1]['accuracy']:.4f})")
    print(f"Best BMA: {best_bma[0]} ({best_bma[1]['accuracy']:.4f})")
    print(f"Improvement: {best_bma[1]['accuracy'] - best_individual[1]['accuracy']:+.4f}")
    
    print("\n✅ Example completed successfully!")
    print("\n⚠️  Remember: This is for research and education purposes only.")
    print("   Not intended for production decisions or control systems.")


if __name__ == "__main__":
    main()
