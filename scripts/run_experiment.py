"""
Main script for Bayesian Model Averaging experiments.

This script provides a command-line interface for running BMA experiments
with various datasets and methods.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.bma.data.loader import DataLoader
from src.bma.models.bma import create_default_models
from src.bma.train.pipeline import BMAPipeline
from src.bma.viz.plots import BMAVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "assets/results",
        "assets/figures",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run_classification_experiment(
    dataset_name: str,
    output_dir: str = "assets/results",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run classification experiment."""
    logger.info(f"Starting classification experiment on {dataset_name}")
    
    # Default configuration
    if config is None:
        config = {
            "test_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "bma_methods": ["cv", "bic", "laplace"]
        }
    
    # Initialize pipeline
    pipeline = BMAPipeline(
        random_state=config["random_state"],
        verbose=True
    )
    
    # Run experiment
    results = pipeline.run_classification_experiment(
        dataset_name=dataset_name,
        bma_methods=config["bma_methods"],
        test_size=config["test_size"],
        cv_folds=config["cv_folds"]
    )
    
    # Save results
    output_file = os.path.join(output_dir, f"classification_{dataset_name}_results.json")
    pipeline.save_results(output_file)
    
    # Create visualizations
    visualizer = BMAVisualizer()
    
    # Model comparison plot
    fig1 = visualizer.plot_model_comparison(results, metric="accuracy")
    fig1.savefig(os.path.join(output_dir, f"classification_{dataset_name}_comparison.png"))
    
    # Model weights plot
    if results["bma_results"]:
        fig2 = visualizer.plot_model_weights(results)
        fig2.savefig(os.path.join(output_dir, f"classification_{dataset_name}_weights.png"))
    
    # Results dashboard
    fig3 = visualizer.create_results_dashboard(results)
    fig3.savefig(os.path.join(output_dir, f"classification_{dataset_name}_dashboard.png"))
    
    logger.info(f"Results saved to {output_dir}")
    return results


def run_regression_experiment(
    dataset_name: str,
    output_dir: str = "assets/results",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run regression experiment."""
    logger.info(f"Starting regression experiment on {dataset_name}")
    
    # Default configuration
    if config is None:
        config = {
            "test_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "bma_methods": ["cv", "bic", "laplace"]
        }
    
    # Initialize pipeline
    pipeline = BMAPipeline(
        random_state=config["random_state"],
        verbose=True
    )
    
    # Run experiment
    results = pipeline.run_regression_experiment(
        dataset_name=dataset_name,
        bma_methods=config["bma_methods"],
        test_size=config["test_size"],
        cv_folds=config["cv_folds"]
    )
    
    # Save results
    output_file = os.path.join(output_dir, f"regression_{dataset_name}_results.json")
    pipeline.save_results(output_file)
    
    # Create visualizations
    visualizer = BMAVisualizer()
    
    # Model comparison plot
    fig1 = visualizer.plot_model_comparison(results, metric="r2")
    fig1.savefig(os.path.join(output_dir, f"regression_{dataset_name}_comparison.png"))
    
    # Model weights plot
    if results["bma_results"]:
        fig2 = visualizer.plot_model_weights(results)
        fig2.savefig(os.path.join(output_dir, f"regression_{dataset_name}_weights.png"))
    
    # Results dashboard
    fig3 = visualizer.create_results_dashboard(results)
    fig3.savefig(os.path.join(output_dir, f"regression_{dataset_name}_dashboard.png"))
    
    logger.info(f"Results saved to {output_dir}")
    return results


def run_synthetic_experiment(
    task_type: str = "classification",
    output_dir: str = "assets/results",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run synthetic data experiment."""
    logger.info(f"Starting synthetic {task_type} experiment")
    
    # Default configuration
    if config is None:
        config = {
            "n_samples": 1000,
            "n_features": 20,
            "noise_level": 0.1,
            "test_size": 0.2,
            "random_state": 42,
            "bma_methods": ["cv", "bic", "laplace"]
        }
    
    # Initialize pipeline
    pipeline = BMAPipeline(
        random_state=config["random_state"],
        verbose=True
    )
    
    # Run experiment
    results = pipeline.run_synthetic_experiment(
        task_type=task_type,
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        noise_level=config["noise_level"],
        test_size=config["test_size"]
    )
    
    # Save results
    output_file = os.path.join(output_dir, f"synthetic_{task_type}_results.json")
    pipeline.save_results(output_file)
    
    # Create visualizations
    visualizer = BMAVisualizer()
    
    # Model comparison plot
    metric = "accuracy" if task_type == "classification" else "r2"
    fig1 = visualizer.plot_model_comparison(results, metric=metric)
    fig1.savefig(os.path.join(output_dir, f"synthetic_{task_type}_comparison.png"))
    
    # Model weights plot
    if results["bma_results"]:
        fig2 = visualizer.plot_model_weights(results)
        fig2.savefig(os.path.join(output_dir, f"synthetic_{task_type}_weights.png"))
    
    # Results dashboard
    fig3 = visualizer.create_results_dashboard(results)
    fig3.savefig(os.path.join(output_dir, f"synthetic_{task_type}_dashboard.png"))
    
    logger.info(f"Results saved to {output_dir}")
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Bayesian Model Averaging Experiments")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression", "synthetic"],
        default="classification",
        help="Type of task to run"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        help="Dataset name (ignored for synthetic tasks)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Setup directories
    setup_directories()
    
    # Load configuration
    config = None
    if args.config:
        config = OmegaConf.load(args.config)
        config = OmegaConf.to_container(config, resolve=True)
    
    # Run experiment
    try:
        if args.task == "classification":
            results = run_classification_experiment(
                dataset_name=args.dataset,
                output_dir=args.output_dir,
                config=config
            )
        elif args.task == "regression":
            results = run_regression_experiment(
                dataset_name=args.dataset,
                output_dir=args.output_dir,
                config=config
            )
        elif args.task == "synthetic":
            results = run_synthetic_experiment(
                task_type="classification",  # Default to classification for synthetic
                output_dir=args.output_dir,
                config=config
            )
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        if args.task == "classification":
            best_individual = max(results["individual_model_results"].items(), 
                               key=lambda x: x[1]["accuracy"])
            best_bma = max(results["bma_results"].items(), 
                          key=lambda x: x[1]["accuracy"])
            
            print(f"Best Individual Model: {best_individual[0]} (Accuracy: {best_individual[1]['accuracy']:.4f})")
            print(f"Best BMA Method: {best_bma[0]} (Accuracy: {best_bma[1]['accuracy']:.4f})")
            improvement = best_bma[1]['accuracy'] - best_individual[1]['accuracy']
            print(f"BMA Improvement: {improvement:+.4f}")
            
        else:  # regression
            best_individual = max(results["individual_model_results"].items(), 
                               key=lambda x: x[1]["r2"])
            best_bma = max(results["bma_results"].items(), 
                          key=lambda x: x[1]["r2"])
            
            print(f"Best Individual Model: {best_individual[0]} (R²: {best_individual[1]['r2']:.4f})")
            print(f"Best BMA Method: {best_bma[0]} (R²: {best_bma[1]['r2']:.4f})")
            improvement = best_bma[1]['r2'] - best_individual[1]['r2']
            print(f"BMA Improvement: {improvement:+.4f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
