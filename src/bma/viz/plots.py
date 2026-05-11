"""
Visualization utilities for Bayesian Model Averaging.

This module provides comprehensive visualization tools for BMA results including
model comparison plots, uncertainty visualization, and calibration plots.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BMAVisualizer:
    """Visualization utilities for Bayesian Model Averaging results."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Initialize BMA Visualizer.

        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = px.colors.qualitative.Set3

    def plot_model_comparison(
        self,
        results: Dict[str, Any],
        metric: str = "accuracy",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot model comparison results.

        Args:
            results: Results dictionary from BMA pipeline
            metric: Metric to plot
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Prepare data
        individual_results = results["individual_model_results"]
        bma_results = results["bma_results"]

        # Individual models
        individual_models = list(individual_results.keys())
        individual_scores = [individual_results[model][metric] for model in individual_models]

        # BMA methods
        bma_methods = list(bma_results.keys())
        bma_scores = [bma_results[method][metric] for method in bma_methods]

        # Plot individual models
        bars1 = ax1.bar(individual_models, individual_scores, color=self.colors[:len(individual_models)])
        ax1.set_title(f"Individual Models - {metric.title()}")
        ax1.set_ylabel(metric.title())
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars1, individual_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # Plot BMA methods
        bars2 = ax2.bar(bma_methods, bma_scores, color=self.colors[len(individual_models):len(individual_models)+len(bma_methods)])
        ax2.set_title(f"BMA Methods - {metric.title()}")
        ax2.set_ylabel(metric.title())
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars2, bma_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_model_weights(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot model weights for BMA methods.

        Args:
            results: Results dictionary from BMA pipeline
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        bma_results = results["bma_results"]
        n_methods = len(bma_results)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        for i, (method, metrics) in enumerate(bma_results.items()):
            weights = metrics["model_weights"]
            models = list(weights.keys())
            weight_values = list(weights.values())

            # Create pie chart
            wedges, texts, autotexts = axes[i].pie(
                weight_values, 
                labels=models, 
                autopct='%1.2f',
                colors=self.colors[:len(models)]
            )
            axes[i].set_title(f"Model Weights - {method.upper()}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        method_name: str = "BMA",
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot calibration curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            method_name: Name of the method
            n_bins: Number of bins
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Use positive class probability

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calibration curve
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(in_bin.sum())

        bin_centers = np.array(bin_centers)
        bin_accuracies = np.array(bin_accuracies)
        bin_counts = np.array(bin_counts)

        # Plot calibration curve
        ax1.plot(bin_centers, bin_accuracies, 'o-', label=method_name, linewidth=2, markersize=8)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Calibration Curve - {method_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot reliability diagram
        ax2.bar(bin_centers, bin_accuracies - bin_centers, width=0.1, alpha=0.7, label=method_name)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Accuracy - Confidence')
        ax2.set_title(f'Reliability Diagram - {method_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_uncertainty_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None,
        method_name: str = "BMA",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot uncertainty analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_std: Predicted standard deviations
            method_name: Name of the method
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Prediction vs True values
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Predictions vs True Values - {method_name}')
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals plot
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residuals Plot - {method_name}')
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Residuals Distribution - {method_name}')
        axes[1, 0].grid(True, alpha=0.3)

        # Uncertainty plot (if available)
        if y_pred_std is not None:
            axes[1, 1].scatter(y_pred, y_pred_std, alpha=0.6, s=50)
            axes[1, 1].set_xlabel('Predicted Values')
            axes[1, 1].set_ylabel('Predicted Uncertainty')
            axes[1, 1].set_title(f'Prediction Uncertainty - {method_name}')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Uncertainty not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f'Prediction Uncertainty - {method_name}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_interactive_comparison(
        self,
        results: Dict[str, Any],
        metric: str = "accuracy",
    ) -> go.Figure:
        """
        Create interactive comparison plot.

        Args:
            results: Results dictionary from BMA pipeline
            metric: Metric to plot

        Returns:
            Plotly figure
        """
        individual_results = results["individual_model_results"]
        bma_results = results["bma_results"]

        # Prepare data
        categories = []
        values = []
        colors = []
        hover_text = []

        # Individual models
        for model_name, metrics in individual_results.items():
            categories.append(f"Individual: {model_name}")
            values.append(metrics[metric])
            colors.append('lightblue')
            hover_text.append(f"Model: {model_name}<br>Method: Individual<br>{metric.title()}: {metrics[metric]:.4f}")

        # BMA methods
        for method_name, metrics in bma_results.items():
            categories.append(f"BMA: {method_name}")
            values.append(metrics[metric])
            colors.append('lightcoral')
            hover_text.append(f"Method: {method_name}<br>Type: BMA<br>{metric.title()}: {metrics[metric]:.4f}")

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition='auto',
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
            )
        ])

        fig.update_layout(
            title=f"Model Comparison - {metric.title()}",
            xaxis_title="Models/Methods",
            yaxis_title=metric.title(),
            showlegend=False,
            height=500,
        )

        return fig

    def plot_model_weights_interactive(
        self,
        results: Dict[str, Any],
    ) -> go.Figure:
        """
        Create interactive model weights plot.

        Args:
            results: Results dictionary from BMA pipeline

        Returns:
            Plotly figure
        """
        bma_results = results["bma_results"]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(bma_results),
            specs=[[{"type": "pie"}] * len(bma_results)],
            subplot_titles=[f"{method.upper()} Weights" for method in bma_results.keys()]
        )

        for i, (method, metrics) in enumerate(bma_results.items()):
            weights = metrics["model_weights"]
            models = list(weights.keys())
            weight_values = list(weights.values())

            fig.add_trace(
                go.Pie(
                    labels=models,
                    values=weight_values,
                    name=method,
                    textinfo='label+percent',
                    textposition='auto',
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            title="Model Weights Comparison",
            height=400,
            showlegend=False,
        )

        return fig

    def create_results_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive results dashboard.

        Args:
            results: Results dictionary from BMA pipeline
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 15))

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Model comparison (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        individual_results = results["individual_model_results"]
        bma_results = results["bma_results"]
        
        all_models = list(individual_results.keys()) + list(bma_results.keys())
        all_scores = [individual_results[model]["accuracy"] for model in individual_results.keys()] + \
                    [bma_results[method]["accuracy"] for method in bma_results.keys()]
        
        colors = ['lightblue'] * len(individual_results) + ['lightcoral'] * len(bma_results)
        bars = ax1.bar(all_models, all_scores, color=colors)
        ax1.set_title("Model Comparison - Accuracy", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Accuracy")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, all_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # 2. Model weights (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if bma_results:
            method = list(bma_results.keys())[0]  # Use first BMA method
            weights = bma_results[method]["model_weights"]
            models = list(weights.keys())
            weight_values = list(weights.values())
            
            wedges, texts, autotexts = ax2.pie(weight_values, labels=models, autopct='%1.1f%%')
            ax2.set_title(f"Model Weights - {method.upper()}", fontsize=14, fontweight='bold')

        # 3. Training time comparison (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        training_times = [individual_results[model]["training_time"] for model in individual_results.keys()] + \
                        [bma_results[method]["training_time"] for method in bma_results.keys()]
        
        bars = ax3.bar(all_models, training_times, color=colors)
        ax3.set_title("Training Time Comparison", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Time (seconds)")
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars, training_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')

        # 4. Metrics comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        individual_metrics = [individual_results[list(individual_results.keys())[0]][m] for m in metrics_to_plot]
        bma_metrics = [bma_results[list(bma_results.keys())[0]][m] for m in metrics_to_plot]
        
        ax4.bar(x - width/2, individual_metrics, width, label='Individual', color='lightblue')
        ax4.bar(x + width/2, bma_metrics, width, label='BMA', color='lightcoral')
        ax4.set_title("Metrics Comparison", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Score")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_to_plot)
        ax4.legend()

        # 5. Summary statistics (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary text
        best_individual = max(individual_results.items(), key=lambda x: x[1]["accuracy"])
        best_bma = max(bma_results.items(), key=lambda x: x[1]["accuracy"])
        
        summary_text = f"""
        Summary:
        • Best Individual Model: {best_individual[0]} (Accuracy: {best_individual[1]['accuracy']:.4f})
        • Best BMA Method: {best_bma[0]} (Accuracy: {best_bma[1]['accuracy']:.4f})
        • Dataset: {results['metadata']['n_samples']} samples, {results['metadata']['n_features']} features
        • Improvement: {best_bma[1]['accuracy'] - best_individual[1]['accuracy']:.4f}
        """
        
        ax5.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")

        return fig
