"""
Streamlit demo application for Bayesian Model Averaging.

This module provides an interactive web interface for exploring BMA methods,
comparing models, and visualizing results.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.bma.data.loader import DataLoader
from src.bma.models.bma import create_default_models
from src.bma.train.pipeline import BMAPipeline
from src.bma.viz.plots import BMAVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bayesian Model Averaging Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #ff7f0e;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Bayesian Model Averaging Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research & Education Disclaimer</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>Not intended for production decisions or control systems</li>
        <li>Results should not be used for critical decision-making</li>
        <li>Models may not generalize to real-world scenarios</li>
        <li>Always validate results with domain experts</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    st.sidebar.subheader("Dataset")
    dataset_type = st.sidebar.selectbox(
        "Task Type",
        ["Classification", "Regression"],
        help="Choose between classification or regression task"
    )
    
    if dataset_type == "Classification":
        dataset_options = ["breast_cancer", "wine", "iris", "synthetic"]
        dataset_name = st.sidebar.selectbox("Dataset", dataset_options)
    else:
        dataset_options = ["diabetes", "boston", "synthetic"]
        dataset_name = st.sidebar.selectbox("Dataset", dataset_options)
    
    # Model configuration
    st.sidebar.subheader("Models")
    use_default_models = st.sidebar.checkbox("Use Default Models", value=True)
    
    if not use_default_models:
        st.sidebar.write("Custom model configuration not implemented in this demo.")
        st.sidebar.write("Please use default models.")
    
    # BMA methods
    st.sidebar.subheader("BMA Methods")
    bma_methods = st.sidebar.multiselect(
        "Select BMA Methods",
        ["Cross-Validation", "BIC", "Laplace Approximation"],
        default=["Cross-Validation", "BIC"],
        help="Choose which BMA methods to evaluate"
    )
    
    # Experiment parameters
    st.sidebar.subheader("Experiment Parameters")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    cv_folds = st.sidebar.slider("CV Folds", 3, 10, 5)
    random_state = st.sidebar.number_input("Random State", 0, 1000, 42)
    
    # Run experiment button
    if st.sidebar.button("🚀 Run Experiment", type="primary"):
        run_experiment(dataset_type.lower(), dataset_name, bma_methods, test_size, cv_folds, random_state)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Author:** kryptologyst  
    **GitHub:** [kryptologyst](https://github.com/kryptologyst)
    
    This demo showcases Bayesian Model Averaging (BMA) methods for combining multiple machine learning models.
    """)

def run_experiment(
    dataset_type: str,
    dataset_name: str,
    bma_methods: List[str],
    test_size: float,
    cv_folds: int,
    random_state: int
):
    """Run the BMA experiment and display results."""
    
    # Convert method names to internal format
    method_mapping = {
        "Cross-Validation": "cv",
        "BIC": "bic", 
        "Laplace Approximation": "laplace"
    }
    internal_methods = [method_mapping[method] for method in bma_methods]
    
    # Initialize pipeline
    pipeline = BMAPipeline(random_state=random_state, verbose=False)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Run experiment
        status_text.text("Loading data...")
        progress_bar.progress(20)
        
        if dataset_type == "classification":
            results = pipeline.run_classification_experiment(
                dataset_name=dataset_name,
                bma_methods=internal_methods,
                test_size=test_size,
                cv_folds=cv_folds
            )
        else:
            results = pipeline.run_regression_experiment(
                dataset_name=dataset_name,
                bma_methods=internal_methods,
                test_size=test_size,
                cv_folds=cv_folds
            )
        
        progress_bar.progress(100)
        status_text.text("Experiment completed!")
        
        # Display results
        display_results(results, dataset_type)
        
    except Exception as e:
        st.error(f"Error running experiment: {str(e)}")
        logger.error(f"Experiment failed: {e}")

def display_results(results: Dict[str, Any], dataset_type: str):
    """Display experiment results."""
    
    st.markdown('<h2 class="section-header">📊 Results</h2>', unsafe_allow_html=True)
    
    # Dataset info
    metadata = results["metadata"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset", metadata.get("dataset", "Unknown"))
    with col2:
        st.metric("Samples", metadata["n_samples"])
    with col3:
        st.metric("Features", metadata["n_features"])
    with col4:
        if dataset_type == "classification":
            st.metric("Classes", metadata["n_classes"])
        else:
            st.metric("Task", "Regression")
    
    # Results summary
    st.markdown('<h3 class="section-header">📈 Performance Summary</h3>', unsafe_allow_html=True)
    
    # Create results DataFrame
    summary_data = []
    
    # Individual models
    for model_name, metrics in results["individual_model_results"].items():
        summary_data.append({
            "Method": "Individual",
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", np.nan),
            "F1 Score": metrics.get("f1_score", np.nan),
            "RMSE": metrics.get("rmse", np.nan),
            "R²": metrics.get("r2", np.nan),
            "Training Time": f"{metrics.get('training_time', 0):.2f}s"
        })
    
    # BMA methods
    for method_name, metrics in results["bma_results"].items():
        summary_data.append({
            "Method": "BMA",
            "Model": method_name.upper(),
            "Accuracy": metrics.get("accuracy", np.nan),
            "F1 Score": metrics.get("f1_score", np.nan),
            "RMSE": metrics.get("rmse", np.nan),
            "R²": metrics.get("r2", np.nan),
            "Training Time": f"{metrics.get('training_time', 0):.2f}s"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Performance comparison
    st.markdown('<h3 class="section-header">📊 Performance Comparison</h3>', unsafe_allow_html=True)
    
    # Select metric for comparison
    if dataset_type == "classification":
        metric_options = ["accuracy", "f1_score", "precision", "recall"]
        default_metric = "accuracy"
    else:
        metric_options = ["r2", "rmse", "mae"]
        default_metric = "r2"
    
    selected_metric = st.selectbox("Select metric for comparison:", metric_options, 
                                 index=metric_options.index(default_metric))
    
    # Create comparison plot
    individual_results = results["individual_model_results"]
    bma_results = results["bma_results"]
    
    # Prepare data for plotting
    categories = []
    values = []
    colors = []
    
    # Individual models
    for model_name, metrics in individual_results.items():
        categories.append(f"Individual: {model_name}")
        values.append(metrics[selected_metric])
        colors.append('lightblue')
    
    # BMA methods
    for method_name, metrics in bma_results.items():
        categories.append(f"BMA: {method_name.upper()}")
        values.append(metrics[selected_metric])
        colors.append('lightcoral')
    
    # Create plot
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Model Comparison - {selected_metric.replace('_', ' ').title()}",
        xaxis_title="Models/Methods",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model weights
    if bma_results:
        st.markdown('<h3 class="section-header">⚖️ Model Weights</h3>', unsafe_allow_html=True)
        
        # Create subplots for model weights
        methods = list(bma_results.keys())
        fig = make_subplots(
            rows=1, cols=len(methods),
            specs=[[{"type": "pie"}] * len(methods)],
            subplot_titles=[f"{method.upper()} Weights" for method in methods]
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display weights table
        weights_data = []
        for method, metrics in bma_results.items():
            for model, weight in metrics["model_weights"].items():
                weights_data.append({
                    "BMA Method": method.upper(),
                    "Model": model,
                    "Weight": f"{weight:.4f}"
                })
        
        weights_df = pd.DataFrame(weights_data)
        st.dataframe(weights_df, use_container_width=True)
    
    # Best model summary
    st.markdown('<h3 class="section-header">🏆 Best Model Summary</h3>', unsafe_allow_html=True)
    
    # Find best individual and BMA models
    best_individual = max(individual_results.items(), key=lambda x: x[1][selected_metric])
    best_bma = max(bma_results.items(), key=lambda x: x[1][selected_metric])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h4>Best Individual Model</h4>
        <p><strong>{best_individual[0]}</strong></p>
        <p>{selected_metric.replace('_', ' ').title()}: {best_individual[1][selected_metric]:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h4>Best BMA Method</h4>
        <p><strong>{best_bma[0].upper()}</strong></p>
        <p>{selected_metric.replace('_', ' ').title()}: {best_bma[1][selected_metric]:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Improvement
    improvement = best_bma[1][selected_metric] - best_individual[1][selected_metric]
    improvement_pct = (improvement / best_individual[1][selected_metric]) * 100
    
    st.markdown(f"""
    <div class="metric-card">
    <h4>BMA Improvement</h4>
    <p>Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)</p>
    <p>Best BMA method outperforms best individual model by {improvement_pct:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
