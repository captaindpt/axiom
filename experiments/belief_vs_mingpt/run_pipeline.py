#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from pathlib import Path
import torch
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_experiment_dir(base_dir="results"):
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"experiment_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def run_tests():
    """Run all unit tests for both models."""
    logger.info("Running unit tests...")
    import pytest
    test_result = pytest.main(["src/tests/", "-v"])
    if test_result != 0:
        logger.error("Unit tests failed! Aborting pipeline...")
        sys.exit(1)
    logger.info("All tests passed!")

def train_models(exp_dir, config):
    """Train both baseline minGPT and belief-enhanced GPT."""
    logger.info("Starting model training...")
    
    # Import training modules
    from training.trainer import train_model
    from models.baseline import BaselineGPT
    from models.belief_gpt import BeliefGPT
    
    # Train baseline
    logger.info("Training baseline minGPT...")
    baseline_model = BaselineGPT(config)
    baseline_metrics = train_model(baseline_model, config)
    
    # Train belief-enhanced
    logger.info("Training belief-enhanced GPT...")
    belief_model = BeliefGPT(config)
    belief_metrics = train_model(belief_model, config)
    
    # Save models and metrics
    torch.save(baseline_model.state_dict(), exp_dir / "baseline_model.pt")
    torch.save(belief_model.state_dict(), exp_dir / "belief_model.pt")
    
    with open(exp_dir / "training_metrics.json", "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "belief": belief_metrics
        }, f, indent=2)
    
    return baseline_model, belief_model

def evaluate_models(baseline_model, belief_model, exp_dir, config):
    """Run inference evaluation on both models."""
    logger.info("Starting model evaluation...")
    
    from evaluation.inference import evaluate_model
    
    # Evaluate both models
    baseline_results = evaluate_model(baseline_model, config)
    belief_results = evaluate_model(belief_model, config)
    
    # Save evaluation results
    with open(exp_dir / "evaluation_results.json", "w") as f:
        json.dump({
            "baseline": baseline_results,
            "belief": belief_results
        }, f, indent=2)

def visualize_results(exp_dir):
    """Generate visualizations of training and evaluation results."""
    logger.info("Generating visualizations...")
    
    from utils.visualization import (
        plot_training_comparison,
        plot_evaluation_comparison
    )
    
    # Load results
    with open(exp_dir / "training_metrics.json", "r") as f:
        training_metrics = json.load(f)
    
    with open(exp_dir / "evaluation_results.json", "r") as f:
        eval_results = json.load(f)
    
    # Generate plots
    plot_training_comparison(training_metrics, exp_dir / "training_comparison.png")
    plot_evaluation_comparison(eval_results, exp_dir / "evaluation_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Run belief-enhanced GPT comparison pipeline")
    parser.add_argument("--config", type=str, default="config.json",
                      help="Path to configuration file")
    parser.add_argument("--skip-tests", action="store_true",
                      help="Skip running unit tests")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Create experiment directory
    exp_dir = setup_experiment_dir()
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Save configuration
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run pipeline
    if not args.skip_tests:
        run_tests()
    
    baseline_model, belief_model = train_models(exp_dir, config)
    evaluate_models(baseline_model, belief_model, exp_dir, config)
    visualize_results(exp_dir)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved in: {exp_dir}")

if __name__ == "__main__":
    main() 