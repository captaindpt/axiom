import os
import torch
import torch.nn as nn
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from belief_transformer import BeliefEnhancedGPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import traceback
import sys
from tqdm import tqdm
import time

def get_device():
    """Get the best available device (Metal/CUDA/CPU)."""
    if torch.backends.mps.is_available():
        return 'mps'  # Apple Metal
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class SyntheticDataset:
    """Simple synthetic dataset for testing."""
    def __init__(self, block_size):
        self.block_size = block_size
        
    def __len__(self):
        return 1000  # Number of synthetic examples
        
    def __getitem__(self, idx):
        # Create random token sequences - ensure tokens are within vocab size
        x = torch.randint(0, 100, (self.block_size,), dtype=torch.long)  # Use long for indices
        y = x.clone()  # Use input as target for now
        return x, y

def create_experiment_dir():
    """Create a timestamped directory for experiment results."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = f'results/experiment_{timestamp}'
        print(f"Creating directory: {os.path.abspath(exp_dir)}")
        os.makedirs(exp_dir, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(exp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Successfully created and verified write permissions for: {exp_dir}")
        return exp_dir
    except Exception as e:
        print(f"Error creating experiment directory: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        traceback.print_exc()
        raise

def plot_metrics(metrics, exp_dir):
    """Plot and save training metrics."""
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_losses'], label='Training Loss')
    if 'val_losses' in metrics:
        plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot stability
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['stability_history'], label='Belief System Stability')
    plt.title('Belief System Stability Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Stability')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'stability_plot.png'))
    plt.close()

def run_experiment(seed: int, exp_dir: str):
    """Run a single experiment with controlled seed."""
    try:
        print(f"\nStarting experiment with seed {seed}")
        
        # Set seeds for reproducibility
        set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Get the best available device
        device = get_device()
        print(f"Using device: {device}")
        
        # Model configuration - Use nano size for faster testing
        config = GPT.get_default_config()
        config.model_type = 'gpt-nano'  # Smallest possible model
        config.vocab_size = 100  # Much smaller vocabulary for testing
        config.block_size = 32   # Smaller context size
        
        print("Initializing model...")
        model = BeliefEnhancedGPT(config)
        
        # Training configuration
        train_config = Trainer.get_default_config()
        train_config.max_iters = 1000
        train_config.batch_size = 8  # Smaller batch size
        train_config.learning_rate = 3e-4
        train_config.num_workers = 0  # Disable multiprocessing for now
        train_config.device = device  # Use the detected device
        
        # Performance debugging
        print("\nModel Configuration:")
        print(f"Number of layers: {config.n_layer}")
        print(f"Number of heads: {config.n_head}")
        print(f"Embedding dimension: {config.n_embd}")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Block size: {config.block_size}")
        print(f"Batch size: {train_config.batch_size}")
        
        # Metrics storage with timing details
        metrics = {
            'train_losses': [],
            'stability_history': [],
            'pattern_stats_history': [],
            'time_per_step': [],
            'forward_times': [],
            'backward_times': [],
            'belief_update_times': []
        }
        
        print("Creating dataset...")
        dataset = SyntheticDataset(config.block_size)
        
        print("Initializing trainer...")
        trainer = Trainer(
            config=train_config,
            model=model,
            train_dataset=dataset
        )
        
        # Progress bar setup
        pbar = tqdm(total=train_config.max_iters, desc=f"Training (Seed {seed})")
        last_time = time.time()
        
        # Add callback to collect metrics
        def collect_metrics(trainer):
            try:
                current_time = time.time()
                step_time = current_time - collect_metrics.last_time
                collect_metrics.last_time = current_time
                
                current_loss = trainer.loss.item()
                metrics['train_losses'].append(current_loss)
                metrics['time_per_step'].append(step_time)
                
                # Time the belief metrics collection
                belief_start = time.time()
                belief_metrics = model.get_belief_metrics()
                belief_time = time.time() - belief_start
                metrics['belief_update_times'].append(belief_time)
                
                metrics['stability_history'].append(belief_metrics['stability'])
                metrics['pattern_stats_history'].append(belief_metrics['pattern_stats'])
                
                # Update progress bar
                avg_time = np.mean(metrics['time_per_step'][-100:]) if metrics['time_per_step'] else 0
                avg_loss = np.mean(metrics['train_losses'][-100:]) if metrics['train_losses'] else 0
                avg_stability = np.mean(metrics['stability_history'][-100:]) if metrics['stability_history'] else 0
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'stability': f'{belief_metrics["stability"]:.4f}',
                    'ms/step': f'{avg_time*1000:.1f}'
                })
                pbar.update(1)
                
                # Detailed logging every 10 steps
                if trainer.iter_num % 10 == 0:
                    active_patterns = belief_metrics['pattern_stats']['active_patterns']
                    dominant_patterns = belief_metrics['pattern_stats']['dominant_patterns']
                    print(f"\nStep {trainer.iter_num}:")
                    print(f"  Loss: {current_loss:.4f} (avg: {avg_loss:.4f})")
                    print(f"  Stability: {belief_metrics['stability']:.4f} (avg: {avg_stability:.4f})")
                    print(f"  Patterns: {active_patterns} active, {dominant_patterns} dominant")
                    print(f"  Times (ms):")
                    print(f"    Total step: {step_time*1000:.1f}")
                    print(f"    Belief update: {belief_time*1000:.1f}")
                    sys.stdout.flush()
                
            except Exception as e:
                print(f"Error in collect_metrics: {str(e)}")
                traceback.print_exc()
        
        # Initialize the last_time attribute
        collect_metrics.last_time = time.time()
        
        trainer.set_callback('on_batch_end', collect_metrics)
        
        print("\nStarting training...")
        print("(Detailed logs every 10 steps, progress bar updates every step)")
        
        # Time the entire training run
        train_start = time.time()
        trainer.run()
        total_train_time = time.time() - train_start
        
        pbar.close()
        print(f"\nTraining completed in {total_train_time:.1f} seconds")
        
        # Add training summary
        avg_loss = np.mean(metrics['train_losses'])
        avg_stability = np.mean(metrics['stability_history'])
        avg_time_per_step = np.mean(metrics['time_per_step'])
        avg_belief_time = np.mean(metrics['belief_update_times'])
        
        print("\nTraining Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Stability: {avg_stability:.4f}")
        print(f"Average Time per Step: {avg_time_per_step*1000:.1f}ms")
        print(f"Average Belief Update Time: {avg_belief_time*1000:.1f}ms")
        print(f"Total Training Time: {total_train_time:.1f}s")
        
        # Save results with detailed error handling
        try:
            results_path = os.path.join(exp_dir, f'results_seed_{seed}.json')
            print(f"\nAttempting to save results to: {os.path.abspath(results_path)}")
            
            # First verify the directory exists
            if not os.path.exists(exp_dir):
                print(f"Warning: Results directory does not exist: {exp_dir}")
                print(f"Attempting to create it now...")
                os.makedirs(exp_dir, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            results = {
                'seed': seed,
                'config': {k: str(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v 
                          for k, v in config.items()},
                'train_config': {k: str(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v 
                                for k, v in train_config.items()},
                'metrics': {
                    k: [float(x) if isinstance(x, (np.floating, torch.Tensor)) else x 
                        for x in v] if isinstance(v, list) else v
                    for k, v in metrics.items()
                },
                'final_belief_stats': model.get_belief_metrics(),
                'device_used': device,
                'training_summary': {
                    'avg_loss': float(avg_loss),
                    'avg_stability': float(avg_stability),
                    'avg_time_per_step': float(avg_time_per_step),
                    'avg_belief_time': float(avg_belief_time),
                    'total_train_time': float(total_train_time)
                }
            }
            
            print("Results dictionary prepared, attempting to write to file...")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results successfully saved to: {results_path}")
            
            # Verify the file was created
            if os.path.exists(results_path):
                print(f"Verified: Results file exists at {results_path}")
                print(f"File size: {os.path.getsize(results_path)} bytes")
            else:
                print(f"Warning: Results file was not created at {results_path}")
            
            return results
            
        except Exception as e:
            print(f"\nError saving results: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Target directory: {exp_dir}")
            print(f"Target file: {results_path}")
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"\nError in experiment with seed {seed}: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Run multiple experiments with different seeds."""
    try:
        print("Creating experiment directory...")
        exp_dir = create_experiment_dir()
        seeds = [42, 123, 456, 789, 1024]
        
        print(f"Will run experiments with seeds: {seeds}")
        all_results = []
        
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Running experiment with seed {seed}")
            print(f"{'='*50}")
            
            results = run_experiment(seed, exp_dir)
            if results is not None:
                all_results.append(results)
        
        if all_results:
            print("\nAnalyzing results across seeds...")
            # Analyze results across seeds
            stability_means = []
            stability_stds = []
            for results in all_results:
                stability_history = results['metrics']['stability_history']
                stability_means.append(np.mean(stability_history))
                stability_stds.append(np.std(stability_history))
            
            # Plot aggregate results
            plt.figure(figsize=(10, 6))
            plt.errorbar(seeds[:len(all_results)], stability_means, yerr=stability_stds, fmt='o-')
            plt.title('Average Stability Across Seeds')
            plt.xlabel('Seed')
            plt.ylabel('Mean Stability (with std)')
            plt.savefig(os.path.join(exp_dir, 'aggregate_stability.png'))
            plt.close()
            
            # Save aggregate results
            aggregate_results = {
                'seeds': seeds[:len(all_results)],
                'stability_means': stability_means,
                'stability_stds': stability_stds
            }
            with open(os.path.join(exp_dir, 'aggregate_results.json'), 'w') as f:
                json.dump(aggregate_results, f, indent=2)
            
            print("\nExperiment completed successfully!")
        else:
            print("\nNo successful experiments to analyze.")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 