# Belief-Enhanced GPT Experiment

This experiment compares a standard minGPT implementation with a belief-enhanced version that incorporates a dynamic belief system. The goal is to demonstrate improved performance in maintaining consistent beliefs and learning from limited but high-quality evidence.

## Architecture Overview

### Belief-Enhanced Attention

The key innovation is the integration of a dynamic belief system into the transformer's attention mechanism. This is accomplished through the `BeliefEnhancedAttention` class which:

1. Projects attention outputs into a belief space
2. Processes patterns through the belief system
3. Projects enhanced patterns back to the embedding space
4. Combines with original attention via residual connection

```
Input -> Standard Attention -> Project to Belief Space -> 
  Process through Belief System -> Project back -> 
    Residual Connection -> Output
```

### Key Components

- `BeliefGPT`: Main model that enhances GPT with belief system
- `BeliefEnhancedBlock`: Modified transformer block with belief attention
- `BeliefEnhancedAttention`: Core attention mechanism with belief integration
- `DynamicBeliefSystem`: Pattern processing and stability maintenance

## Running the Experiment

### Quick Start

```bash
# From project root
./run_comparison.sh
```

This will:
1. Set up a Python virtual environment
2. Install dependencies
3. Run the full comparison pipeline

### Advanced Usage

```bash
# Use custom configuration
./run_comparison.sh --config custom_config.json

# Skip unit tests
./run_comparison.sh --skip-tests
```

## Configuration

The experiment is controlled by a hierarchical configuration system:

```python
ExperimentConfig
├── ModelConfig         # Architecture parameters
│   ├── Standard GPT parameters (n_layer, n_head, etc.)
│   └── Belief system parameters (dimension, thresholds, etc.)
├── TrainingConfig     # Training hyperparameters
└── EvaluationConfig   # Inference and evaluation settings
```

### Default Configuration

```python
ModelConfig:
  n_layer: 12
  n_head: 12
  n_embd: 768
  vocab_size: 50257
  block_size: 1024
  belief_dimension: 10
  belief_learning_rate: 0.1
  belief_temporal_decay_rate: 0.1
  belief_activation_threshold: 0.1
  belief_dominance_threshold: 0.5

TrainingConfig:
  batch_size: 64
  learning_rate: 6e-4
  max_iters: 100000
  warmup_iters: 2000
  grad_norm_clip: 1.0
  weight_decay: 0.1
  betas: (0.9, 0.95)

EvaluationConfig:
  batch_size: 32
  num_samples: 1000
  max_tokens: 100
  temperature: 1.0
```

## Dataset Strategy

### Training Tasks

1. **Character-Level Language Modeling** (from minGPT's `chargpt`)
   - Base task: Character prediction on input.txt
   - Belief enhancement: Track consistency in character patterns
   - Metrics: 
     - Standard: Character prediction accuracy
     - Belief: Pattern stability across context

2. **Number Addition** (from minGPT's `adder`)
   - Base task: Adding n-digit numbers
   - Belief enhancement: Mathematical relationship stability
   - Metrics:
     - Standard: Addition accuracy
     - Belief: Consistency in mathematical patterns

3. **Sequence Sorting** (from minGPT's demo)
   - Base task: Sorting number sequences
   - Belief enhancement: Order relationship stability
   - Metrics:
     - Standard: Sorting accuracy
     - Belief: Stability of ordering beliefs

### Validation Strategy

For each task, we create three types of validation sets:

1. **Standard Validation**
   - Direct from minGPT's original validation sets
   - Ensures we maintain base performance

2. **Belief Stability Tests**
   - Character-Level:
     ```
     Input: "The capital of France is Paris. The capital of England is London. The capital of France is Berlin."
     Expected: Model should maintain Paris as France's capital
     ```
   - Addition:
     ```
     Input: "2+2=4, 3+3=6, 2+2=5"
     Expected: Model should maintain 2+2=4
     ```
   - Sorting:
     ```
     Input: "1,2,3 -> 1,2,3; 4,5,6 -> 4,5,6; 1,2,3 -> 3,2,1"
     Expected: Model should maintain ascending sort order
     ```

3. **Cross-Task Transfer**
   - Test if beliefs learned in one domain transfer to others
   - Example: Number ordering from sorting → number magnitude in addition

### Testing Protocol

1. **Base Performance** (minGPT's original metrics)
   - Character-level perplexity
   - Addition accuracy
   - Sorting accuracy

2. **Belief System Metrics**
   - Pattern Stability Score:
     ```python
     def calculate_stability(model, test_cases):
         stability_scores = []
         for base, contradiction, probe in test_cases:
             # Feed base knowledge
             model.process(base)
             initial_belief = model.get_belief()
             
             # Introduce contradiction
             model.process(contradiction)
             
             # Test with probe
             model.process(probe)
             final_belief = model.get_belief()
             
             # Calculate stability
             stability = belief_consistency(initial_belief, final_belief)
             stability_scores.append(stability)
         return np.mean(stability_scores)
     ```

   - Learning Efficiency:
     ```python
     def calculate_learning_efficiency(model, task_data):
         examples_needed = 0
         while not model.has_stable_belief():
             model.process(next(task_data))
             examples_needed += 1
         return examples_needed
     ```

3. **Cross-Domain Transfer**
   ```python
   def test_cross_domain_transfer(model, source_task, target_task):
       # Train on source task
       model.train(source_task)
       source_beliefs = model.get_beliefs()
       
       # Test on target task
       transfer_score = evaluate_belief_transfer(
           source_beliefs, 
           target_task
       )
       return transfer_score
   ```

### Success Criteria

1. **Must Match or Exceed Base Performance**
   - Character-level: Match minGPT perplexity
   - Addition: Match minGPT accuracy
   - Sorting: Match minGPT accuracy

2. **Must Show Belief Stability**
   - >95% stability on contradictory inputs
   - >90% pattern recognition
   - >85% cross-task transfer

3. **Must Demonstrate Efficiency**
   - 30% fewer examples needed for stable learning
   - 25% faster convergence on new patterns
   - 20% better generalization to novel cases

### Data Generation

All data generation builds on minGPT's existing datasets:

```python
class BeliefEnhancedDataset:
    def __init__(self, base_dataset, belief_tests):
        self.base = base_dataset
        self.belief_tests = belief_tests
    
    def __getitem__(self, idx):
        if idx < len(self.base):
            # Return standard minGPT example
            return self.base[idx]
        else:
            # Return belief stability test
            return self.belief_tests[idx - len(self.base)]

def generate_belief_tests(task_type):
    if task_type == "char":
        return generate_char_belief_tests()
    elif task_type == "addition":
        return generate_addition_belief_tests()
    elif task_type == "sorting":
        return generate_sorting_belief_tests()
```

### Running Experiments

1. **Individual Task Training**
```bash
# Train on character-level task
./run_comparison.sh --task char

# Train on addition task
./run_comparison.sh --task addition

# Train on sorting task
./run_comparison.sh --task sorting
```

2. **Full Evaluation**
```bash
# Run all tasks and evaluations
./run_comparison.sh --full-eval
```

3. **Custom Configuration**
```bash
# Use custom hyperparameters
./run_comparison.sh --config custom_config.json
```

## Pipeline Stages

1. **Test Suite**
   - Runs all unit tests for both models
   - Ensures basic functionality and integration

2. **Training**
   - Trains both models under identical conditions
   - Collects training metrics:
     - Loss
     - Belief stability
     - Pattern strength
     - Training speed

3. **Evaluation**
   - Tests both models on:
     - Language modeling
     - Belief consistency
     - Pattern recognition
     - Adaptation to new evidence

4. **Visualization**
   - Generates comparison plots
   - Metrics over time
   - Performance differences
   - Belief system dynamics

## Results Directory Structure

Each experiment run creates a timestamped directory:

```
results/
└── experiment_YYYYMMDD_HHMMSS/
    ├── config.json              # Experiment configuration
    ├── baseline_model.pt        # Saved baseline model
    ├── belief_model.pt          # Saved belief model
    ├── training_metrics.json    # Training performance data
    ├── evaluation_results.json  # Evaluation metrics
    ├── training_comparison.png  # Training visualization
    └── evaluation_comparison.png # Evaluation visualization
```

## Metrics Tracked

### Training Metrics
- Loss
- Learning rate
- Gradient norms
- Belief stability
- Pattern strength
- Training time

### Evaluation Metrics
- Perplexity
- Belief consistency score
- Pattern recognition accuracy
- Adaptation speed
- Inference time

## Extending the Experiment

To modify the experiment:

1. **Change Model Architecture**
   - Modify `BeliefEnhancedAttention` in `models/belief_gpt.py`
   - Adjust belief system integration

2. **Add New Metrics**
   - Add to `get_belief_metrics()` in `BeliefGPT`
   - Update visualization in `utils/visualization.py`

3. **Modify Training**
   - Adjust parameters in `config.py`
   - Modify training loop in `training/trainer.py`

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for full list

## Future Improvements

1. **Architecture**
   - Experiment with different belief integration methods
   - Try multiple belief systems per layer
   - Investigate belief system sharing across layers

2. **Training**
   - Implement distributed training
   - Add gradient accumulation
   - Experiment with different optimizers

3. **Evaluation**
   - Add more sophisticated belief consistency tests
   - Implement challenge datasets
   - Add interpretability analysis

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## References

- Original minGPT implementation
- Dynamic belief system paper/implementation
- Transformer architecture papers
- Related work in belief/knowledge integration 