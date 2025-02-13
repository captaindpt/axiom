# Dynamic Belief System

A Python implementation of a dynamic belief system designed to enhance large language models' ability to maintain consistent beliefs and learn effectively from limited but high-quality evidence. The system models pattern formation, temporal dynamics, and stability in cognitive networks.

## Acknowledgments

This project builds upon [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy, which provides an elegant and minimal PyTorch implementation of GPT. The minGPT code is used as a foundation for the transformer architecture in this project. All credit for the original minGPT implementation goes to Andrej Karpathy.

## Key Applications for LLMs

This belief system addresses several critical challenges in LLM reasoning:

1. **Consistency Maintenance**: Helps LLMs maintain consistent beliefs even when exposed to contradictory information
2. **Efficient Learning**: Enables learning from small quantities of high-quality evidence
3. **Belief Adaptation**: Allows controlled updates to beliefs when presented with strong new evidence
4. **Chain Reasoning**: Supports making logical connections across established beliefs

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dynamic_belief_system.py      # Minimal implementation
│   ├── dynamic_belief_system_v2.py   # Enhanced implementation
│   ├── utils/
│   │   └── animator.py
│   ├── demos/
│   │   └── llm_belief_demo.py        # LLM enhancement demonstration
│   └── tests/
│       ├── __init__.py
│       ├── integration/
│       │   └── test_complex_scenarios.py
│       └── unit/
│           ├── test_belief_system_v2.py
│           ├── test_dynamic_belief_system.py
│           └── test_enhanced_features.py
├── demo.py                           # Interactive demonstration
└── stress_test.py                    # System stress testing
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dynamic-belief-system
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## LLM Enhancement Demonstration

The project includes a comprehensive demonstration of how this belief system could enhance LLM reasoning capabilities:

```bash
python src/demos/llm_belief_demo.py
```

This demonstration showcases:

1. **Core Belief Formation**
   - Building strong beliefs from limited but high-quality evidence
   - Establishing fundamental knowledge with high confidence

2. **Noise Resistance**
   - Maintaining consistent beliefs despite contradictory information
   - Distinguishing between noise and valid updates

3. **Belief Adaptation**
   - Controlled updates to beliefs when presented with strong evidence
   - Balancing stability with adaptability

4. **Chain Reasoning**
   - Making logical connections across established beliefs
   - Evaluating the strength of reasoning chains

The demonstration includes visualizations of:
- Belief network evolution
- System stability over time
- Pattern strength and state transitions
- Reasoning chain formation

## Running Tests

The project uses pytest for testing. There are three types of tests:
- Unit tests for basic functionality
- Integration tests for complex scenarios
- Enhanced feature tests

To run all tests:
```bash
PYTHONPATH=$PYTHONPATH:. python -m pytest src/tests/ -v
```

To run specific test categories:
```bash
# Run only unit tests
PYTHONPATH=$PYTHONPATH:. python -m pytest src/tests/unit/ -v

# Run only integration tests
PYTHONPATH=$PYTHONPATH:. python -m pytest src/tests/integration/ -v
```

## Basic Usage

```python
from src.dynamic_belief_system import MinimalDynamicBeliefSystem

# Create a belief system with dimension 3
dbs = MinimalDynamicBeliefSystem(dimension=3)

# Process patterns
dbs.process_expression("0+1")  # Create connection between nodes 0 and 1
dbs.process_expression("1+2")  # Create connection between nodes 1 and 2

# Get pattern statistics
stats = dbs.get_pattern_stats()
print(stats)
```

## Enhanced Usage

```python
from src.dynamic_belief_system_v2 import DynamicBeliefSystem, PatternState

# Create enhanced belief system
dbs = DynamicBeliefSystem(
    dimension=3,
    learning_rate=0.2,
    temporal_decay_rate=0.1,
    activation_threshold=0.1,
    dominance_threshold=0.5
)

# Process and check pattern state
state = dbs.process_expression("0+1")
print(f"Pattern State: {state['state']}")
print(f"Pattern Strength: {state['strength']}")
print(f"System Stability: {dbs.stability}")
```

## Configuration

Both implementations accept several parameters:

- `dimension`: Size of the belief matrix (default: 10)
- `learning_rate`: Rate of pattern reinforcement (default: 0.1)
- `temporal_decay_rate`: Rate of pattern decay over time (default: 0.1)
- `activation_threshold`: Threshold for pattern activation (v2 only, default: 0.1)
- `dominance_threshold`: Threshold for pattern dominance (v2 only, default: 0.5)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write and run tests for your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Belief-Enhanced GPT Training

This repository contains code for training a GPT model enhanced with a belief system.

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Training

The main training script is located at `experiments/mingpt_belief/train_with_beliefs.py`. 

To run the training:
```bash
# Add the current directory to PYTHONPATH and run the training script
PYTHONPATH=$PYTHONPATH:. python experiments/mingpt_belief/train_with_beliefs.py
```

### GPU Configuration

The code automatically detects and uses the best available device (CUDA GPU/Apple Metal/CPU). No additional configuration is needed.

### Training Parameters

Current configuration:
- Model: GPT-nano (smallest variant)
- Batch size: 8
- Max iterations: 1000
- Learning rate: 3e-4
- Vocabulary size: 100
- Block size: 32

### Results

Results will be saved in the `results/experiment_<timestamp>` directory, including:
- Training metrics (loss, stability)
- Model configurations
- Performance statistics
- Plots of training progress

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (optional but recommended) 