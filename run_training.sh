#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run training
python experiments/mingpt_belief/train_with_beliefs.py 