#!/bin/bash

# Ensure PYTHONPATH includes necessary directories
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/requirements_installed
fi

# Run the comparison pipeline
echo "Running belief-enhanced GPT comparison pipeline..."
python experiments/belief_vs_mingpt/run_pipeline.py "$@"

# Deactivate virtual environment
deactivate 