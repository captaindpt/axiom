import numpy as np
import pytest
from src.dynamic_belief_system import MinimalDynamicBeliefSystem
import time

def test_initialization():
    """Test proper initialization of the minimal belief system"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    assert dbs.dimension == 3
    assert dbs.P.shape == (3, 3)
    assert np.all(dbs.P == 0)

def test_pattern_processing():
    """Test basic pattern processing"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    
    # Process a pattern
    dbs.process_expression("0+1")
    assert dbs.P[0,1] > 0
    
    # Process it again to strengthen
    dbs.process_expression("0+1")
    assert dbs.P[0,1] > 0.1

def test_temporal_decay():
    """Test temporal decay of patterns"""
    dbs = MinimalDynamicBeliefSystem(
        dimension=3,
        temporal_decay_rate=0.5
    )
    
    # Create pattern
    dbs.process_expression("0+1")
    initial_strength = dbs.P[0,1]
    
    # Let it decay
    time.sleep(0.2)
    dbs.process_expression("1+2")  # Process different pattern to trigger decay
    
    assert dbs.P[0,1] < initial_strength

def test_pattern_importance():
    """Test pattern importance calculation"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    
    # Create frequently used pattern
    for _ in range(3):
        dbs.process_expression("0+1")
    
    # Create less frequent pattern
    dbs.process_expression("1+2")
    
    assert dbs.P[0,1] > dbs.P[1,2]

def test_stability_calculation():
    """Test stability calculation"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    
    # System should start with valid stability
    assert 0 <= dbs.stability <= 1
    
    # Add some patterns
    dbs.process_expression("0+1")
    dbs.process_expression("1+2")
    
    # Stability should change but remain valid
    assert 0 <= dbs.stability <= 1
    
    # Reinforce a pattern to increase stability
    for _ in range(3):
        dbs.process_expression("0+1")
    
    # Stability should increase with pattern reinforcement
    assert dbs.stability > 0

def test_basic_pattern_formation():
    """Test basic pattern formation with simple arithmetic pairs"""
    dbs = MinimalDynamicBeliefSystem(dimension=4)
    
    # Process commutative pairs
    dbs.process_expression("2+3")
    dbs.process_expression("3+2")
    
    # Check that pattern is forming
    assert dbs.P[2, 3] > 0
    assert dbs.P[3, 2] > 0
    assert np.allclose(dbs.P[2, 3], dbs.P[3, 2])  # Should be symmetric

def test_stability_increases():
    """Test that stability increases with consistent patterns"""
    dbs = MinimalDynamicBeliefSystem(dimension=4)
    
    # Process same expression multiple times
    stabilities = []
    for _ in range(5):
        dbs.process_expression("1+2")
        stabilities.append(dbs.stability)
    
    # Check that stability generally increases
    assert stabilities[-1] > stabilities[0]

def test_history_tracking():
    """Test that history is properly tracked"""
    dbs = MinimalDynamicBeliefSystem(dimension=2)
    
    # Process a few expressions
    expressions = ["0+1", "1+0", "0+1"]
    for expr in expressions:
        dbs.process_expression(expr)
    
    # Check history
    history = dbs.get_history()
    assert len(history) == len(expressions)
    assert all(isinstance(h[0], np.ndarray) and isinstance(h[1], float) 
              for h in history)

def test_invalid_expression_handling():
    """Test that invalid expressions are handled gracefully"""
    dbs = MinimalDynamicBeliefSystem(dimension=2)
    
    # These should not raise exceptions
    dbs.process_expression("invalid")
    dbs.process_expression("5+6")  # Out of dimension bounds
    dbs.process_expression("")
    
    # System should maintain valid state
    assert np.all(dbs.P >= 0)
    assert np.all(dbs.P <= 1) 