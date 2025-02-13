import numpy as np
import pytest
from src.dynamic_belief_system import MinimalDynamicBeliefSystem
import time

def test_temporal_decay():
    """Test that patterns decay over time"""
    dbs = MinimalDynamicBeliefSystem(dimension=3, temporal_decay_rate=0.5)
    
    # Establish initial pattern
    dbs.process_expression("0+1")
    initial_strength = dbs.P[0,1]
    
    # Wait a short time
    time.sleep(0.1)
    
    # Process a different pattern
    dbs.process_expression("1+2")
    
    # Check that original pattern has decayed
    assert dbs.P[0,1] < initial_strength

def test_pattern_importance():
    """Test that frequently used patterns become more important"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    
    # Establish pattern with multiple repetitions
    for _ in range(3):
        dbs.process_expression("0+1")
    
    frequent_strength = dbs.P[0,1]
    
    # Establish different pattern once
    dbs.process_expression("1+2")
    
    # Frequent pattern should be stronger
    assert dbs.P[0,1] > dbs.P[1,2]

def test_weak_pattern_influence():
    """Test that weak patterns have diminishing influence over time"""
    dbs = MinimalDynamicBeliefSystem(dimension=3, importance_threshold=0.05)
    
    # Create a weak pattern
    dbs.process_expression("0+1")
    initial_weak_strength = dbs.P[0,1]
    
    # Create a strong pattern
    for _ in range(3):
        dbs.process_expression("1+2")
    strong_pattern_strength = dbs.P[1,2]
    
    # Verify initial conditions
    assert initial_weak_strength > 0, "Weak pattern should initially exist"
    assert strong_pattern_strength > initial_weak_strength, "Strong pattern should be stronger than weak pattern"
    
    # Let patterns evolve
    time.sleep(0.5)
    
    # Force pattern evaluation
    dbs.process_expression("1+2")
    
    # Check relative influence
    final_weak_strength = dbs.P[0,1]
    final_strong_strength = dbs.P[1,2]
    
    # Weak pattern should have diminished
    assert final_weak_strength < initial_weak_strength, "Weak pattern should diminish over time"
    assert final_weak_strength < final_strong_strength, "Weak pattern should have less influence than strong pattern"
    
    # Strong pattern should maintain significant strength
    assert final_strong_strength > 0.5 * strong_pattern_strength, "Strong pattern should maintain significant strength"
    
    # Verify pattern evolution
    strength_ratio_initial = initial_weak_strength / strong_pattern_strength
    strength_ratio_final = final_weak_strength / final_strong_strength
    assert strength_ratio_final < strength_ratio_initial, "Weak pattern should lose relative influence over time"

def test_pattern_stats():
    """Test pattern statistics tracking"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    
    # Process some patterns
    dbs.process_expression("0+1")
    dbs.process_expression("0+1")
    dbs.process_expression("1+2")
    
    stats = dbs.get_pattern_stats()
    
    assert stats['active_patterns'] > 0
    assert stats['max_strength'] > 0
    assert stats['mean_strength'] > 0
    assert np.any(stats['pattern_frequency'] > 1)  # Some patterns used multiple times
    assert stats['oldest_pattern_age'] > 0

def test_stability_with_importance():
    """Test that stability calculation considers pattern importance"""
    dbs = MinimalDynamicBeliefSystem(dimension=3)
    
    # Establish important pattern
    for _ in range(3):
        dbs.process_expression("0+1")
    
    initial_stability = dbs.stability
    
    # Add less important pattern
    dbs.process_expression("1+2")
    
    # Stability should be maintained due to importance weighting
    assert dbs.stability >= initial_stability * 0.9  # Allow for small fluctuation 