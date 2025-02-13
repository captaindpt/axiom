import numpy as np
import pytest
from src.dynamic_belief_system_v2 import DynamicBeliefSystem, PatternState
import time

def test_initialization():
    """Test proper initialization of the belief system"""
    dbs = DynamicBeliefSystem(dimension=3)
    assert dbs.dimension == 3
    assert dbs.P.shape == (3, 3)
    assert dbs.activation_state.shape == (3, 3)
    assert np.all(dbs.P == 0)
    assert np.all(dbs.activation_state == PatternState.INACTIVE.value)

def test_pattern_activation_states():
    """Test pattern state transitions"""
    dbs = DynamicBeliefSystem(
        dimension=3,
        activation_threshold=0.1,
        dominance_threshold=0.5
    )
    
    # Initial pattern should start inactive
    dbs.process_expression("0+1")
    state = dbs.get_pattern_state(0, 1)
    assert state['state'] == PatternState.INACTIVE
    
    # Pattern should become active after multiple reinforcements
    for _ in range(3):
        dbs.process_expression("0+1")
    state = dbs.get_pattern_state(0, 1)
    assert state['state'] == PatternState.ACTIVE
    
    # Pattern should become dominant after more reinforcement
    for _ in range(5):
        dbs.process_expression("0+1")
    state = dbs.get_pattern_state(0, 1)
    assert state['state'] == PatternState.DOMINANT

def test_pattern_decay():
    """Test pattern decay behavior"""
    dbs = DynamicBeliefSystem(
        dimension=3,
        temporal_decay_rate=0.5,
        activation_threshold=0.1
    )
    
    # Establish pattern
    for _ in range(3):
        dbs.process_expression("0+1")
    
    initial_state = dbs.get_pattern_state(0, 1)
    assert initial_state['state'] != PatternState.INACTIVE
    
    # Let pattern decay
    time.sleep(0.2)
    dbs.process_expression("1+2")  # Process different pattern to trigger decay
    
    decayed_state = dbs.get_pattern_state(0, 1)
    assert decayed_state['strength'] < initial_state['strength']

def test_pattern_confidence():
    """Test pattern confidence calculation"""
    dbs = DynamicBeliefSystem(dimension=3)
    
    # Process pattern multiple times
    for _ in range(3):
        dbs.process_expression("0+1")
    
    state = dbs.get_pattern_state(0, 1)
    assert state['confidence'] > 0
    assert state['frequency'] == 3

def test_pattern_stats():
    """Test pattern statistics tracking"""
    dbs = DynamicBeliefSystem(
        dimension=3,
        activation_threshold=0.1,
        dominance_threshold=0.5
    )
    
    # Create patterns with different strengths
    dbs.process_expression("0+1")  # Should be inactive
    
    for _ in range(3):
        dbs.process_expression("1+2")  # Should become active
    
    for _ in range(5):
        dbs.process_expression("0+2")  # Should become dominant
    
    stats = dbs.get_pattern_stats()
    assert stats['active_patterns'] > 0
    assert stats['dominant_patterns'] > 0
    assert 0 < stats['mean_strength'] <= 1
    assert 0 < stats['mean_confidence'] <= 1

def test_pattern_reactivation():
    """Test pattern reactivation behavior"""
    dbs = DynamicBeliefSystem(
        dimension=3,
        activation_threshold=0.1,
        temporal_decay_rate=0.5
    )
    
    # Establish pattern
    for _ in range(3):
        dbs.process_expression("0+1")
    
    # Let it decay
    time.sleep(0.2)
    dbs.process_expression("1+2")
    
    # Reactivate pattern
    for _ in range(2):
        dbs.process_expression("0+1")
    
    state = dbs.get_pattern_state(0, 1)
    assert state['state'] != PatternState.INACTIVE
    assert state['strength'] > dbs.activation_threshold

def test_stability_calculation():
    """Test stability calculation with activation states"""
    dbs = DynamicBeliefSystem(dimension=3)
    
    # Establish a stable pattern
    for _ in range(3):
        state = dbs.process_expression("0+1")
    stable_state = dbs.stability
    
    # Introduce contradictory pattern
    state = dbs.process_expression("1+0")
    contradicted_stability = dbs.stability
    assert contradicted_stability < stable_state, "Stability should decrease with contradictory pattern"
    
    # Introduce unrelated pattern
    state = dbs.process_expression("1+2")
    new_pattern_stability = dbs.stability
    assert new_pattern_stability != stable_state, "Stability should change with new pattern"
    
    # Reinforce original pattern
    stabilities = []
    for _ in range(3):
        state = dbs.process_expression("0+1")
        stabilities.append(dbs.stability)
    
    # Stability should not decrease during reinforcement
    assert all(stabilities[i] <= stabilities[i+1] for i in range(len(stabilities)-1)), "Stability should not decrease during pattern reinforcement"
    
    # Final stability should be different from contradicted state
    assert stabilities[-1] != contradicted_stability, "Stability should evolve from contradicted state" 