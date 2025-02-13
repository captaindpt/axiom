import numpy as np
import pytest
import time
from src.dynamic_belief_system_v2 import DynamicBeliefSystem, PatternState

def test_pattern_interference():
    """Test how system handles interfering patterns"""
    dbs = DynamicBeliefSystem(
        dimension=4,
        activation_threshold=0.1,
        dominance_threshold=0.5
    )
    
    # Establish first strong pattern
    for _ in range(5):
        dbs.process_expression("0+1")
    state1 = dbs.get_pattern_state(0, 1)
    assert state1['state'] == PatternState.DOMINANT
    
    # Create interfering pattern sharing node 1
    for _ in range(5):
        dbs.process_expression("1+2")
    state2 = dbs.get_pattern_state(1, 2)
    assert state2['state'] == PatternState.DOMINANT
    
    # Check if original pattern maintained strength
    state1_after = dbs.get_pattern_state(0, 1)
    assert state1_after['strength'] >= 0.8 * state1['strength'], "Original pattern should maintain most strength"

def test_pattern_chain_formation():
    """Test if system can maintain chain of related patterns"""
    dbs = DynamicBeliefSystem(dimension=5)
    
    # Create chain: 0->1->2->3
    sequences = [
        ["0+1"] * 4,
        ["1+2"] * 4,
        ["2+3"] * 4
    ]
    
    # Process sequences with interleaving
    for _ in range(3):
        for seq in sequences:
            for expr in seq:
                dbs.process_expression(expr)
    
    # Verify all links in chain are active
    chain_strengths = [
        dbs.get_pattern_state(0, 1)['strength'],
        dbs.get_pattern_state(1, 2)['strength'],
        dbs.get_pattern_state(2, 3)['strength']
    ]
    
    assert all(strength > 0.3 for strength in chain_strengths), "Chain links should maintain significant strength"
    assert max(chain_strengths) - min(chain_strengths) < 0.3, "Chain links should have similar strengths"

def test_pattern_recovery_under_noise():
    """Test pattern recovery after noise interference"""
    dbs = DynamicBeliefSystem(
        dimension=4,
        activation_threshold=0.1,
        temporal_decay_rate=0.2
    )
    
    # Establish strong pattern
    for _ in range(5):
        dbs.process_expression("0+1")
    initial_state = dbs.get_pattern_state(0, 1)
    
    # Introduce noise
    noise_patterns = ["1+2", "2+3", "0+2", "1+3", "2+1", "3+0"]
    for pattern in noise_patterns:
        dbs.process_expression(pattern)
    
    # Attempt recovery
    for _ in range(3):
        dbs.process_expression("0+1")
    
    recovery_state = dbs.get_pattern_state(0, 1)
    assert recovery_state['state'] != PatternState.INACTIVE, "Pattern should recover from noise"
    assert recovery_state['strength'] > 0.7 * initial_state['strength'], "Pattern should recover significant strength"

def test_stability_under_contradiction():
    """Test system stability when faced with contradictory patterns"""
    dbs = DynamicBeliefSystem(
        dimension=3,
        learning_rate=0.2,  # Faster learning for clearer effects
        temporal_decay_rate=0.1
    )
    
    # Establish primary pattern
    for _ in range(5):
        dbs.process_expression("0+1")
        time.sleep(0.01)  # Small delay to ensure temporal effects
    
    # Let system stabilize
    time.sleep(0.05)
    dbs.process_expression("0+1")  # One more to trigger stability calculation
    initial_stability = dbs.stability
    initial_state = dbs.get_pattern_state(0, 1)
    
    assert initial_state['state'] == PatternState.DOMINANT, "Pattern should reach dominant state"
    assert initial_stability > 0.8, "System should reach high stability"
    
    # Introduce contradictory pattern
    for _ in range(3):
        dbs.process_expression("1+0")  # Same nodes, different direction
        time.sleep(0.01)
    
    # Check stability response
    assert dbs.stability < initial_stability, "Stability should decrease under contradiction"
    assert dbs.stability < 0.8, "Stability should be significantly impacted"
    
    # Verify both patterns exist but compete
    forward_state = dbs.get_pattern_state(0, 1)
    backward_state = dbs.get_pattern_state(1, 0)
    
    assert forward_state['strength'] > 0, "Original pattern should maintain some strength"
    assert backward_state['strength'] > 0, "Contradictory pattern should have some strength"
    
    # Strengthen original pattern
    for _ in range(3):
        dbs.process_expression("0+1")
        time.sleep(0.01)
    
    # Check recovery
    recovery_state = dbs.get_pattern_state(0, 1)
    assert recovery_state['strength'] > backward_state['strength'], "Original pattern should dominate after reinforcement"

def test_pattern_dominance_competition():
    """Test competition between patterns for dominance"""
    dbs = DynamicBeliefSystem(
        dimension=4,
        activation_threshold=0.1,
        dominance_threshold=0.5
    )
    
    # Create two strong patterns
    for _ in range(4):
        dbs.process_expression("0+1")
        dbs.process_expression("2+3")
    
    # Verify both patterns reached similar strength
    state1 = dbs.get_pattern_state(0, 1)
    state2 = dbs.get_pattern_state(2, 3)
    
    assert abs(state1['strength'] - state2['strength']) < 0.1, "Independent patterns should reach similar strengths"
    
    # Strengthen one pattern
    for _ in range(3):
        dbs.process_expression("0+1")
    
    # Check if strengthened pattern became dominant while other remained active
    final_state1 = dbs.get_pattern_state(0, 1)
    final_state2 = dbs.get_pattern_state(2, 3)
    
    assert final_state1['state'] == PatternState.DOMINANT, "Strengthened pattern should become dominant"
    assert final_state2['state'] == PatternState.ACTIVE, "Unstrengthened pattern should remain active"

def test_pattern_reactivation_cascade():
    """Test if reactivating one pattern affects related patterns"""
    dbs = DynamicBeliefSystem(
        dimension=5,
        temporal_decay_rate=0.3
    )
    
    # Create connected patterns
    sequences = [
        ["0+1"] * 3,
        ["1+2"] * 3,
        ["2+3"] * 3
    ]
    
    for seq in sequences:
        for expr in seq:
            dbs.process_expression(expr)
    
    # Let patterns decay
    time.sleep(0.2)
    
    # Reactivate middle pattern
    for _ in range(2):
        dbs.process_expression("1+2")
    
    # Check if connected patterns show any reactivation effects
    states = [
        dbs.get_pattern_state(0, 1),
        dbs.get_pattern_state(1, 2),
        dbs.get_pattern_state(2, 3)
    ]
    
    # Middle pattern should be strongest but connected patterns should show some activation
    assert states[1]['strength'] > states[0]['strength'], "Middle pattern should be strongest"
    assert states[0]['strength'] > 0, "Connected patterns should maintain some strength"
    assert states[2]['strength'] > 0, "Connected patterns should maintain some strength" 