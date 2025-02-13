from src.belief_system import DynamicBeliefSystem, PatternState
from src.utils.animator import BeliefSystemAnimator
import numpy as np
import time

def main():
    """
    Simple demonstration showing belief system state evolution.
    Each step will show:
    1. Current belief matrix state
    2. Current attack/input being processed
    3. System stability
    4. Active patterns
    """
    print("=== Dynamic Belief System State Evolution ===")
    
    # Initialize system
    dbs = DynamicBeliefSystem(dimension=4)  # 4x4 matrix for clarity
    animator = BeliefSystemAnimator(dbs)
    
    # Test sequence showing different pattern formations
    test_sequence = [
        # Establish base pattern
        ("0→1", "Establishing base pattern", 3),
        # Create connected pattern
        ("1→2", "Creating connected pattern", 2),
        # Add conflicting pattern
        ("1→0", "Introducing conflict", 2),
        # Reinforce original pattern
        ("0→1", "Reinforcing original pattern", 2),
        # Create distant pattern
        ("2→3", "Creating distant pattern", 2),
    ]
    
    # Process each test case
    for pattern, description, repetitions in test_sequence:
        print(f"\n=== {description} ===")
        
        for i in range(repetitions):
            print(f"\nStep {i+1}: Processing {pattern}")
            
            # Create attack matrix
            source, target = map(int, pattern.split('→'))
            attack_matrix = np.zeros((4, 4))
            attack_matrix[source, target] = 1.0
            
            # Process pattern
            dbs.process_expression(f"{source}+{target}")
            
            # Update visualization
            animator.update_state(dbs.P, attack_matrix)
            
            # Show current state info
            state = dbs.get_pattern_state(source, target)
            print(f"Pattern strength: {state['strength']:.3f}")
            print(f"Pattern state: {state['state']}")
            print(f"System stability: {dbs.stability:.3f}")
            
            time.sleep(1)  # Pause to observe state
    
    # Show final state
    animator.plot_final_state()

if __name__ == "__main__":
    main() 