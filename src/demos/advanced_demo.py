from src.belief_system import DynamicBeliefSystem, PatternState
from src.utils.animator import BeliefSystemAnimator
import numpy as np
import time

def main():
    """
    Advanced demonstration showing complex belief system state evolution.
    Each step will show:
    1. Current belief matrix state
    2. Current attack/input being processed
    3. System stability
    4. Active patterns
    """
    print("=== Advanced Dynamic Belief System State Evolution ===")
    
    # Initialize larger system for more complex patterns
    dbs = DynamicBeliefSystem(
        dimension=6,  # 6x6 matrix for more complex patterns
        learning_rate=0.15,  # Slower learning for more nuanced evolution
        temporal_decay_rate=0.08  # Slower decay to observe interactions
    )
    animator = BeliefSystemAnimator(dbs)
    
    # Complex test sequences
    test_sequences = [
        # Circular Pattern Formation (0→1→2→0)
        [
            ("0→1", "Creating circular pattern part 1", 3),
            ("1→2", "Creating circular pattern part 2", 3),
            ("2→0", "Completing circular pattern", 3)
        ],
        
        # Intersecting Pattern (1→3→2)
        [
            ("1→3", "Creating intersecting pattern part 1", 2),
            ("3→2", "Completing intersecting pattern", 2)
        ],
        
        # Competing Patterns
        [
            ("4→5", "Creating competing pattern chain 1", 3),
            ("5→3", "Extending chain 1", 2),
            ("5→2", "Creating competing chain 2", 3),
            ("2→3", "Completing competing chain", 2)
        ],
        
        # Pattern Interference
        [
            ("1→0", "Creating interference", 2),
            ("0→1", "Testing pattern resilience", 3),
            ("2→1", "Adding cross-connection", 2)
        ]
    ]
    
    # Process each sequence
    for sequence_idx, sequence in enumerate(test_sequences):
        print(f"\n=== Test Sequence {sequence_idx + 1} ===")
        
        for pattern, description, repetitions in sequence:
            print(f"\n=== {description} ===")
            
            for i in range(repetitions):
                print(f"\nStep {i+1}: Processing {pattern}")
                
                # Create attack matrix
                source, target = map(int, pattern.split('→'))
                attack_matrix = np.zeros((6, 6))
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
                
                # Show overall system stats
                stats = dbs.get_pattern_stats()
                print(f"Active patterns: {stats['active_patterns']}")
                print(f"Dominant patterns: {stats['dominant_patterns']}")
                print(f"Mean strength: {stats['mean_strength']:.3f}")
                
                time.sleep(1)  # Pause to observe state
            
            # Longer pause between pattern sequences
            time.sleep(2)
    
    # Show final state
    animator.plot_final_state()

if __name__ == "__main__":
    main() 