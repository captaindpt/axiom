import numpy as np
from src.dynamic_belief_system import MinimalDynamicBeliefSystem as BeliefSystem
from belief_system_animator import BeliefSystemAnimator
from belief_system_logger import BeliefSystemLogger

class FailureTestingDemo:
    def __init__(self):
        self.belief_system = None
        self.animator = None
        self.logger = None
    
    def _create_attack_matrix(self, pattern):
        """Create an attack matrix for a given pattern."""
        size = 4  # Using 4x4 matrix for this demo
        matrix = np.zeros((size, size))
        
        # Parse pattern (e.g., "0→1" means matrix[0,1] = 1)
        source, target = pattern.split('→')
        matrix[int(source), int(target)] = 1.0
        
        return matrix
    
    def _matrix_to_expression(self, matrix):
        """Convert a matrix to an expression string."""
        rows, cols = np.where(matrix > 0)
        expressions = []
        for i, j in zip(rows, cols):
            expressions.append(f"{i}→{j}")
        return " ".join(expressions)

    def test_psychological_warfare(self):
        """Test psychological warfare attack on belief system"""
        print("\nPSYCHOLOGICAL WARFARE TEST")
        print("==========================")
        
        # Initialize components
        self.belief_system = BeliefSystem(dimension=4)  # Using 4x4 matrices
        self.animator = BeliefSystemAnimator(self.belief_system)
        self.logger = BeliefSystemLogger("psychological_warfare")
        
        # Phase 1: Reality Anchoring
        print("\nPhase 1: Reality Anchoring")
        print("--------------------------")
        initial_states = []
        
        # Build 3 chains of reality anchors
        for i in range(3):
            pattern = f"{i}→{i+1}"
            print(f"\nEstablishing reality anchor: {pattern}")
            
            # Create and apply attack matrix
            attack_matrix = self._create_attack_matrix(pattern)
            self.belief_system.process_expression(self._matrix_to_expression(attack_matrix))
            
            # Log and visualize
            self.logger.log_step("Reality Anchoring", f"Establishing {pattern}", 
                                self.belief_system.P, attack_matrix)
            self.animator.update_state(self.belief_system.P, attack_matrix)
            
            initial_states.append({
                'pattern': pattern,
                'strength': self.belief_system.P[i][i+1]
            })
            
            print(f"Pattern strength: {self.belief_system.P[i][i+1]:.3f}")
            print(f"System stability: {self.belief_system.stability:.3f}")
        
        self.logger.log_phase_summary("Reality Anchoring", {'initial_states': initial_states})
        
        # Phase 2: Reality Distortion Campaign
        print("\nPhase 2: Reality Distortion Campaign")
        print("----------------------------------")
        
        attack_results = []
        for wave in range(5):
            print(f"\nAttack Wave {wave + 1}")
            
            # Create contradictory patterns
            contradictions = [
                (f"{i+1}→{i}", f"{i}→{i+1}")
                for i in range(2)
            ]
            
            for target, source in contradictions:
                print(f"\nAttacking {source} with {target}")
                
                # Create and apply attack
                attack_matrix = self._create_attack_matrix(target)
                self.belief_system.process_expression(self._matrix_to_expression(attack_matrix))
                
                # Log and visualize
                self.logger.log_step("Reality Distortion", 
                                   f"Wave {wave+1}: {target} vs {source}",
                                   self.belief_system.P, attack_matrix)
                self.animator.update_state(self.belief_system.P, attack_matrix)
                
                attack_results.append({
                    'wave': wave + 1,
                    'target': target,
                    'source': source,
                    'impact': self.belief_system.P[int(target[0])][int(target[2])]
                })
                
                print(f"Attack impact: {attack_results[-1]['impact']:.3f}")
                print(f"System stability: {self.belief_system.stability:.3f}")
        
        # Final visualization and logging
        self.animator.plot_final_state()
        self.logger.log_phase_summary("Reality Distortion", {'attack_results': attack_results})
        
        # Calculate destruction rates
        destruction_rates = {}
        for state in initial_states:
            pattern = state['pattern']
            i, j = int(pattern[0]), int(pattern[2])
            initial = state['strength']
            final = self.belief_system.P[i][j]
            rate = ((final - initial) / initial) * 100 if initial > 0 else 0
            destruction_rates[pattern] = rate
        
        print("\nFinal Results")
        print("-------------")
        print(f"Average stability: {self.logger.get_summary_stats()['avg_stability']:.3f}")
        print(f"Min stability: {self.logger.get_summary_stats()['min_stability']:.3f}")
        print(f"Max stability: {self.logger.get_summary_stats()['max_stability']:.3f}")
        print(f"Final stability: {self.belief_system.stability:.3f}")
        print("\nDestruction rates:")
        for pattern, rate in destruction_rates.items():
            print(f"{pattern}: {rate:.1f}%")
        
        return {
            'initial_states': initial_states,
            'attack_results': attack_results,
            'destruction_rates': destruction_rates,
            'final_stability': self.belief_system.stability,
            'stats': self.logger.get_summary_stats()
        }

if __name__ == '__main__':
    demo = FailureTestingDemo()
    demo.test_psychological_warfare()