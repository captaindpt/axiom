from src.dynamic_belief_system_v2 import DynamicBeliefSystem, PatternState
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time

class LLMBeliefSystemDemo:
    """
    Demonstrates how the Dynamic Belief System could enhance LLM reasoning by:
    1. Building strong beliefs from limited but high-quality evidence
    2. Maintaining consistency against contradictory information
    3. Adapting beliefs when presented with strong new evidence
    4. Resisting noise while preserving important patterns
    """
    
    def __init__(self, dimension: int = 10):
        """
        Initialize with a larger dimension to represent more complex belief networks.
        Each node represents a key concept/fact that the LLM needs to maintain.
        """
        self.dbs = DynamicBeliefSystem(
            dimension=dimension,
            learning_rate=0.2,          # Moderate learning rate for balanced adaptation
            temporal_decay_rate=0.1,     # Slow decay to maintain long-term beliefs
            activation_threshold=0.15,    # Higher threshold for belief activation
            dominance_threshold=0.6      # Higher threshold for belief dominance
        )
        self.history: List[Dict] = []
        self.dimension = dimension
    
    def visualize_belief_network(self, title: str = "Belief Network"):
        """Visualize the current state of beliefs as a network heatmap"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.dbs.P, annot=True, cmap='viridis', vmin=0, vmax=1,
                   xticklabels=range(self.dimension),
                   yticklabels=range(self.dimension))
        plt.title(title)
        plt.xlabel("Target Concept")
        plt.ylabel("Source Concept")
        plt.show()
    
    def plot_stability_history(self, events: List[str], stabilities: List[float]):
        """Plot system stability over time with event annotations"""
        plt.figure(figsize=(12, 6))
        plt.plot(stabilities, marker='o')
        plt.title("System Stability Over Time")
        plt.xlabel("Events")
        plt.ylabel("Stability")
        
        # Annotate significant events
        for i, event in enumerate(events):
            if event:  # Only annotate non-empty events
                plt.annotate(event, (i, stabilities[i]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                           arrowprops=dict(arrowstyle='->'))
        
        plt.grid(True)
        plt.show()

    def demonstrate_core_belief_formation(self) -> Tuple[List[str], List[float]]:
        """
        Demonstrate how the system forms strong core beliefs from limited but high-quality evidence.
        In an LLM context, this represents learning fundamental, well-supported facts.
        """
        print("\n=== Core Belief Formation from Limited Evidence ===")
        events = []
        stabilities = []
        
        # Establish core beliefs (e.g., basic facts or principles)
        core_beliefs = [
            ("0+1", "Basic Fact A → B"),
            ("1+2", "Basic Fact B → C"),
            ("2+3", "Basic Fact C → D")
        ]
        
        print("\nEstablishing core beliefs with strong evidence:")
        for belief, description in core_beliefs:
            print(f"\nProcessing: {description}")
            # Reinforce each core belief multiple times (simulating strong evidence)
            for _ in range(4):
                state = self.dbs.process_expression(belief)
                events.append("")
                stabilities.append(self.dbs.stability)
            
            final_state = self.dbs.get_pattern_state(int(belief[0]), int(belief[2]))
            events[-1] = description
            print(f"Belief strength: {final_state['strength']:.3f}")
            print(f"Belief state: {final_state['state']}")
        
        self.visualize_belief_network("Core Belief Network")
        return events, stabilities

    def demonstrate_noise_resistance(self) -> Tuple[List[str], List[float]]:
        """
        Demonstrate how the system maintains core beliefs despite noisy/contradictory information.
        This simulates an LLM's ability to maintain consistent knowledge despite exposure to
        misinformation or contradictory data.
        """
        print("\n=== Noise Resistance Demonstration ===")
        events = []
        stabilities = []
        
        # Record initial state of core beliefs
        core_states_before = {
            (0,1): self.dbs.get_pattern_state(0, 1)['strength'],
            (1,2): self.dbs.get_pattern_state(1, 2)['strength'],
            (2,3): self.dbs.get_pattern_state(2, 3)['strength']
        }
        
        # Introduce noise (contradictory or irrelevant information)
        print("\nIntroducing noisy information:")
        noise_patterns = [
            ("1+0", "Contradiction to A → B"),
            ("4+5", "Irrelevant fact E → F"),
            ("5+6", "Irrelevant fact F → G"),
            ("3+2", "Contradiction to C → D")
        ]
        
        for pattern, description in noise_patterns:
            state = self.dbs.process_expression(pattern)
            events.append(description)
            stabilities.append(self.dbs.stability)
            print(f"\n{description}")
            print(f"System stability: {self.dbs.stability:.3f}")
        
        # Check core beliefs after noise
        print("\nChecking core beliefs after noise:")
        for (i,j), initial_strength in core_states_before.items():
            current_strength = self.dbs.get_pattern_state(i, j)['strength']
            change = current_strength - initial_strength
            print(f"Core belief {i}→{j}:")
            print(f"  Initial strength: {initial_strength:.3f}")
            print(f"  Current strength: {current_strength:.3f}")
            print(f"  Change: {change:.3f}")
        
        self.visualize_belief_network("Belief Network After Noise")
        return events, stabilities

    def demonstrate_belief_adaptation(self) -> Tuple[List[str], List[float]]:
        """
        Demonstrate how the system can adapt beliefs when presented with strong new evidence,
        while maintaining overall stability. This simulates an LLM learning and updating its
        knowledge based on new, well-supported information.
        """
        print("\n=== Belief Adaptation Demonstration ===")
        events = []
        stabilities = []
        
        # Introduce new strong evidence that builds on existing beliefs
        print("\nIntroducing new strong evidence:")
        new_evidence = [
            ("3+4", "New fact D → E"),
            ("4+5", "New fact E → F")
        ]
        
        # First exposure to new evidence
        for pattern, description in new_evidence:
            state = self.dbs.process_expression(pattern)
            events.append(f"First exposure: {description}")
            stabilities.append(self.dbs.stability)
            print(f"\n{description}")
            print(f"Initial strength: {state['strength']:.3f}")
        
        # Reinforce new evidence with supporting information
        print("\nReinforcing new evidence:")
        for pattern, description in new_evidence:
            for _ in range(3):
                state = self.dbs.process_expression(pattern)
                events.append("")
                stabilities.append(self.dbs.stability)
            
            final_state = self.dbs.get_pattern_state(int(pattern[0]), int(pattern[2]))
            events[-1] = f"After reinforcement: {description}"
            print(f"\n{description}")
            print(f"Final strength: {final_state['strength']:.3f}")
            print(f"Final state: {final_state['state']}")
        
        self.visualize_belief_network("Adapted Belief Network")
        return events, stabilities

    def demonstrate_chain_reasoning(self) -> None:
        """
        Demonstrate how the system can support chain reasoning by activating
        sequences of related beliefs. This simulates an LLM's ability to
        make logical connections across its knowledge base.
        """
        print("\n=== Chain Reasoning Demonstration ===")
        
        # Test different paths through the belief network
        paths = [
            [(0,1), (1,2), (2,3)],  # Original path
            [(0,1), (1,2), (2,3), (3,4)],  # Extended path
            [(2,3), (3,4), (4,5)]  # New path
        ]
        
        print("\nTesting reasoning paths:")
        for path in paths:
            print(f"\nPath: {' → '.join(str(n) for n in range(path[0][0], path[-1][1] + 1))}")
            
            # Calculate path strength as the minimum connection strength
            path_strengths = [self.dbs.get_pattern_state(i, j)['strength'] for i,j in path]
            path_strength = min(path_strengths)
            
            print(f"Path strength: {path_strength:.3f}")
            print("Individual connections:")
            for (i,j), strength in zip(path, path_strengths):
                state = self.dbs.get_pattern_state(i, j)
                print(f"  {i}→{j}: {strength:.3f} ({state['state']})")

def main():
    """
    Run a comprehensive demonstration of how the belief system could enhance LLM reasoning.
    """
    print("=== LLM Belief System Enhancement Demonstration ===")
    print("This demonstration shows how a dynamic belief system could help LLMs:")
    print("1. Build strong beliefs from limited but high-quality evidence")
    print("2. Maintain consistency against contradictory information")
    print("3. Adapt beliefs appropriately to new evidence")
    print("4. Support chain reasoning across beliefs")
    
    demo = LLMBeliefSystemDemo(dimension=7)  # Using 7 nodes for demonstration
    
    # Track events and stability throughout the demonstration
    all_events = []
    all_stabilities = []
    
    # 1. Core Belief Formation
    events, stabilities = demo.demonstrate_core_belief_formation()
    all_events.extend(events)
    all_stabilities.extend(stabilities)
    
    # 2. Noise Resistance
    events, stabilities = demo.demonstrate_noise_resistance()
    all_events.extend(events)
    all_stabilities.extend(stabilities)
    
    # 3. Belief Adaptation
    events, stabilities = demo.demonstrate_belief_adaptation()
    all_events.extend(events)
    all_stabilities.extend(stabilities)
    
    # 4. Chain Reasoning
    demo.demonstrate_chain_reasoning()
    
    # Visualize overall stability throughout the demonstration
    demo.plot_stability_history(all_events, all_stabilities)

if __name__ == "__main__":
    main() 