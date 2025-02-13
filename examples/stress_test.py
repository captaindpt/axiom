from src.dynamic_belief_system import MinimalDynamicBeliefSystem
import numpy as np

def print_matrix_stats(P):
    """Print statistics about the belief matrix"""
    print(f"Max value: {np.max(P):.6f}")
    print(f"Min non-zero: {np.min(P[P > 0]):.6f}")
    print(f"Mean non-zero: {np.mean(P[P > 0]):.6f}")
    print(f"Number of non-zero elements: {np.count_nonzero(P)}")
    print(f"Matrix sparsity: {1 - np.count_nonzero(P) / P.size:.2%}")

# Test 1: Rapid Pattern Oscillation
print("\nTest 1: Rapid Pattern Oscillation")
print("=" * 50)
dbs1 = MinimalDynamicBeliefSystem(dimension=5, learning_rate=0.5)  # Higher learning rate
oscillating_patterns = ["0+1", "1+2", "2+3", "3+4"] * 10  # Repeat pattern multiple times
for expr in oscillating_patterns:
    dbs1.process_expression(expr)
print("\nFinal Matrix Stats:")
print_matrix_stats(dbs1.P)

# Test 2: Pattern Contradiction
print("\nTest 2: Pattern Contradiction")
print("=" * 50)
dbs2 = MinimalDynamicBeliefSystem(dimension=3)
contradicting_patterns = [
    "0+1", "0+1", "0+1",  # Establish strong pattern
    "1+0", "1+0", "1+0",  # Reinforce reverse pattern
    "0+1", "1+0", "0+1", "1+0"  # Oscillate
]
for expr in contradicting_patterns:
    dbs2.process_expression(expr)
    print(f"\nAfter {expr}:")
    print(dbs2.P)
    print(f"Stability: {dbs2.stability:.4f}")

# Test 3: Maximum Connectivity
print("\nTest 3: Maximum Connectivity")
print("=" * 50)
dbs3 = MinimalDynamicBeliefSystem(dimension=4)
# Try to connect every node to every other node
for i in range(4):
    for j in range(4):
        if i != j:
            dbs3.process_expression(f"{i}+{j}")
print("\nFinal Matrix Stats:")
print_matrix_stats(dbs3.P)

# Test 4: Stability Recovery
print("\nTest 4: Stability Recovery")
print("=" * 50)
dbs4 = MinimalDynamicBeliefSystem(dimension=3)
# First establish a strong pattern
for _ in range(5):
    dbs4.process_expression("0+1")
print("\nStability after establishing pattern:", dbs4.stability)

# Then introduce noise
for i in range(3):
    for j in range(3):
        if i != j:
            dbs4.process_expression(f"{i}+{j}")
print("\nStability after noise:", dbs4.stability)

# Try to recover original pattern
for _ in range(5):
    dbs4.process_expression("0+1")
print("\nStability after recovery attempt:", dbs4.stability)
print("\nFinal Matrix:")
print(dbs4.P) 