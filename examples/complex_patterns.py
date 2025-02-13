from src.dynamic_belief_system import MinimalDynamicBeliefSystem
import numpy as np

def test_pattern_sequence(dbs, expressions, name="Test"):
    print(f"\n{name}")
    print("=" * 50)
    
    for expr in expressions:
        dbs.process_expression(expr)
        print(f"\nAfter '{expr}':")
        print(dbs.P)
        print(f"Stability: {dbs.stability:.4f}")
    
    return dbs.stability

# Test 1: Mixed Operations (this should work but start showing limitations)
dbs1 = MinimalDynamicBeliefSystem(dimension=5)
mixed_patterns = [
    "1+2", "2+1",  # Commutative pair
    "2+3", "3+2",  # Another commutative pair
    "1+3", "3+1",  # Third commutative pair
    "1+2",         # Reinforcement
    "2+1",         # Reinforcement
]
stability1 = test_pattern_sequence(dbs1, mixed_patterns, "Test 1: Basic Patterns with Reinforcement")

# Test 2: Rapid Pattern Switching (should stress the stability mechanism)
dbs2 = MinimalDynamicBeliefSystem(dimension=5)
switching_patterns = [
    "1+2", "2+1",
    "3+4", "4+3",
    "1+2", "3+4",
    "2+1", "4+3",
    "1+3", "2+4",
    "3+1", "4+2"
]
stability2 = test_pattern_sequence(dbs2, switching_patterns, "Test 2: Rapid Pattern Switching")

# Test 3: Pattern Interference (should reveal limitations)
dbs3 = MinimalDynamicBeliefSystem(dimension=5)
interference_patterns = [
    "1+2", "2+1",  # Establish first pattern
    "1+2", "2+1",  # Reinforce it
    "2+3", "3+2",  # New pattern sharing node 2
    "1+3", "3+1",  # Pattern that might interfere
    "1+2", "2+1",  # Try to maintain original pattern
]
stability3 = test_pattern_sequence(dbs3, interference_patterns, "Test 3: Pattern Interference")

# Test 4: Dense Pattern Network (should definitely stress the system)
dbs4 = MinimalDynamicBeliefSystem(dimension=5)
dense_patterns = [
    "0+1", "1+0",
    "1+2", "2+1",
    "2+3", "3+2",
    "3+4", "4+3",
    "0+2", "2+0",
    "1+3", "3+1",
    "2+4", "4+2",
]
stability4 = test_pattern_sequence(dbs4, dense_patterns, "Test 4: Dense Pattern Network")

print("\nFinal Stability Comparison")
print("=" * 50)
print(f"Basic Patterns Stability: {stability1:.4f}")
print(f"Switching Patterns Stability: {stability2:.4f}")
print(f"Interference Patterns Stability: {stability3:.4f}")
print(f"Dense Patterns Stability: {stability4:.4f}")

# Analysis of final states
print("\nPattern Strength Analysis")
print("=" * 50)

def analyze_patterns(P, threshold=0.01):
    connections = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j] > threshold:
                connections.append((i, j, P[i,j]))
    return sorted(connections, key=lambda x: x[2], reverse=True)

print("\nDense Network Final Patterns (strength > 0.01):")
for i, j, strength in analyze_patterns(dbs4.P):
    print(f"{i}→{j}: {strength:.4f}") 