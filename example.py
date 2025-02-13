from src.dynamic_belief_system import MinimalDynamicBeliefSystem
import numpy as np

# Create a belief system with 4x4 dimension
dbs = MinimalDynamicBeliefSystem(dimension=4)

# Process a series of commutative expressions
expressions = [
    "1+2", "2+1",  # First pair
    "2+3", "3+2",  # Second pair
    "1+3", "3+1",  # Third pair
]

print("Training the system with commutative pairs...")
print("-" * 50)

for expr in expressions:
    dbs.process_expression(expr)
    print(f"\nAfter processing '{expr}':")
    print("Belief Matrix P:")
    print(dbs.P)
    print(f"Stability: {dbs.stability:.4f}")

print("\nFinal Belief Matrix:")
print("-" * 50)
print(dbs.P)
print("\nFinal Stability:", dbs.stability)

# The system should have learned commutative patterns
# Check if P[i,j] ≈ P[j,i] for all pairs
print("\nChecking Commutativity Learning:")
print("-" * 50)
for i in range(4):
    for j in range(i+1, 4):
        if dbs.P[i,j] > 0:
            print(f"P[{i},{j}] = {dbs.P[i,j]:.4f}, P[{j},{i}] = {dbs.P[j,i]:.4f}") 