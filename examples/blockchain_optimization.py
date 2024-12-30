import numpy as np
from quark_py import QuantumOptimizer

def optimize_blockchain_consensus():
    # Initialize quantum optimizer
    n_qubits = 4
    optimizer = QuantumOptimizer(n_qubits)
    
    # Create sample transaction graph
    transaction_graph = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    stake_weights = np.array([0.4, 0.3, 0.3])
    
    # Optimize transaction ordering
    ordering, confidence = optimizer.optimize_consensus(
        transaction_graph, 
        stake_weights
    )
    
    print(f"Optimized ordering: {ordering}")
    print(f"Confidence score: {confidence:.2f}")

if __name__ == "__main__":
    optimize_blockchain_consensus()