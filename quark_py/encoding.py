import numpy as np
from typing import Tuple
from scipy.linalg import hadamard

class QubitEncoding:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.hadamard = hadamard(2**n_qubits) / np.sqrt(2**n_qubits)
        
    def encode_graph(self, graph: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n = len(graph)
        state = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Map graph structure to quantum state
        for i in range(n):
            for j in range(n):
                if graph[i,j]:
                    idx = self._graph_index(i, j, n)
                    state[idx] = weights[i]
        
        state /= np.linalg.norm(state)
        return self.hadamard @ state
    
    def encode_weights(self, weights: np.ndarray, importance: np.ndarray) -> np.ndarray:
        flat_weights = weights.flatten()
        flat_importance = importance.flatten()
        state = np.zeros(2**self.n_qubits, dtype=complex)
        
        for i, (w, imp) in enumerate(zip(flat_weights, flat_importance)):
            if i < 2**self.n_qubits:
                state[i] = w * imp
                
        state /= np.linalg.norm(state)
        return self.hadamard @ state
    
    def decode_to_ordering(self, quantum_state: np.ndarray) -> np.ndarray:
        classical_state = self.hadamard @ quantum_state
        n = int(np.sqrt(len(classical_state)))
        ordering = np.zeros(n, dtype=int)
        
        amplitudes = np.abs(classical_state)**2
        for pos in range(n):
            max_amp_idx = np.argmax(amplitudes)
            ordering[pos] = max_amp_idx // n
            amplitudes[max_amp_idx:(max_amp_idx+n)] = 0
            
        return ordering
    
    def decode_to_weights(self, quantum_state: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        classical_state = self.hadamard @ quantum_state
        weights = np.abs(classical_state[:np.prod(shape)])
        return weights.reshape(shape)
    
    def _graph_index(self, i: int, j: int, n: int) -> int:
        return i * n + j