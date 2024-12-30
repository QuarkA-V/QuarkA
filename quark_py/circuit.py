import numpy as np
from typing import List, Optional
from scipy.linalg import expm

class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.H = self._hadamard_gate()
        self.CNOT = self._cnot_gate()
        
    def apply_hamiltonian(self, state: np.ndarray, hamiltonian: np.ndarray, time: float) -> np.ndarray:
        evolution = expm(-1j * hamiltonian * time)
        return evolution @ state
    
    def phase_estimation(self, state: np.ndarray) -> np.ndarray:
        # Hadamard layer
        state = np.kron(self.H, np.eye(2**(self.n_qubits-1))) @ state
        
        # Controlled phase rotations
        for i in range(self.n_qubits):
            phase = 2 * np.pi / (2**(i+1))
            control = i
            for target in range(self.n_qubits):
                if target != control:
                    state = self._controlled_phase(state, control, target, phase)
        
        state = self._qft_inverse(state)
        return state
    
    def quantum_interference(self, state: np.ndarray) -> np.ndarray:
        # Create that sweet superposition
        state = np.kron(self.H, np.eye(2**(self.n_qubits-1))) @ state
        
        # Phase shifts for interference
        for i in range(self.n_qubits):
            state = self._phase_shift(state, i, np.pi/2)
            
        # Entangle the qubits
        for i in range(self.n_qubits-1):
            state = self._apply_cnot(state, i, i+1)
            
        return state
    
# Continuing quantum/circuit.py with helper methods
    def _hadamard_gate(self) -> np.ndarray:
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return H
    
    def _cnot_gate(self) -> np.ndarray:
        return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
    
    def _controlled_phase(self, state: np.ndarray, control: int, target: int, phase: float) -> np.ndarray:
        n = len(state)
        controlled_phase = np.eye(n, dtype=complex)
        
        for i in range(n):
            if (i >> control) & 1 and (i >> target) & 1:
                controlled_phase[i,i] = np.exp(1j * phase)
                
        return controlled_phase @ state
    
    def _qft_inverse(self, state: np.ndarray) -> np.ndarray:
        n = len(state)
        qft = np.zeros((n,n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                qft[i,j] = np.exp(-2j * np.pi * i * j / n) / np.sqrt(n)
                
        return qft.conj().T @ state
    
    def _phase_shift(self, state: np.ndarray, qubit: int, phase: float) -> np.ndarray:
        n = len(state)
        phase_matrix = np.eye(n, dtype=complex)
        
        for i in range(n):
            if (i >> qubit) & 1:
                phase_matrix[i,i] = np.exp(1j * phase)
                
        return phase_matrix @ state
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        n = len(state)
        cnot = np.eye(n)
        
        for i in range(n):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                cnot[i,j] = 1
                cnot[i,i] = 0
                
        return cnot @ state