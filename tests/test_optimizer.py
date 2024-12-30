import unittest
import numpy as np
from quark_py import QuantumOptimizer

class TestQuantumOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = QuantumOptimizer(n_qubits=2)
        
    def test_quantum_annealing(self):
        hamiltonian = np.array([[1, 0], [0, -1]])
        initial_state = np.array([1, 0]) / np.sqrt(2)
        final_state = self.optimizer.quantum_annealing(hamiltonian, initial_state)
        self.assertEqual(len(final_state), 2**self.optimizer.n_qubits)

    def test_optimize_consensus(self):
        graph = np.array([[0, 1], [0, 0]])
        weights = np.array([0.6, 0.4])
        ordering, confidence = self.optimizer.optimize_consensus(graph, weights)
        self.assertEqual(len(ordering), len(graph))
        self.assertTrue(0 <= confidence <= 1)
