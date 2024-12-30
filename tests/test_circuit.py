import unittest
import numpy as np
from quark_py.circuit import QuantumCircuit

class TestQuantumCircuit(unittest.TestCase):
    def setUp(self):
        self.circuit = QuantumCircuit(n_qubits=2)
        self.test_state = np.array([1, 0, 0, 0], dtype=complex)
    
    def test_hadamard_gate(self):
        H = self.circuit._hadamard_gate()
        self.assertEqual(H.shape, (2, 2))
        np.testing.assert_array_almost_equal(H @ H, np.eye(2))
    
    def test_cnot_gate(self):
        CNOT = self.circuit._cnot_gate()
        self.assertEqual(CNOT.shape, (4, 4))
        # Test CNOT effect on |00⟩
        result = CNOT @ np.array([1, 0, 0, 0])
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])
        # Test CNOT effect on |10⟩
        result = CNOT @ np.array([0, 0, 1, 0])
        np.testing.assert_array_almost_equal(result, [0, 0, 0, 1])
    
    def test_phase_estimation(self):
        state = self.circuit.phase_estimation(self.test_state)
        self.assertEqual(len(state), 2**self.circuit.n_qubits)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0)

    def test_quantum_interference(self):
        state = self.circuit.quantum_interference(self.test_state)
        self.assertEqual(len(state), 2**self.circuit.n_qubits)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0)