import unittest
import numpy as np
from quark_py.encoding import QubitEncoding

class TestQubitEncoding(unittest.TestCase):
    def setUp(self):
        self.encoder = QubitEncoding(n_qubits=2)
        
    def test_encode_graph(self):
        graph = np.array([[0, 1], [0, 0]])
        weights = np.array([0.6, 0.4])
        state = self.encoder.encode_graph(graph, weights)
        self.assertEqual(len(state), 2**self.encoder.n_qubits)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0)
    
    def test_encode_weights(self):
        weights = np.array([[1.0, -0.5], [0.3, 0.8]])
        importance = np.ones_like(weights)
        state = self.encoder.encode_weights(weights, importance)
        self.assertEqual(len(state), 2**self.encoder.n_qubits)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0)
    
    def test_decode_to_ordering(self):
        state = np.ones(2**self.encoder.n_qubits) / np.sqrt(2**self.encoder.n_qubits)
        ordering = self.encoder.decode_to_ordering(state)
        self.assertTrue(isinstance(ordering, np.ndarray))
        self.assertTrue(np.all(ordering >= 0))
    
    def test_decode_to_weights(self):
        state = np.ones(2**self.encoder.n_qubits) / np.sqrt(2**self.encoder.n_qubits)
        shape = (2, 2)
        weights = self.encoder.decode_to_weights(state, shape)
        self.assertEqual(weights.shape, shape)