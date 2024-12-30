import unittest
import numpy as np
from quark_py.compression import ModelCompressor

class TestModelCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = ModelCompressor(n_qubits=2)
        self.test_weights = {
            "layer1": np.random.randn(4, 4),
            "layer2": np.random.randn(4, 2)
        }
    
    def test_compress_model(self):
        compressed = self.compressor.compress_model(
            self.test_weights,
            target_size=0.5
        )
        self.assertEqual(len(compressed), len(self.test_weights))
        for layer_name in self.test_weights:
            self.assertIn(layer_name, compressed)
            self.assertEqual(compressed[layer_name].shape, 
                           self.test_weights[layer_name].shape)
    
    def test_calculate_importance(self):
        importance = self.compressor._calculate_importance(self.test_weights)
        for layer_name in self.test_weights:
            self.assertIn(layer_name, importance)
            self.assertEqual(importance[layer_name].shape,
                           self.test_weights[layer_name].shape)
            self.assertTrue(np.all(importance[layer_name] >= 0))
            self.assertLessEqual(np.max(importance[layer_name]), 1.0)