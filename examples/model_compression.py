import numpy as np
from quark_py import ModelCompressor

def compress_neural_network():
    # Initialize model compressor
    compressor = ModelCompressor(n_qubits=4)
    
    # Create sample model weights
    model_weights = {
        "layer1": np.random.randn(64, 32),
        "layer2": np.random.randn(32, 16)
    }
    
    # Compress model
    compressed_weights = compressor.compress_model(
        model_weights,
        target_size=0.5
    )
    
    # Print results
    for layer_name, weights in compressed_weights.items():
        print(f"{layer_name} compressed shape: {weights.shape}")

if __name__ == "__main__":
    compress_neural_network()
