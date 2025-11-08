"""
Quick test to verify the FD-CNN-CA model can be built without errors.
This doesn't require actual data - just tests the model architecture.
"""
import numpy as np
import tensorflow as tf
from train_fd_cnn_ca import build_fd_cnn_ca_model, CoordinateAttention, FrequencyDynamicConv

def test_model_build():
    """Test that the model can be built and compiled"""
    print("Testing FD-CNN-CA model architecture...")
    
    # Create dummy input shape (matches config: 128 mel bands, 128 time frames, 1 channel)
    input_shape = (128, 128, 1)
    n_classes = 5  # Example: 5 animal classes
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {n_classes}")
    
    try:
        # Build model
        model = build_fd_cnn_ca_model(input_shape, n_classes)
        print("\n✅ Model built successfully!")
        print(f"Model parameters: {model.count_params():,}")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully!")
        
        # Test forward pass
        dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output probabilities sum: {output.sum():.4f}")
        
        # Test Coordinate Attention layer
        print("\nTesting Coordinate Attention layer...")
        ca_layer = CoordinateAttention(reduction=32)
        test_input = tf.random.normal((2, 64, 64, 128))
        ca_output = ca_layer(test_input)
        print(f"✅ Coordinate Attention test passed!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {ca_output.shape}")
        
        # Test Frequency-Dynamic Conv layer
        print("\nTesting Frequency-Dynamic Conv layer...")
        fdy_layer = FrequencyDynamicConv(filters=64, kernel_sizes=[(3, 3), (5, 5), (7, 7)])
        test_input = tf.random.normal((2, 64, 64, 32))
        fdy_output = fdy_layer(test_input)
        print(f"✅ Frequency-Dynamic Conv test passed!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {fdy_output.shape}")
        
        print("\n" + "="*50)
        print("✅ All tests passed! Model is ready for training.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    success = test_model_build()
    exit(0 if success else 1)

