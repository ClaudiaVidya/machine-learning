import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

def create_lightweight_cnn(input_shape=(64, 64, 3), learning_rate=0.001):
    """Create lightweight CNN for eye open/closed classification"""
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25),
        
        # Global pooling + dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', name='fc1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu', name='fc2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output: Binary classification
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='EyeStateDetector_Lightweight')
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    return model


if __name__ == "__main__":
    print("\n🧠 Creating CNN Model...")
    model = create_lightweight_cnn(input_shape=(64, 64, 3))
    print("✅ Model created successfully!")
    model.summary()