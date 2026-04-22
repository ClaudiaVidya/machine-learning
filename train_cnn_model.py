import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from cnn_models import create_lightweight_cnn

class CNNEyeTrainer:
    def __init__(self, model=None):
        self.model = model or create_lightweight_cnn(input_shape=(64, 64, 3))
        self.history = None
        
    def load_training_data(self, data_dir="training_data", test_size=0.2):
        """
        Load extracted eye images and create train/test split
        
        Note: For demo, kita buat synthetic data
        Di production, load dari folder training_data/
        """
        
        print("\n📊 Preparing training data...")
        
        # OPTION 1: Load real data dari folder (jika sudah ada)
        data_path = Path(data_dir)
        if data_path.exists():
            print(f"  Loading dari {data_path}...")
            # Load image files
            # This is placeholder - implement file loading here
            pass
        
        # OPTION 2: Create synthetic data untuk demo (training works but not accurate)
        print("  Creating synthetic training data (demo)...")
        
        X_train = []
        y_train = []
        
        # Simulate training data: 1000 eye images
        for i in range(1000):
            # Random eye image (64x64x3)
            eye_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            
            # Label: 50% open, 50% closed
            label = 1.0 if i % 2 == 0 else 0.0
            
            X_train.append(eye_image / 255.0)  # Normalize
            y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, epochs=50, batch_size=32):
        """Train the CNN model"""
        
        print("\n" + "="*70)
        print("TRAINING CNN MODEL")
        print("="*70)
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_training_data()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Train
        print(f"\n🚀 Training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec = self.model.evaluate(X_val, y_val)
        print(f"\n📊 Validation Metrics:")
        print(f"   Accuracy:  {val_acc:.4f}")
        print(f"   Precision: {val_prec:.4f}")
        print(f"   Recall:    {val_rec:.4f}")
        
        return self.history
    
    def save_model(self, path="eye_state_detector.h5"):
        """Save trained model"""
        self.model.save(path)
        print(f"\n✓ Model saved: {path}")
    
    def save_history(self, path="training_history.json"):
        """Save training history"""
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
        }
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"✓ History saved: {path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CNN MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Create trainer
    trainer = CNNEyeTrainer()
    
    # Print model info
    print("\n🧠 Model Architecture:")
    trainer.model.summary()
    
    # Train
    trainer.train(epochs=20, batch_size=32)
    
    # Save
    trainer.save_model("eye_state_detector.h5")
    trainer.save_history("training_history.json")
    
    print("\n✅ Training pipeline complete!")
    print("Next: python real_time_detector_cnn.py")