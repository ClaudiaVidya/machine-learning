import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime
from collections import deque
import json
import time


class CNNEyeDetector:
    """CNN-based eye open/closed detector"""
    
    def __init__(self, model_path="eye_state_detector.h5"):
        """Load pre-trained CNN model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ CNN Model loaded: {model_path}")
        except:
            print(f"⚠ CNN Model not found. Fallback ke Laplacian method")
            self.model = None
    
    def predict_eye_state(self, eye_image):
        """
        Predict if eye is open or closed using CNN
        
        Args:
            eye_image: Eye crop image (BGR)
        
        Returns:
            float: Confidence 0-1 (1=open, 0=closed)
        """
        
        if self.model is None:
            return None
        
        try:
            # Prep image
            eye_resized = cv2.resize(eye_image, (64, 64))
            eye_normalized = eye_resized / 255.0
            eye_batch = np.expand_dims(eye_normalized, axis=0)
            
            # Predict
            prediction = self.model.predict(eye_batch, verbose=0)[0][0]
            
            return float(prediction)
        
        except Exception as e:
            return None


class DrowsinessDetectorCNN:
    """Drowsiness detector with CNN + Rule-based hybrid"""
    
    def __init__(self, use_cnn=True):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # CNN detector
        self.cnn_detector = CNNEyeDetector() if use_cnn else None
        self.use_cnn = use_cnn and self.cnn_detector is not None
        
        # Baseline
        self.baseline = self.load_baseline()
        
        # Thresholds
        self.drowsy_threshold = 65
        self.min_consecutive_frames = 10
        self.max_missing_frames = 20
        self.cnn_eye_open_threshold = 0.5
        
        # Metrics
        self.frame_count = 0
        self.drowsy_count = 0
        self.eyes_detected_count = 0
        self.face_detected_count = 0
        self.consecutive_drowsy_frames = 0
        self.face_missing_frames = 0
        self.face_present = False
        
        # Buffers
        self.drowsy_buffer = deque(maxlen=30)
        self.eye_buffer = deque(maxlen=30)
        self.face_buffer = deque(maxlen=30)
        
        # State
        self.is_drowsy = False
        self.alert_level = "NORMAL"
        self.confirmed_drowsy = False
        self.face_missing_alert = False
        
        # Method tracking
        self.detection_method = "CNN" if self.use_cnn else "LAPLACIAN"
    
    def load_baseline(self):
        """Load YawDD baseline"""
        baseline_path = Path(__file__).parent / "baseline_analysis" / "YawDD_baseline.json"
        
        try:
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline = json.load(f)
                print(f"✓ Baseline loaded")
                return baseline
            else:
                return self._create_default_baseline()
        except:
            return self._create_default_baseline()
    
    def _create_default_baseline(self):
        return {
            'total_videos': 49,
            'overall_drowsiness_rate': 0.44,
            'overall_eye_detection_rate': 52.25,
            'by_condition': {
                'Normal': {'drowsiness_rate': 0.25, 'eye_detection_rate': 38.43},
                'Talking': {'drowsiness_rate': 0.40, 'eye_detection_rate': 42.51},
                'Yawning': {'drowsiness_rate': 0.00, 'eye_detection_rate': 31.92},
                'Unknown': {'drowsiness_rate': 0.59, 'eye_detection_rate': 62.76}
            }
        }
    
    def is_eye_open_laplacian(self, eye_region, threshold=65):
        """Fallback: Laplacian variance method"""
        try:
            if eye_region.size == 0 or eye_region.shape[0] < 5:
                return False
            
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var > threshold
        except:
            return False
    
    def is_eye_open(self, eye_region):
        """
        Detect if eye is open using CNN or Laplacian
        Hybrid approach: CNN if available, fallback to Laplacian
        """
        
        # Try CNN first
        if self.use_cnn and self.cnn_detector.model is not None:
            cnn_pred = self.cnn_detector.predict_eye_state(eye_region)
            if cnn_pred is not None:
                return cnn_pred >= self.cnn_eye_open_threshold
        
        # Fallback to Laplacian
        return self.is_eye_open_laplacian(eye_region, self.drowsy_threshold)
    
    def detect_drowsiness(self, frame):
        """Detect drowsiness dengan CNN/Laplacian hybrid"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        self.face_detected_count += len(faces)
        
        # Face tracking
        if len(faces) == 0:
            self.consecutive_drowsy_frames = 0
            self.face_missing_frames += 1
            self.face_present = False
            self.face_buffer.append(0)
            
            if self.face_missing_frames > self.max_missing_frames:
                self.face_missing_alert = True
            
            return False, 0, 0, 0
        
        # Face present
        self.face_missing_frames = 0
        self.face_missing_alert = False
        self.face_present = True
        self.face_buffer.append(1)
        
        # Eye detection dengan CNN
        drowsy_faces = 0
        eyes_detected_total = 0
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                eyes_detected_total += 1
                
                # Check eye state dengan CNN
                eyes_open = sum(1 for (ex, ey, ew, eh) in eyes 
                              if self.is_eye_open(roi_color[ey:ey+eh, ex:ex+ew]))
                
                # Drowsy jika semua mata tertutup
                if eyes_open == 0 and len(eyes) >= 2:
                    drowsy_faces += 1
        
        self.eyes_detected_count += eyes_detected_total
        
        # Stabilization
        is_currently_drowsy = drowsy_faces > 0
        
        if is_currently_drowsy:
            self.consecutive_drowsy_frames += 1
        else:
            self.consecutive_drowsy_frames = 0
        
        confirmed_drowsy = self.consecutive_drowsy_frames >= self.min_consecutive_frames
        
        if confirmed_drowsy:
            self.drowsy_count += 1
        
        return confirmed_drowsy, len(faces), eyes_detected_total, eyes_detected_total / len(faces) * 100 if len(faces) > 0 else 0
    
    def update_metrics(self, is_drowsy, eye_detection_rate):
        """Update metrics"""
        self.frame_count += 1
        self.drowsy_buffer.append(1 if is_drowsy else 0)
        self.eye_buffer.append(eye_detection_rate)
        
        avg_drowsy_rate = sum(self.drowsy_buffer) / len(self.drowsy_buffer) * 100
        avg_eye_detection = np.mean(self.eye_buffer) if self.eye_buffer else 0
        
        self._update_alert_level(avg_drowsy_rate)
        
        return avg_drowsy_rate, avg_eye_detection
    
    def _update_alert_level(self, drowsy_rate):
        """Update alert level"""
        if self.face_missing_alert:
            self.alert_level = "CRITICAL"
            self.is_drowsy = True
            return
        
        if not self.face_present:
            self.alert_level = "ALERT"
            self.is_drowsy = False
            return
        
        if drowsy_rate < 20:
            self.alert_level = "NORMAL"
            self.is_drowsy = False
        elif drowsy_rate < 50:
            self.alert_level = "ALERT"
            self.is_drowsy = False
        else:
            self.alert_level = "CRITICAL"
            self.is_drowsy = True
    
    def get_alert_color(self):
        colors = {
            "NORMAL": (0, 255, 0),
            "ALERT": (0, 255, 255),
            "CRITICAL": (0, 0, 255)
        }
        return colors.get(self.alert_level, (255, 255, 255))
    
    def get_metrics_summary(self):
        if self.frame_count == 0:
            return {}
        
        return {
            'frame_count': self.frame_count,
            'drowsy_rate': (self.drowsy_count / self.frame_count) * 100,
            'eye_detection_rate': self.eyes_detected_count / self.frame_count * 100 if self.frame_count > 0 else 0,
            'face_detection_rate': self.face_detected_count / self.frame_count * 100 if self.frame_count > 0 else 0,
            'alert_level': self.alert_level,
            'detection_method': self.detection_method
        }


def run_real_time_detection_cnn():
    """Main detection loop dengan CNN"""
    
    print("\n" + "="*70)
    print("REAL-TIME DROWSINESS DETECTION (CNN-POWERED)")
    print("="*70)
    print()
    
    # Initialize with CNN
    detector = DrowsinessDetectorCNN(use_cnn=True)
    
    print(f"🧠 Detection Method: {detector.detection_method}")
    print(f"📊 Baseline Drowsy Rate: {detector.baseline['overall_drowsiness_rate']:.2f}%")
    print("⚡ Performance mode: inferensi setiap 2 frame")
    print()
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open camera")
        return
    
    print("✓ Camera opened successfully")
    print("Press 'q' to quit")
    print()
    
    # Run inference every N frames to reduce CPU load.
    inference_interval = 2
    last_detection = (False, 0, 0, 0)

    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect with frame skipping for better FPS on CPU.
            if detector.frame_count % inference_interval == 0:
                last_detection = detector.detect_drowsiness(frame)

            is_drowsy, face_count, eyes_count, eye_rate = last_detection
            avg_drowsy_rate, avg_eye_detection = detector.update_metrics(is_drowsy, eye_rate)
            
            # Keep native frame size to avoid extra resize overhead.
            display_frame = frame
            
            # Draw metrics
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (10, 10), (400, 150), (0, 0, 0), -1)
            
            color = detector.get_alert_color()
            cv2.putText(display_frame, f"STATUS: {detector.alert_level}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(display_frame, f"Method: {detector.detection_method}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Drowsy: {avg_drowsy_rate:.1f}%", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Frames: {detector.frame_count}", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Drowsiness Detection (CNN)", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Detection stopped")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Frames: {detector.frame_count}")
        print(f"FPS: {detector.frame_count/elapsed:.1f}")
        print(f"Drowsy Rate: {detector.get_metrics_summary()['drowsy_rate']:.2f}%")


if __name__ == "__main__":
    run_real_time_detection_cnn()