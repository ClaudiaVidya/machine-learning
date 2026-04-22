import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from collections import deque
import time

class DrowsinessDetector:
    """Real-time drowsiness detector using dataset baseline"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self.baseline = self.load_baseline()
        
        self.drowsy_threshold = 65  # INCREASED - stricter eye closure detection
        self.min_consecutive_frames = 10  # Require 10+ consecutive frames to confirm drowsy
        self.eyes_required_both = True  # Require BOTH eyes closed
        self.min_face_presence_threshold = 5  # Require 5+ consecutive frames with face
        self.max_missing_frames = 20  # Alert if face missing > 20 frames (666ms at 30fps)
        
        self.baseline_drowsy_rate = float(self.baseline.get('overall_drowsiness_rate', 0.44))
        self.baseline_eye_detection = float(self.baseline.get('overall_eye_detection_rate', 52.25))
        
        self.frame_count = 0
        self.drowsy_count = 0
        self.eyes_detected_count = 0
        self.face_detected_count = 0
        self.consecutive_drowsy_frames = 0  # Counter for stabilization
        self.face_missing_frames = 0  # NEW: Track if face disappears
        self.face_present = False  # NEW: Current face status
        
        self.drowsy_buffer = deque(maxlen=30)
        self.eye_buffer = deque(maxlen=30)
        self.face_buffer = deque(maxlen=30)  # NEW: Face presence buffer
        
        self.is_drowsy = False
        self.alert_level = "NORMAL"  # NORMAL, ALERT, CRITICAL
        self.confirmed_drowsy = False  # Only set when threshold met
        self.face_missing_alert = False  # NEW: Face missing alert
        
    def load_baseline(self):
        """Load dataset baseline metrics"""
        baseline_path = Path(__file__).parent / "baseline_analysis" / "YawDD_baseline.json"
        
        try:
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline = json.load(f)
                print(f"✓ Baseline loaded from {baseline_path}")
                return baseline
            else:
                print(f"⚠ Baseline not found at {baseline_path}")
                print("  Creating default baseline...")
                return self._create_default_baseline()
        except Exception as e:
            print(f"✗ Error loading baseline: {e}")
            return self._create_default_baseline()
    
    def _create_default_baseline(self):
        """Default baseline if file not found"""
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
    
    def is_eye_open(self, eye_region, threshold=None):
        """
        Detect if eye is open using Laplacian variance
        IMPROVED: 
        - Higher threshold (65) means stricter - requires clear, SHARP details = OPEN eye
        - Closed/narrow eye has low variance = correctly detected as CLOSED
        - Better handling of edge cases
        """
        if threshold is None:
            threshold = self.drowsy_threshold
        
        try:
            if eye_region.size == 0:
                return False
            
            if eye_region.shape[0] < 5 or eye_region.shape[1] < 5:
                return False
            
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # HIGHER threshold = more OPEN eyes, LOWER variance = CLOSED eyes
            # variance > 65: Eye is clearly OPEN (lots of detail/sharpness)
            # variance ≤ 65: Eye is CLOSED or mostly closed (smooth/blurry)
            is_open = laplacian_var > threshold
            
            return is_open
        except Exception as e:
            return False
    
    def detect_drowsiness(self, frame):
        """
        Improved drowsiness detection:
        - Track face presence (must be visible to assess)
        - Alert if face missing (person out of frame)
        - Only assess drowsiness when face clearly visible
        - Include gesture detection for drowsy posture
        
        Returns: (is_drowsy, face_count, eyes_count, eye_detection_rate)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        self.face_detected_count += len(faces)
        
        # IMPROVED: Track face presence status
        if len(faces) == 0:
            # Face MISSING - cannot assess drowsiness
            self.consecutive_drowsy_frames = 0  # Reset
            self.face_missing_frames += 1  # Track missing
            self.face_present = False
            self.face_buffer.append(0)  # 0 = no face
            
            # If face is missing too long = ALERT!
            if self.face_missing_frames > self.max_missing_frames:
                self.face_missing_alert = True
            
            return False, 0, 0, 0
        
        # Face PRESENT - reset missing counter
        self.face_missing_frames = 0
        self.face_missing_alert = False
        self.face_present = True
        self.face_buffer.append(1)  # 1 = face detected
        
        # Now assess drowsiness only when face is clearly visible
        drowsy_faces = 0
        eyes_detected_total = 0
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in face ROI
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:  # REQUIRE at least 2 eyes detected
                eyes_detected_total += 1
                
                # Count open eyes
                eyes_open = sum(1 for (ex, ey, ew, eh) in eyes 
                              if self.is_eye_open(roi_color[ey:ey+eh, ex:ex+ew]))
                
                # IMPROVED: Mark drowsy if ALL eyes are closed
                # Also check for drowsy gesture (low eye count or closed eyes)
                if eyes_open == 0 and len(eyes) >= 2:
                    drowsy_faces += 1
                elif eyes_open == 1:  # Only 1 eye open could indicate drowsy posture
                    # Check head position - if tilted + only 1 eye = might be drowsy
                    # However be conservative here
                    pass
        
        self.eyes_detected_count += eyes_detected_total
        
        # NEW: Improved drowsiness confirmation logic
        is_currently_drowsy = drowsy_faces > 0
        
        if is_currently_drowsy:
            self.consecutive_drowsy_frames += 1
        else:
            self.consecutive_drowsy_frames = 0  # Reset if not drowsy
        
        # Only COUNT as drowsy if threshold met
        confirmed_drowsy = self.consecutive_drowsy_frames >= self.min_consecutive_frames
        
        if confirmed_drowsy:
            self.drowsy_count += 1
        
        return confirmed_drowsy, len(faces), eyes_detected_total, eyes_detected_total / len(faces) * 100 if len(faces) > 0 else 0
    
    def update_metrics(self, is_drowsy, eye_detection_rate):
        """Update real-time metrics with improved detection"""
        self.frame_count += 1
        
        # Only add to buffer when confirmed (after stabilization)
        self.drowsy_buffer.append(1 if is_drowsy else 0)
        self.eye_buffer.append(eye_detection_rate)
        
        # Calculate smoothed rates - now more stable due to stabilization
        avg_drowsy_rate = sum(self.drowsy_buffer) / len(self.drowsy_buffer) * 100
        avg_eye_detection = np.mean(self.eye_buffer) if self.eye_buffer else 0
        
        # Determine alert level based on stabilized metrics
        self._update_alert_level(avg_drowsy_rate)
        
        return avg_drowsy_rate, avg_eye_detection
    
    def _update_alert_level(self, drowsy_rate):
        """Update alert level based on drowsy rate and face presence"""
        # IMPROVED: Check face presence first
        if self.face_missing_alert:
            # Face missing for too long = CRITICAL alert!
            self.alert_level = "CRITICAL"
            self.is_drowsy = True
            return
        
        if not self.face_present:
            # Face briefly missing = ALERT
            self.alert_level = "ALERT"
            self.is_drowsy = False
            return
        
        # Face is present - assess drowsiness
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
        """Get color for display based on alert level"""
        colors = {
            "NORMAL": (0, 255, 0),      # Green
            "ALERT": (0, 255, 255),      # Yellow
            "CRITICAL": (0, 0, 255)      # Red
        }
        return colors.get(self.alert_level, (255, 255, 255))
    
    def get_metrics_summary(self):
        """Get current metrics summary"""
        if self.frame_count == 0:
            return {}
        
        return {
            'frame_count': self.frame_count,
            'drowsy_rate': (self.drowsy_count / self.frame_count) * 100,
            'eye_detection_rate': self.eyes_detected_count / self.frame_count * 100 if self.frame_count > 0 else 0,
            'face_detection_rate': self.face_detected_count / self.frame_count * 100 if self.frame_count > 0 else 0,
            'alert_level': self.alert_level,
            'baseline_drowsy': self.baseline_drowsy_rate,
            'baseline_eye_detection': self.baseline_eye_detection
        }
    
    def compare_with_baseline(self):
        """Compare current metrics with dataset baseline"""
        metrics = self.get_metrics_summary()
        
        if not metrics:
            return None
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'current': metrics,
            'baseline': {
                'drowsy_rate': self.baseline_drowsy_rate,
                'eye_detection_rate': self.baseline_eye_detection
            },
            'difference': {
                'drowsy_rate': metrics['drowsy_rate'] - self.baseline_drowsy_rate,
                'eye_detection_rate': metrics['eye_detection_rate'] - self.baseline_eye_detection
            },
            'assessment': self._assess_accuracy(metrics)
        }
        
        return comparison
    
    def _assess_accuracy(self, metrics):
        """Assess accuracy based on metrics"""
        assessments = []
        
        # Eye detection assessment
        if metrics['eye_detection_rate'] > 85:
            assessments.append("✓ Eye detection: EXCELLENT")
        elif metrics['eye_detection_rate'] > 70:
            assessments.append("✓ Eye detection: GOOD")
        else:
            assessments.append("✗ Eye detection: POOR (check lighting/positioning)")
        
        # Drowsy detection assessment
        if metrics['drowsy_rate'] > 40:
            assessments.append("✓ Drowsiness: DETECTABLE (high sensitivity)")
        elif metrics['drowsy_rate'] > 20:
            assessments.append("✓ Drowsiness: MODERATE")
        else:
            assessments.append("ℹ Drowsiness: LOW (normal for awake state)")
        
        # Baseline comparison
        drowsy_diff = metrics['drowsy_rate'] - self.baseline_drowsy_rate
        if abs(drowsy_diff) < 20:
            assessments.append("✓ Baseline comparison: REASONABLE")
        else:
            assessments.append("⚠ Baseline comparison: SIGNIFICANT GAP")
        
        return assessments


# ==================== MAIN APPLICATION ====================

def draw_metrics_panel(frame, detector, metrics, smoothed_drowsy, smoothed_eye):
    """Draw metrics panel on frame"""
    height, width = frame.shape[:2]
    
    # Background panel (top-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Alert status (large, top)
    alert_color = detector.get_alert_color()
    cv2.putText(frame, f"STATUS: {detector.alert_level}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, 2)
    
    # Metrics
    y_pos = 75
    metrics_text = [
        f"Drowsy: {smoothed_drowsy:.1f}% (baseline: {detector.baseline_drowsy_rate:.2f}%)",
        f"Eyes: {smoothed_eye:.1f}% (baseline: {detector.baseline_eye_detection:.2f}%)",
        f"Frames: {detector.frame_count}",
        f"Face detect: {metrics['face_detection_rate']:.1f}%"
    ]
    
    for text in metrics_text:
        cv2.putText(frame, text, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 30
    
    return frame


def draw_assessment_panel(frame, detector):
    """Draw assessment panel on frame"""
    metrics = detector.get_metrics_summary()
    assessment = detector._assess_accuracy(metrics)
    
    height, width = frame.shape[:2]
    
    # Background panel (top-right)
    overlay = frame.copy()
    cv2.rectangle(overlay, (width-400, 10), (width-10, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Assessment
    y_pos = 40
    for text in assessment:
        color = (0, 255, 0) if "✓" in text else (0, 255, 255) if "⚠" in text else (0, 0, 255)
        cv2.putText(frame, text, (width-390, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 30
    
    return frame


def draw_face_detection(frame, detector):
    """Draw face and eye detection on frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Face rectangle
        color = detector.get_alert_color()
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Eyes in face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = detector.eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            is_open = detector.is_eye_open(roi_color[ey:ey+eh, ex:ex+ew])
            eye_color = (0, 255, 0) if is_open else (0, 0, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
    
    return frame


def run_real_time_detection():
    """Main real-time detection loop"""
    
    print("\n" + "="*70)
    print("REAL-TIME CAMERA DROWSINESS DETECTION (with Dataset Baseline)")
    print("="*70)
    print()
    
    # Initialize detector
    detector = DrowsinessDetector()
    
    print(f"📊 Dataset Baseline:")
    print(f"   Drowsiness: {detector.baseline_drowsy_rate:.2f}%")
    print(f"   Eye Detection: {detector.baseline_eye_detection:.2f}%")
    print(f"   Total videos analyzed: {detector.baseline.get('total_videos', 49)}")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open camera")
        return
    
    print("✓ Camera opened successfully")
    print()
    print("Press:")
    print("  'q' to quit")
    print("  's' to save metrics snapshot")
    print("  'r' to reset metrics")
    print()
    
    # Metrics snapshots
    metrics_snapshots = []
    start_time = time.time()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to read frame")
                break
            
            # Detect drowsiness
            is_drowsy, face_count, eyes_count, eye_rate = detector.detect_drowsiness(frame)
            
            # Update metrics
            avg_drowsy_rate, avg_eye_detection = detector.update_metrics(is_drowsy, eye_rate)
            
            # Create display frame
            display_frame = frame.copy()
            display_frame = cv2.resize(display_frame, (1280, 720))
            
            # Draw visualizations
            metrics = detector.get_metrics_summary()
            display_frame = draw_metrics_panel(display_frame, detector, metrics, avg_drowsy_rate, avg_eye_detection)
            display_frame = draw_assessment_panel(display_frame, detector)
            display_frame = draw_face_detection(display_frame, detector)
            
            # Display
            cv2.imshow("Real-Time Drowsiness Detection (Dataset Baseline)", display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Stopping detection...")
                break
            
            elif key == ord('s'):
                # Save metrics snapshot
                comparison = detector.compare_with_baseline()
                metrics_snapshots.append(comparison)
                print(f"\n✓ Snapshot saved ({len(metrics_snapshots)})")
                if comparison:
                    print(f"   Drowsy: {comparison['current']['drowsy_rate']:.2f}%")
                    print(f"   Eyes: {comparison['current']['eye_detection_rate']:.2f}%")
            
            elif key == ord('r'):
                # Reset
                detector.frame_count = 0
                detector.drowsy_count = 0
                detector.eyes_detected_count = 0
                detector.face_detected_count = 0
                detector.drowsy_buffer.clear()
                detector.eye_buffer.clear()
                print("\n↻ Metrics reset")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        elapsed_time = time.time() - start_time
        final_comparison = detector.compare_with_baseline()
        
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(f"Duration: {elapsed_time:.1f} seconds")
        print(f"Frames processed: {detector.frame_count}")
        print(f"FPS: {detector.frame_count/elapsed_time:.1f}")
        print()
        
        if final_comparison:
            print("📊 FINAL METRICS:")
            print(f"   Drowsiness: {final_comparison['current']['drowsy_rate']:.2f}% (baseline: {final_comparison['baseline']['drowsy_rate']:.2f}%)")
            print(f"   Eye Detection: {final_comparison['current']['eye_detection_rate']:.2f}% (baseline: {final_comparison['baseline']['eye_detection_rate']:.2f}%)")
            print(f"   Face Detection: {final_comparison['current']['face_detection_rate']:.2f}%")
            print()
            
            print("📈 ASSESSMENT:")
            for assessment in final_comparison['assessment']:
                print(f"   {assessment}")
            print()
            
            # Save final report
            report_dir = Path(__file__).parent / "real_time_analysis"
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"real_time_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(final_comparison, f, indent=2)
            
            print(f"✓ Report saved: {report_path}")
        
        # Save snapshots if any
        if metrics_snapshots:
            snapshots_path = Path(__file__).parent / "real_time_analysis" / f"snapshots_{timestamp}.json"
            with open(snapshots_path, 'w') as f:
                json.dump(metrics_snapshots, f, indent=2)
            print(f"✓ Snapshots saved: {snapshots_path}")
        
        print()


def show_menu():
    """Show main menu"""
    print("\n" + "="*70)
    print("REAL-TIME CAMERA DROWSINESS DETECTION")
    print("="*70)
    print("\nOptions:")
    print("  1. Start real-time detection (webcam)")
    print("  2. Show dataset baseline info")
    print("  3. Check previous reports")
    print("  4. Exit")
    print()
    return input("Select option (1-4): ").strip()


def show_baseline_info(detector):
    """Show baseline information"""
    print("\n" + "="*70)
    print("DATASET BASELINE INFORMATION")
    print("="*70)
    print()
    
    print(f"📊 Overall Metrics (from {detector.baseline.get('total_videos', 49)} videos):")
    print(f"   Drowsiness rate: {detector.baseline_drowsy_rate:.2f}%")
    print(f"   Eye detection rate: {detector.baseline_eye_detection:.2f}%")
    print()
    
    print("📈 By Condition:")
    for condition, metrics in detector.baseline.get('by_condition', {}).items():
        print(f"   {condition.upper()}:")
        print(f"      Drowsiness: {metrics.get('drowsiness_rate', 0):.2f}%")
        print(f"      Eye detection: {metrics.get('eye_detection_rate', 0):.2f}%")
    print()


def show_previous_reports():
    """Show previous reports"""
    report_dir = Path(__file__).parent / "real_time_analysis"
    
    if not report_dir.exists():
        print("\n✗ No reports found yet")
        return
    
    reports = list(report_dir.glob("real_time_*.json"))
    
    if not reports:
        print("\n✗ No reports found yet")
        return
    
    print("\n" + "="*70)
    print("PREVIOUS REPORTS")
    print("="*70)
    print()
    
    for i, report_path in enumerate(sorted(reports, reverse=True)[:5]):
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            print(f"{i+1}. {report_path.name}")
            print(f"   Time: {data['timestamp']}")
            print(f"   Drowsy: {data['current']['drowsy_rate']:.2f}% | Eyes: {data['current']['eye_detection_rate']:.2f}%")
            print()
        except:
            pass


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    try:
        while True:
            choice = show_menu()
            
            if choice == '1':
                run_real_time_detection()
            
            elif choice == '2':
                detector = DrowsinessDetector()
                show_baseline_info(detector)
            
            elif choice == '3':
                show_previous_reports()
            
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("✗ Invalid option")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
