import cv2
import numpy as np
from pathlib import Path
import json
import os

class YawDDDataPreparer:
    def __init__(self, yawdd_path="YawDD dataset"):
        self.yawdd_path = Path(yawdd_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    def extract_eyes_from_video(self, video_path, output_dir, max_frames=500):
        """
        Extract eye images dari video
        
        Args:
            video_path: Path ke video file
            output_dir: Output folder untuk eye crops
            max_frames: Max frames to extract
        """
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        eye_count = 0
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📹 Processing: {video_path.name}")
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    eye_crop = roi_color[ey:ey+eh, ex:ex+ew]
                    
                    # Resize to standard size
                    eye_crop = cv2.resize(eye_crop, (64, 64))
                    
                    # Save
                    filename = output_path / f"eye_{frame_count}_{eye_count}.jpg"
                    cv2.imwrite(str(filename), eye_crop)
                    eye_count += 1
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count} frames, extracted {eye_count} eyes")
        
        cap.release()
        print(f"✓ Extracted {eye_count} eyes from {frame_count} frames")
        
        return eye_count
    
    def prepare_all_videos(self, output_base_dir="training_data"):
        """Extract data dari semua YawDD videos"""
        
        output_base = Path(output_base_dir)
        output_base.mkdir(exist_ok=True)
        
        # Find all videos
        video_files = list(self.yawdd_path.rglob("*.avi"))
        video_files += list(self.yawdd_path.rglob("*.mp4"))
        
        total_eyes = 0
        stats = {}
        
        print(f"\n🎬 Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            video_name = video_path.stem
            output_dir = output_base / video_name
            
            try:
                eye_count = self.extract_eyes_from_video(
                    video_path, 
                    output_dir, 
                    max_frames=500
                )
                total_eyes += eye_count
                stats[video_name] = eye_count
            except Exception as e:
                print(f"  ✗ Error: {e}")
                stats[video_name] = 0
        
        # Save stats
        stats_file = output_base / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ EXTRACTION COMPLETE")
        print(f"   Total eyes extracted: {total_eyes}")
        print(f"   Stats saved: {stats_file}")
        
        return total_eyes


if __name__ == "__main__":
    print("\n" + "="*70)
    print("YAWDD DATA PREPARATION")
    print("="*70)
    
    preparer = YawDDDataPreparer(yawdd_path="YawDD dataset")
    
    # Extract eyes dari all videos
    total = preparer.prepare_all_videos(output_base_dir="training_data")
    
    print(f"\n✓ Ready for CNN training: {total} eye images extracted")