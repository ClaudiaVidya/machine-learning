import cv2
import math
import sys
import numpy as np

print("Loading Haar Cascade classifiers...")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("❌ Haar Cascade tidak ditemukan!")
    sys.exit(1)

print("✓ Haar Cascade berhasil dimuat")

def get_eye_centroid(eye_region):
    h, w = eye_region.shape
    return (w // 2, h // 2)

def is_eye_open(eye_region, threshold=40):
    """
    Deteksi mata terbuka dengan mengecek variance pixel intensity
    """
    if eye_region is None or eye_region.size == 0:
        return False
    
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold

def open_camera():
    print("🔍 Mencari camera...")
    for camera_index in range(5):
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Camera ditemukan di index {camera_index}")
                    return cap
                cap.release()
        except:
            pass
    return None

cap = open_camera()

if cap is None:
    print("\n❌ CAMERA TIDAK TERDETEKSI!")
    print("\nPastikan:")
    print("  ✓ Camera/Webcam USB sudah terhubung ke laptop")
    print("  ✓ Driver camera sudah terinstall")
    print("  ✓ Camera tidak digunakan aplikasi lain (Zoom, Teams, dll)")
    sys.exit(1)

print("\n" + "="*50)
print("✓ PROGRAM SIAP MENJALANKAN DETEKSI KANTUK!")
print("="*50)
print("Petunjuk:")
print("  • Posisikan wajah di depan camera")
print("  • Program akan deteksi jika mata terpejam")
print("  • Tekan 'q' untuk keluar")
print("  • Tekan 'r' untuk reset counter")
print("="*50 + "\n")

DROWSY_THRESHOLD = 10  
DROWSY_FRAMES = 0
EYE_ASPECT_THRESHOLD = 40 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, fw, fh) = faces[0]

        roi_gray = gray[y:y+fh, x:x+fw]
        roi_color = frame[y:y+fh, x:x+fw]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        left_eye_open = False
        right_eye_open = False

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            
            left_eye = eyes[0]
            right_eye = eyes[1]

            (ex1, ey1, ew1, eh1) = left_eye
            left_roi = roi_color[ey1:ey1+eh1, ex1:ex1+ew1]
            left_eye_open = is_eye_open(left_roi, EYE_ASPECT_THRESHOLD)

            (ex2, ey2, ew2, eh2) = right_eye
            right_roi = roi_color[ey2:ey2+eh2, ex2:ex2+ew2]
            right_eye_open = is_eye_open(right_roi, EYE_ASPECT_THRESHOLD)

            color_l = (0, 255, 0) if left_eye_open else (0, 0, 255)
            color_r = (0, 255, 0) if right_eye_open else (0, 0, 255)
            
            cv2.rectangle(roi_color, (ex1, ey1), (ex1+ew1, ey1+eh1), color_l, 2)
            cv2.rectangle(roi_color, (ex2, ey2), (ex2+ew2, ey2+eh2), color_r, 2)

        both_eyes_closed = not left_eye_open and not right_eye_open and len(eyes) >= 2

        if both_eyes_closed:
            DROWSY_FRAMES += 1
            if DROWSY_FRAMES >= DROWSY_THRESHOLD:
                cv2.putText(frame, "⚠ DROWSINESS DETECTED!", (30, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "MATA TERPEJAM!", (30, 130),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            DROWSY_FRAMES = max(0, DROWSY_FRAMES - 1)
            cv2.putText(frame, "✓ Mata Terbuka", (30, 80),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Eyes detected: {len(eyes)}", (30, h-60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Drowsy Frame: {DROWSY_FRAMES}/{DROWSY_THRESHOLD}",
                  (30, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Face not detected", (30, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        DROWSY_FRAMES = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n✓ Program dihentikan")
        break
    elif key == ord('r'):
        DROWSY_FRAMES = 0
        print("Reset counter")

cap.release()
cv2.destroyAllWindows()
print("✓ Selesai")