# 🚗 Drowsiness Detection - Video & Camera Analysis

Program untuk mendeteksi drowsiness (kantuk) dari **video dataset YawDD** dan **real-time camera**, serta membandingkan hasilnya.

---

## 📁 Struktur Project

```
Drowsiness_ML Project/
├── drowsiness_detection.py                    # Program original (camera only)
├── drowsiness_video_analyzer.py               # Analisis video + camera
├── compare_video_camera.py                    # Bandingkan video dengan camera
├── real_person_test.py                        # Test dengan orang asli (simple)
├── real_person_comprehensive_test.py          # Test dengan orang asli (complete)
├── README.md                                  # File ini
├── TESTING_GUIDE.md                           # Guide testing dengan orang asli
├── YawDD dataset/
│   ├── Dash/
│   │   ├── Female/                            # Video wanita driver dashboard
│   │   └── Male/                              # Video pria driver dashboard
│   └── Mirror/
│       └── Female_mirror/                     # Video wanita di cermin
├── analysis_output/                           # Output dari video_analyzer
├── analysis_reports/                          # Report dari compare_video_camera
├── real_person_recordings/                    # Video recording dari orang asli
└── real_person_analysis/                      # Analysis report dari orang asli
```

---

## 🎬 Dataset YawDD

Dataset berisi video orang-orang dalam kondisi driving dengan beberapa kategori:

- **Dash**: Video dari perspektif dashboard (driver view)
  - Female: 6 video
  - Male: 8 video
- **Mirror**: Video dari perspektif cermin
  - Female_mirror: Video wanita

Setiap video menampilkan orang dalam berbagai kondisi:
- Dengan/tanpa kacamata
- Drowsy (mengantuk) dan alert (terjaga)

---

## 🚀 Cara Menggunakan

### 1. **Program Video Analyzer** (`drowsiness_video_analyzer.py`)

Menganalisis video dari dataset YawDD dan generate laporan statistik.

```bash
python drowsiness_video_analyzer.py
```

**Menu:**
- **Opsi 1**: Analisis satu video dengan preview
- **Opsi 2**: Analisis semua video (generate laporan JSON)
- **Opsi 3**: Real-time detection dari camera
- **Opsi 4**: Keluar

**Output:**
- Video dengan annotasi drowsiness detection
- File JSON berisi statistik lengkap

---

### 3. **Real Person Testing** (`real_person_test.py`)

**Test model dengan orang asli dan bandingkan dengan dataset baseline.**

```bash
python real_person_test.py
```

**Fitur:**
- Analisis semua video YawDD dataset → baseline
- Rekam Anda selama 60 detik (30s terjaga + 30s mengantuk)
- Bandingkan dengan baseline dataset
- Lihat apakah model generalize well ke kondisi real

**Contoh Output:**
```
👤 REAL PERSON (Anda):
  • Drowsiness rate: 15.50%
  • Eye detection rate: 94.20%

📈 PERBANDINGAN DENGAN DATASET:
  • Dataset: 25.30% drowsiness
  • Anda: 15.50% drowsiness
  • Selisih: -9.80%
  ✓ MIRIP - Model bekerja konsisten
```

---

### 4. **Comprehensive Real Person Testing** (`real_person_comprehensive_test.py`)

**Test model dengan berbagai kondisi dan analisis detail.**

```bash
python real_person_comprehensive_test.py
```

**Test yang Tersedia:**
1. **Awake Test** - Tetap terjaga, mata terbuka
2. **Drowsy Test** - Simulasi mengantuk, pejam mata
3. **Normal Blink** - Natural blinking tanpa mengantuk
4. **Glasses Test** - Dengan kacamata

**Full Test Suite:**
- Jalankan semua 4 test berturut-turut (~2-3 menit)
- Otomatis bandingkan hasil
- Analisis impact kacamata, lighting, dll

**Contoh Comparison Output:**
```
📊 PERBANDINGAN SEMUA KONDISI
Condition        Drowsy %    Eye Detect %    Status
awake            5.20%       95.30%          ✓ Good
drowsy          62.50%       92.10%          ✓ Excellent
normal_blink    18.30%       96.50%          ✓ Good  
with_glasses     8.70%       89.20%          ⚠️ Slightly lower

✓ Model EXCELLENT - generalize well ke kondisi real
```

**Contoh Output Analisis:**
```
📊 HASIL ANALISIS VIDEO:
  • Total frames: 1500
  • Drowsy frames: 450
  • Drowsiness rate: 30.00%
  • Eye detection rate: 95.50%
```

---

### 2. **Program Comparison** (`compare_video_camera.py`)

**Bandingkan hasil deteksi dari video dataset dengan real-time camera Anda.**

```bash
python compare_video_camera.py
```

**Menu:**
- **Opsi 1**: Bandingkan satu video dengan camera (30 detik)
- **Opsi 2**: Analisis SEMUA video + bandingkan dengan camera
- **Opsi 3**: Hanya analisis satu video (tanpa camera)
- **Opsi 4**: Keluar

**Contoh Perbandingan:**
```
📊 HASIL PERBANDINGAN VIDEO vs CAMERA

🎥 VIDEO: 1-FemaleNoGlasses
  • Drowsiness rate: 25.50%
  • Eye detection rate: 96.20%

📷 CAMERA (Real-time)
  • Drowsiness rate: 22.30%
  • Eye detection rate: 94.80%

📈 ANALISIS:
  • Perbedaan drowsiness rate: -3.20%
  ✓ Kedua source memiliki pola drowsiness yang MIRIP
```

---

## 🔍 Cara Kerja Deteksi

### **Deteksi Mata Terpejam:**

1. **Face Detection**: Menggunakan Haar Cascade untuk mendeteksi wajah
2. **Eye Detection**: Menggunakan Haar Cascade untuk mendeteksi mata
3. **Eye Openness Estimation**: Menggunakan Laplacian variance untuk mengukur variance pixel intensity di area mata
   - Mata terbuka = variance tinggi
   - Mata terpejam = variance rendah

### **Threshold:**
- Eye openness variance threshold: **40** (dapat disesuaikan)
- Drowsy frames threshold: **10 frame konsekutif** sebelum alert

---

## 📊 Output & Laporan

### **Analisis Video Menghasilkan:**
- `analysis_output/` folder berisi:
  - Video hasil analisis (dengan annotasi)
  - File JSON statistik

### **Perbandingan Generate:**
- `analysis_reports/` folder berisi:
  - `comparison_YYYYMMDD_HHMMSS.json` - Laporan perbandingan detail
  - `summary.json` - Summary dari semua video

### **Format JSON Output:**
```json
{
  "timestamp": "2024-03-10T...",
  "video": {
    "video_name": "1-FemaleNoGlasses",
    "total_frames": 1500,
    "drowsy_frames": 450,
    "drowsiness_rate": 30.0,
    "eye_detection_rate": 95.5
  },
  "camera": {
    "drowsy_frames": 25,
    "drowsiness_rate": 22.3,
    "eye_detection_rate": 94.8
  },
  "comparison": {
    "drowsiness_rate_diff": -7.7
  }
}
```

---

## ⚙️ Konfigurasi & Customization

### **Mengubah Threshold Sensitivitas:**

Edit di file Python:
```python
EYE_ASPECT_THRESHOLD = 40      # Semakin kecil = lebih sensitif terhadap mata terpejam
DROWSY_THRESHOLD = 10          # Frame konsekutif sebelum alert
```

### **Menambah Durasi Camera Recording:**
```bash
python compare_video_camera.py
# Saat diminta: Durasi camera (detik, default 30): 60
```

---

## 🎯 Use Cases

1. **Driver Safety Testing**: 
   - Test akurasi dengan video dataset terlebih dahulu
   - Bandingkan hasil dengan real-time detection
   - Verifikasi apakah sistem bekerja di kondisi real

2. **Model Validation**:
   - Gunakan YawDD dataset sebagai benchmark
   - Lihat berapa % accuracy di berbagai kondisi (dengan/tanpa kacamata)
   - Identifikasi weakness dalam deteksi

3. **Performance Comparison**:
   - Bandingkan deteksi kondisi dataset vs kondisi real Anda
   - Lihat apakah ada perbedaan significant

---

## 💡 Tips

- **Pencahayaan**: Pastikan pencahayaan bagus untuk deteksi mata lebih akurat
- **Positioning**: Posisikan wajah langsung ke camera, bukan dari samping
- **Durasi**: Untuk testing, gunakan 20-30 detik per video
- **Multiple Tests**: Lakukan beberapa test dengan kondisi berbeda untuk hasil lebih reliable

---

## 🔧 Troubleshooting

### **"Camera tidak ditemukan"**
- Pastikan camera/webcam sudah di-connect
- Driver camera sudah installed
- Camera tidak digunakan aplikasi lain (Zoom, Teams, dll)

### **"Video tidak bisa dibuka"**
- Pastikan format video adalah .avi atau .mp4
- Video file tidak corrupt

### **Deteksi tidak akurat**
- Coba sesuaikan threshold dengan edit file Python
- Pastikan pencahayaan cukup
- Posisikan wajah lebih mendekati camera

---

## 📝 License

Data YawDD Dataset - Diambil dari public source

---

**Dibuat untuk: Vehicle Driver Drowsiness Detection**  
**Teknologi: OpenCV + Haar Cascade Classifiers**

