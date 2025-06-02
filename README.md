# 🔒 Sistem Autentikasi Wajah dengan Anti-Spoofing 🛡️

Aplikasi autentikasi wajah canggih berbasis Python yang dilengkapi fitur **anti-spoofing** untuk membedakan wajah asli dari upaya penipuan menggunakan foto, video, atau topeng. Solusi ideal untuk sistem absensi, login aplikasi, atau sistem keamanan berbasis pengenalan wajah yang andal.

[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Framework: Tkinter](https://img.shields.io/badge/GUI-Tkinter-orange.svg)](#)
[![AI Models: TFLite & ONNX](https://img.shields.io/badge/AI%20Models-TFLite%20%26%20ONNX-brightgreen.svg)](#models)
[![License: Personal Project](https://img.shields.io/badge/License-Personal%20Use-green.svg)](/LICENSE)

---

## ✨ Fitur Utama

* 📸 **Deteksi Wajah Real-time**: Menggunakan model Haar Cascade klasik dan model TFLite modern untuk akurasi tinggi.
* 🛡️ **Anti-Spoofing Tingkat Lanjut**: Mendeteksi upaya penipuan (foto, video, topeng) secara efektif menggunakan model ONNX.
* 👤 **Pengenalan Wajah Akurat**: Memanfaatkan model MobileFaceNet (TFLite) untuk mengenali identitas pengguna terdaftar.
* ✍️ **Registrasi Pengguna Mudah**: Antarmuka intuitif bagi setiap pengguna untuk mendaftarkan data wajah mereka.
* 🖥️ **Antarmuka Grafis (GUI)**: Dibangun dengan Tkinter, menyediakan pengalaman pengguna yang ramah dan mudah dioperasikan.
* 🗃️ **Manajemen Database Wajah**: Data wajah pengguna disimpan secara lokal dan dapat dikelola dengan mudah.
* ⚙️ **Pengaturan Fleksibel & Logging**: Menu pengaturan untuk kustomisasi (mis., sensitivitas) dan pencatatan aktivitas sistem untuk audit.

---

## 🛠️ Teknologi Inti

Sistem ini dibangun menggunakan kombinasi teknologi berikut:

* **Bahasa Pemrograman**: Python
* **Deteksi Wajah**: Haar Cascade (OpenCV), Model TFLite (mis., `480-float16.tflite`)
* **Anti-Spoofing**: Model ONNX (mis., `AntiSpoofing_bin_128.onnx`)
* **Pengenalan Wajah**: MobileFaceNet TFLite (mis., `MobileFaceNet_9925_9680.tflite`)
* **GUI Framework**: Tkinter
* **Library Pendukung**: OpenCV, TensorFlow Lite, ONNX Runtime, NumPy, PIL, dan lainnya (lihat `requirements.txt`).

---

## 📁 Struktur Folder Proyek

```
face-id/
├── main.py                # Entry point aplikasi
├── face_detector.py       # Modul deteksi wajah
├── anti_spoofing.py       # Modul anti-spoofing
├── face_recognizer.py     # Modul pengenalan wajah
├── utils.py               # Fungsi utilitas
├── models/                # Model AI (TFLite, ONNX, XML)
│   ├── 480-float16.tflite
│   ├── MobileFaceNet_9925_9680.tflite
│   ├── AntiSpoofing_bin_128.onnx
│   └── ...
├── face_database/         # Database wajah pengguna
├── requirements.txt       # Daftar dependensi Python
├── README.md              # Dokumentasi ini
└── ...
```

---

## 🚀 Cara Instalasi & Konfigurasi

Ikuti langkah-langkah berikut untuk menjalankan sistem ini di komputer Anda:

1.  **Clone Repositori Ini:**
    Buka terminal atau PowerShell, lalu jalankan:
    ```powershell
    git clone <repo-url>
    cd face-id
    ```
    *(Ganti `<repo-url>` dengan URL repositori Anda)*

2.  **Buat dan Aktifkan Virtual Environment** (Sangat Direkomendasikan):
    Ini membantu mengisolasi dependensi proyek.
    ```powershell
    python -m venv env
    .\env\Scripts\activate
    ```
    *(Untuk Linux/macOS, gunakan `source env/bin/activate`)*

3.  **Install Semua Dependensi:**
    Pastikan semua library yang dibutuhkan terpasang.
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Siapkan Model AI:** 🧠
    > **PENTING:** Pastikan semua file model AI yang diperlukan sudah tersedia di dalam folder `models/`.
    >
    > Jika belum, unduh model-model berikut dan letakkan di dalam folder `models/`:
    > * `480-float16.tflite` (untuk deteksi wajah)
    > * `MobileFaceNet_9925_9680.tflite` (untuk pengenalan wajah)
    > * `AntiSpoofing_bin_128.onnx` (untuk anti-spoofing)
    > * `haarcascade_frontalface_default.xml` (untuk deteksi wajah klasik OpenCV)

5.  **Jalankan Aplikasi:**
    Setelah semua siap, jalankan script utama:
    ```powershell
    python main.py
    ```

---

## 📖 Cara Penggunaan Aplikasi

Setelah aplikasi berjalan, berikut adalah cara penggunaannya:

1.  🆕 **Registrasi Pengguna Baru:**
    * Klik tombol **"Register"** pada antarmuka utama.
    * Masukkan **Nama Pengguna** Anda.
    * Ikuti instruksi di layar untuk mengambil beberapa sampel gambar wajah Anda dari berbagai sudut.

2.  🔑 **Login / Autentikasi Wajah:**
    * Klik tombol **"Login"**.
    * Posisikan wajah Anda di depan kamera.
    * Sistem akan secara otomatis mendeteksi, melakukan pengecekan anti-spoofing, dan mengenali wajah Anda jika sudah terdaftar.

3.  ⚙️ **Akses Pengaturan (Settings):**
    * Klik tombol **"Settings"** (jika tersedia).
    * Di sini Anda mungkin dapat menyesuaikan parameter seperti sensitivitas deteksi, mode anti-spoofing aktif/nonaktif, atau konfigurasi lainnya.

---

## ⚠️ Troubleshooting Umum

Mengalami masalah? Coba cek beberapa solusi berikut:

* **Kamera tidak terdeteksi**:
    * Pastikan webcam Anda terhubung dengan benar ke komputer.
    * Periksa apakah webcam tidak sedang digunakan oleh aplikasi lain.
    * Coba restart aplikasi atau komputer Anda.
* **Model tidak ditemukan / Error saat load model**:
    * Pastikan semua file model (`.tflite`, `.onnx`, `.xml`) telah diunduh dan ditempatkan dengan benar di dalam folder `models/`.
    * Periksa kembali nama file model, pastikan tidak ada kesalahan ketik.
* **Error terkait dependensi / Library tidak ditemukan**:
    * Pastikan Anda telah mengaktifkan virtual environment (jika menggunakannya).
    * Jalankan ulang `pip install -r requirements.txt` untuk memastikan semua library terinstal dengan versi yang benar.
* **Kegagalan Anti-Spoofing (sering terdeteksi sebagai spoof padahal asli, atau sebaliknya)**:
    * Pastikan kondisi pencahayaan di sekitar Anda cukup baik dan merata.
    * Hindari penggunaan aksesori yang menutupi sebagian besar wajah (kacamata hitam tebal, masker penuh, dll.) saat registrasi atau login.
    * Periksa pengaturan sensitivitas anti-spoofing jika tersedia.

---
