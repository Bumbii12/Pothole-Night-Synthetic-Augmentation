# Deteksi Lubang Jalan Malam Hari dengan YOLOv12 dan Data Sintetis CycleGAN-Turbo

Repositori ini berisi 15 notebook pelatihan dan evaluasi model deteksi lubang jalan (pothole) beserta seluruh hasilnya. Eksperimen ini merupakan bagian deteksi dari skripsi **"Deteksi dan Estimasi Dimensi Pothole pada Skenario Malam Hari: Augmentasi YOLOv12 Menggunakan Data Sintetis CycleGAN-Turbo"** (Informatika, Universitas Bengkulu).

Pertanyaan yang dijawab: apakah menambahkan citra malam sintetis hasil CycleGAN-Turbo ke data latih membuat YOLOv12 lebih baik dalam mendeteksi lubang jalan pada malam hari, dan berapa rasio penambahan yang paling efektif.

**Hasil singkat.** Penambahan data sintetis sebesar 50% dari data latih asli memberi hasil terbaik pada ketiga varian YOLOv12, dengan kenaikan mAP@0.5 sebesar 9,1% hingga 13,9% dibanding baseline. Penambahan 100% juga menaikkan mAP@0.5, tetapi hanya 1,7% hingga 7,7%. Model terpilih adalah **YOLOv12s + sintetis 50%** (mAP@0.5 0,6616; mAP@0.5:0.95 0,3317).

## Isi repositori

```
.
├── notebook/
│   ├── 01_YOLOv12n_Baseline.ipynb
│   ├── ...
│   └── 15_YOLO11s_Sintetis100.ipynb
└── Output Yolo/
    ├── Baseline/          # yolov12{n,s,m}_baseline
    ├── 50 Sintetis/       # yolov12{n,s,m}_sintetis50
    ├── 100 Sintetis/      # yolov12{n,s,m}_sintetis100
    └── SOTA/              # yolov8s_* dan yolo11s_* (model pembanding)
```

Setiap folder hasil (`Output Yolo/<grup>/<nama_run>/`) berisi:

| Isi | Keterangan |
|---|---|
| `train/` | Hasil pelatihan Ultralytics: `weights/best.pt` dan `last.pt`, `results.csv`, `results.png`, `args.yaml`, kurva PR/F1/P/R dan confusion matrix pada data validasi, contoh batch latih dan validasi, log TensorBoard |
| `eval_test/` | Hasil evaluasi `best.pt` pada data uji: kurva PR/F1/P/R, confusion matrix, serta perbandingan label dan prediksi |
| `summary_<nama_run>.json` | Ringkasan metrik uji, epoch terbaik, jumlah parameter, kecepatan inferensi, dan versi lingkungan |

Catatan: pada run `yolov12m_sintetis50`, `yolov12m_sintetis100`, dan `yolov8s_baseline`, isi folder berada satu tingkat lebih dalam (`<nama_run>/<nama_run>/`).

## Daftar notebook

| No | Notebook | Model | Skenario | Dataset Kaggle |
|---|---|---|---|---|
| 01 | `01_YOLOv12n_Baseline` | YOLOv12n | Baseline | `pororozagi/the-basic` |
| 02 | `02_YOLOv12s_Baseline` | YOLOv12s | Baseline | `pororozagi/the-basic` |
| 03 | `03_YOLOv12m_Baseline` | YOLOv12m | Baseline | `pororozagi/the-basic` |
| 04 | `04_YOLOv12n_Sintetis50` | YOLOv12n | Sintetis 50% | `pororozagi/cycle-half` |
| 05 | `05_YOLOv12s_Sintetis50` | YOLOv12s | Sintetis 50% | `pororozagi/cycle-half` |
| 06 | `06_YOLOv12m_Sintetis50` | YOLOv12m | Sintetis 50% | `pororozagi/cycle-half` |
| 07 | `07_YOLOv12n_Sintetis100` | YOLOv12n | Sintetis 100% | `pororozagi/aug-cycle` |
| 08 | `08_YOLOv12s_Sintetis100` | YOLOv12s | Sintetis 100% | `pororozagi/aug-cycle` |
| 09 | `09_YOLOv12m_Sintetis100` | YOLOv12m | Sintetis 100% | `pororozagi/aug-cycle` |
| 10 | `10_YOLOv8s_Baseline` | YOLOv8s | Baseline | `pororozagi/the-basic` |
| 11 | `11_YOLOv8s_Sintetis50` | YOLOv8s | Sintetis 50% | `pororozagi/cycle-half` |
| 12 | `12_YOLOv8s_Sintetis100` | YOLOv8s | Sintetis 100% | `pororozagi/aug-cycle` |
| 13 | `13_YOLO11s_Baseline` | YOLO11s | Baseline | `pororozagi/the-basic` |
| 14 | `14_YOLO11s_Sintetis50` | YOLO11s | Sintetis 50% | `pororozagi/cycle-half` |
| 15 | `15_YOLO11s_Sintetis100` | YOLO11s | Sintetis 100% | `pororozagi/aug-cycle` |

Semua notebook memiliki struktur yang sama. Perbedaannya hanya ada di sel **4. Konfigurasi eksperimen** (`MODEL_WEIGHTS`, `DATASET_SLUG`, `RUN_NAME`, `OUTPUT_GROUP`). Setiap notebook menyimpan output lengkap dan catatan *Insight* setelah setiap tahap.

## Data dan skenario

Dataset sekunder berjumlah 1.000 citra: 800 citra siang (Kaggle 276, Roboflow 524) dan 200 citra malam dari Night Pothole Dataset (NPD). Pembagiannya:

| Split | Isi | Keterangan |
|---|---|---|
| Latih | 700 citra siang | Ditambah citra malam sintetis sesuai skenario, lalu diaugmentasi 3× |
| Validasi | 100 siang + 100 malam | Sama untuk semua skenario |
| Uji | 100 citra malam NPD (166 objek) | Sama untuk semua skenario |

| Skenario | Citra latih sebelum augmentasi | Citra latih setelah augmentasi 3× |
|---|---|---|
| Baseline | 700 | 2.100 |
| Sintetis 50% | 700 + 350 sintetis = 1.050 | 3.150 |
| Sintetis 100% | 700 + 700 sintetis = 1.400 | 4.200 |

Citra sintetis dihasilkan dengan CycleGAN-Turbo (translasi siang ke malam) dan hanya dimasukkan ke data latih, sehingga data validasi dan data uji tetap berisi citra asli.

## Pengaturan pelatihan

Seluruh 15 konfigurasi memakai hyperparameter yang sama, tanpa hyperparameter tuning. Hyperparameter lain mengikuti nilai bawaan Ultralytics, termasuk `seed=0`.

| Parameter | Nilai |
|---|---|
| Bobot awal | Pretrained COCO (`yolov12{n,s,m}.pt`, `yolov8s.pt`, `yolo11s.pt`) |
| Epoch | 50 |
| Ukuran citra | 640 × 640 |
| Batch | 16 |
| Optimizer | SGD (lr0 0,01; momentum 0,937; weight decay 0,0005) |
| Bobot yang dievaluasi | `best.pt` (fitness validasi tertinggi: 0,1 × mAP@0.5 + 0,9 × mAP@0.5:0.95) |

Lingkungan: Google Colab dengan GPU Tesla T4, Python 3.13, PyTorch 2.9.0 (CUDA 12.8), dan Ultralytics 8.3.63 dari fork [`sunsmarterjie/yolov12`](https://github.com/sunsmarterjie/yolov12) pada commit `3bca22b`.

## Cara menjalankan

1. Buka notebook di Google Colab dan pilih runtime **T4 GPU**.
2. Simpan kredensial Kaggle di **Colab Secrets** dengan nama `KAGGLE_USERNAME` dan `KAGGLE_KEY`. Jangan menulis kredensial langsung di notebook.
3. Jalankan bagian **1** dan **2** satu kali. Bagian 2 memasang PyTorch 2.9.0 lalu memulai ulang runtime, jadi pesan *"Your session crashed"* memang wajar.
4. Lanjutkan dari bagian **3. Instal dependensi** sampai selesai. Jangan memakai *Run all* dari atas. Sel pemeriksaan versi akan berhenti dengan pesan kesalahan bila versi lingkungan tidak sesuai.
5. Hasil disimpan ke Google Drive di `MyDrive/Output Yolo/<OUTPUT_GROUP>/<RUN_NAME>/`.

Untuk memakai bobot yang sudah dilatih tanpa melatih ulang:

```python
from ultralytics import YOLO

model = YOLO("Output Yolo/50 Sintetis/yolov12s_sintetis50/train/weights/best.pt")
results = model.predict("citra_jalan.jpg", conf=0.25)
```

## Hasil pada data uji

Evaluasi dilakukan pada 100 citra malam NPD. mAP dihitung pada seluruh rentang confidence, sedangkan Precision dan Recall dilaporkan pada confidence yang menghasilkan F1 maksimum. Angka di bawah diambil dari `summary_<nama_run>.json`.

| Model | Skenario | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Parameter | Inferensi (ms/citra) |
|---|---|---|---|---|---|---|---|
| YOLOv12n | Baseline | 0,7049 | 0,5037 | 0,5744 | 0,2951 | 2,51 juta | 11,7 |
| YOLOv12n | Sintetis 50% | 0,7258 | 0,5899 | **0,6269** | **0,3306** | 2,51 juta | 9,6 |
| YOLOv12n | Sintetis 100% | 0,6132 | 0,5843 | 0,5867 | 0,2976 | 2,51 juta | 12,4 |
| YOLOv12s | Baseline | 0,6908 | 0,5518 | 0,5807 | 0,2673 | 9,07 juta | 16,4 |
| **YOLOv12s** | **Sintetis 50%** | 0,7505 | 0,5843 | **0,6616** | **0,3317** | 9,07 juta | 18,5 |
| YOLOv12s | Sintetis 100% | 0,6794 | 0,6127 | 0,6253 | 0,3157 | 9,07 juta | 17,6 |
| YOLOv12m | Baseline | 0,7549 | 0,5060 | 0,5783 | 0,2988 | 19,58 juta | 63,9 |
| YOLOv12m | Sintetis 50% | 0,7331 | 0,5783 | **0,6357** | **0,3252** | 19,58 juta | 63,9 |
| YOLOv12m | Sintetis 100% | 0,5999 | 0,6053 | 0,5882 | 0,3030 | 19,58 juta | 65,4 |
| YOLOv8s | Baseline | 0,6926 | 0,5566 | 0,5828 | 0,2656 | 11,13 juta | 11,7 |
| YOLOv8s | Sintetis 50% | 0,8166 | 0,5964 | **0,6662** | **0,3274** | 11,13 juta | 12,2 |
| YOLOv8s | Sintetis 100% | 0,6566 | 0,5644 | 0,5954 | 0,2899 | 11,13 juta | 10,5 |
| YOLO11s | Baseline | 0,6011 | 0,6084 | 0,5973 | 0,2912 | 9,41 juta | 12,6 |
| YOLO11s | Sintetis 50% | 0,7535 | 0,5542 | **0,6118** | **0,3166** | 9,41 juta | 12,8 |
| YOLO11s | Sintetis 100% | 0,7804 | 0,5542 | 0,5979 | 0,2971 | 9,41 juta | 11,3 |

Kecepatan inferensi diukur oleh Ultralytics saat evaluasi pada GPU T4 dan dapat sedikit berbeda setiap kali dijalankan.

### Temuan utama

**Rasio 50% paling efektif.** Pada ketiga varian YOLOv12, skenario sintetis 50% menghasilkan mAP@0.5 dan mAP@0.5:0.95 tertinggi. Dibanding baseline, mAP@0.5 naik 9,1% (nano), 13,9% (small), dan 9,9% (medium).

**Data sintetis lebih banyak tidak selalu lebih baik.** Skenario sintetis 100% hanya menaikkan mAP@0.5 sebesar 2,1% (nano), 7,7% (small), dan 1,7% (medium). Pada skenario ini Recall naik, tetapi Precision turun, terutama pada varian nano dan medium.

**Pola yang sama pada arsitektur lain.** Dengan sintetis 50%, mAP@0.5 YOLOv8s naik 14,3% dan YOLO11s naik 2,4%. Skenario sintetis 100% kembali memberi kenaikan yang lebih kecil, yaitu 2,2% pada YOLOv8s dan 0,1% pada YOLO11s.

**Model terpilih: YOLOv12s + sintetis 50%.** mAP@0.5 model ini (0,6616) setara dengan YOLOv8s sintetis 50% (0,6662). YOLOv12s dipilih karena mAP@0.5:0.95-nya lebih tinggi (0,3317 dibanding 0,3274) dan jumlah parameternya lebih sedikit (9,07 juta dibanding 11,13 juta). Bobot model ini ada di `Output Yolo/50 Sintetis/yolov12s_sintetis50/train/weights/best.pt` dan dipakai pada tahap estimasi dimensi lubang dengan Inverse Perspective Mapping (IPM) pada dataset primer malam hari dari Jl. Bencoolen, Bengkulu.

### Kurva model terpilih

Kurva pelatihan dan validasi YOLOv12s + sintetis 50%:

![Kurva pelatihan YOLOv12s sintetis 50%](Output%20Yolo/50%20Sintetis/yolov12s_sintetis50/train/results.png)

Kurva Precision-Recall dan confusion matrix pada data uji:

<p>
  <img src="Output%20Yolo/50%20Sintetis/yolov12s_sintetis50/eval_test/PR_curve.png" width="49%" alt="Kurva PR data uji">
  <img src="Output%20Yolo/50%20Sintetis/yolov12s_sintetis50/eval_test/confusion_matrix.png" width="49%" alt="Confusion matrix data uji">
</p>

## Keterbatasan

- Seluruh model dilatih dengan satu konfigurasi hyperparameter tanpa tuning, sehingga hasilnya belum tentu optimal untuk setiap varian.
- Hanya dua rasio data sintetis yang diuji (50% dan 100%).
- Data uji berisi 100 citra malam dari satu sumber (NPD), sehingga hasilnya belum mencakup seluruh variasi pencahayaan malam di lapangan.
