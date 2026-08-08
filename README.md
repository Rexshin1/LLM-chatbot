# LLM-chatbot

LLM-chatbot adalah proyek pembelajaran dan engineering untuk membangun **Large Language Model (LLM)** berbasis arsitektur decoder-only Transformer dari nol. Tujuan utama proyek ini adalah menguasai seluruh pipeline LLM, mulai dari preprocessing data hingga inference dan antarmuka chat, tanpa bergantung pada model pretrained atau API eksternal.

---

## 🎯 Tujuan Proyek
- Membangun LLM end-to-end secara mandiri menggunakan Python dan PyTorch.
- Menjelajahi setiap komponen Transformer: tokenizer, arsitektur model, training loop, dan decoding.
- Menerapkan arsitektur yang modular, dapat diuji, dan dapat dikembangkan ulang.
- Menyediakan antarmuka chat lokal untuk demonstrasi hasil inferensi.

---

## 🔧 Teknologi
- Python 3.14
- PyTorch
- NumPy
- pytest
- Vite / React untuk antarmuka web ringan

---

## 📁 Struktur Direktori Utama
- `configs/` : Konfigurasi model dan hyperparameter
- `data/` : Dataset mentah, processed, dan file tokenizer
- `src/` : Implementasi tokenizer, dataset, model, training, dan inference
- `scripts/` : Utility script untuk audit, training, evaluasi, dan generate
- `tests/` : Unit test dengan `pytest`
- `checkpoints/` : Model checkpoint dan bobot hasil training
- `logs/` : Log pelatihan dan hasil evaluasi
- `web/` : Frontend web untuk chat interface

---

## 🚀 Mulai Cepat
1. Aktifkan virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instal dependensi:

```bash
pip install -r requirements.txt
```

3. Verifikasi environment:

```bash
python3 scripts/check_environment.py
```

4. Jalankan test dasar:

```bash
pytest tests/test_environment.py -v
```

---

## 🧠 Komponen Utama
- `src/tokenizer/` : Tokenizer dan vocabulary builder
- `src/dataset/` : Loader data, preprocessing, dan batching
- `src/model/` : Implementasi Transformer decoder, attention, embedding, dan feed-forward
- `src/training/` : Training loop, optimizer, checkpoint, dan utilitas training
- `src/inference/` : Generator teks dan evaluator performa
- `web/` : UI chat untuk interaksi model secara lokal

---

## ⚙️ Contoh Workflow
1. Siapkan data mentah di `data/raw/`
2. Jalankan preprocessing menuju `data/processed/`
3. Latih model menggunakan script training
4. Evaluasi model dengan script evaluasi
5. Jalankan inference/chat melalui antarmuka web atau script generator

---

## ✨ Catatan Penting
- Pastikan file sensitif seperti `.env`, `data/raw/`, `data/processed/`, `checkpoints/`, dan `logs/` tidak ikut dipush ke GitHub.
- Proyek ini dirancang untuk eksplorasi model dari nol, bukan untuk produksi skala besar.
- Untuk pengembangan lanjutan, fokuskan pada peningkatan tokenizer, stabilitas training, dan kualitas generasi teks.

---

## 📌 Cara Menghubungkan dan Menjalankan
Jika kamu ingin menambahkan repo ini ke remote GitHub, jalankan:

```bash
git remote add origin https://github.com/Rexshin1/LLM-chatbot.git
git push -u origin main
```

---

## 📚 Rencana Pengembangan
- Membangun pipeline dataset lengkap
- Mengoptimalkan tokenizer dan vocabulary
- Memperkuat arsitektur Transformer
- Meningkatkan quality text generation
- Menyempurnakan UI chat dan integrasi inference
