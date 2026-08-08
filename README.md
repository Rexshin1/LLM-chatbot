# LLM From Scratch - Pembelajaran Mandiri & Engineering

Repository ini adalah project pembelajaran sekaligus engineering serius untuk membangun **Large Language Model (LLM)** berbasis arsitektur decoder-only Transformer dari nol menggunakan Linux, Python, dan PyTorch.

Tujuan utama project ini adalah memahami setiap bagian detail dari pipeline LLM secara mendalam, mulai dari pemrosesan data mentah hingga antarmuka chat interaktif, tanpa menggunakan model pretrained (seperti model Hugging Face) sebagai pondasi utama dan tanpa menggunakan API eksternal (OpenAI, Gemini, Claude).

---

## 🚀 Prinsip Utama
1. **From Scratch**: Seluruh model utama dibangun secara manual dari representasi tensor dasar.
2. **Modular & Clean Code**: Setiap tahapan dipisah ke dalam modul independen dengan test script untuk mempermudah debugging dan pemahaman.
3. **No API / Pretrained shortcuts**: Implementasi murni berfokus pada matematika dan logika dibalik Transformer.
4. **Keamanan Sistem**: Menghindari penghapusan atau perubahan file sistem Linux di luar ruang lingkup project.

---

## 🛠️ Teknologi yang Digunakan
- **OS**: Linux (Ubuntu 26.04 LTS)
- **Language**: Python 3.14.4
- **Framework**: PyTorch (CPU version `2.13.0+cpu` untuk hardware AMD Ryzen 3 3250U)
- **Library Pendukung**: NumPy, tqdm, pytest

---

## 📂 Struktur Direktori
```text
llm-from-scratch/
├── README.md               # Dokumentasi utama project
├── .gitignore             # File exclusion untuk Git
├── requirements.txt       # Dependency minimal project
├── configs/               # Konfigurasi model dan hyperparameter
├── data/                  # Penyimpanan dataset
│   ├── raw/               # Dataset asli mentah
│   ├── processed/         # Dataset hasil preprocessing
│   └── tokenizer/         # File konfigurasi & vocab tokenizer
├── src/                   # Kode sumber utama
│   ├── tokenizer/         # Implementasi Tokenizer (BPE/WordPiece)
│   ├── dataset/           # Dataset loader dan batching
│   ├── model/             # Arsitektur Transformer
│   ├── training/          # Loop pelatihan & backpropagation
│   └── inference/         # Text generation & decoding
├── scripts/               # Script utilitas (check env, helper)
├── tests/                 # Unit testing menggunakan pytest
├── checkpoints/           # Menyimpan bobot model hasil training
└── logs/                  # Log pelatihan & evaluasi
```

---

## ⚙️ Setup Environment

### 1. Membuat Virtual Environment
Project ini menggunakan Python virtual environment agar dependency tidak berbenturan dengan sistem utama.

```bash
# Membuat virtual environment bernama .venv
python3 -m venv .venv

# Mengaktifkan virtual environment
source .venv/bin/activate
```

### 2. Menginstal Dependency
Instal library dasar yang dibutuhkan menggunakan pip:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Environment Audit
Gunakan script pembantu untuk memverifikasi spesifikasi sistem dan instalasi PyTorch:
```bash
python3 scripts/check_environment.py
```

### 4. Menjalankan Test Suite
Verifikasi kelayakan environment dengan menjalankan test suite menggunakan pytest:
```bash
pytest tests/test_environment.py -v
```

---

## 📍 Project Roadmap

Project ini dikerjakan secara bertahap dan modular melalui 10 tahapan berikut:

- [x] **Phase 1: Environment & Project Foundation**
  Membangun fondasi project, setup virtual environment, audit hardware, konfigurasi Git, dan testing library PyTorch.
- [ ] **Phase 2: Dataset Pipeline**
  Pengumpulan data mentah, text cleaning, preprocessing, dan pipeline loading data.
- [ ] **Phase 3: Tokenizer**
  Membuat tokenizer dari nol (BPE/Byte Pair Encoding), encoding teks ke token ID, dan decoding kembali ke teks.
- [ ] **Phase 4: Transformer Architecture**
  Membangun komponen Transformer Block: Embedding, Positional Encoding, Self-Attention, Multi-Head Attention, Feed-Forward Network, Layer Normalization, Residual Connections, dan Language Modeling Head.
- [ ] **Phase 5: Training Pipeline**
  Implementasi training loop, loss function, backpropagation, optimizer, dan mekanisme penyimpanan model checkpoint.
- [ ] **Phase 6: Evaluation**
  Evaluasi model menggunakan metrik perplexity dan loss pada dataset validasi.
- [ ] **Phase 7: Text Generation**
  Implementasi decoding strategis (Greedy Search, Temperature Scaling, Top-k, Top-p sampling) untuk generate text.
- [ ] **Phase 8: Instruction Tuning**
  Melatih model agar dapat mengikuti instruksi format prompt (fine-tuning terarah).
- [ ] **Phase 9: Inference API**
  Membuat API sederhana untuk melayani pemanggilan model secara lokal dengan latency rendah.
- [ ] **Phase 10: Chat Interface**
  Membangun antarmuka pengguna grafis (Web UI atau terminal chat) interaktif.
