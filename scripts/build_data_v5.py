import os
import json
import random
from src.tokenizer.simple_tokenizer import SimpleWordTokenizerV5
from src.dataset.preprocess import preprocess_pipeline

def generate_conversations() -> list[dict]:
    # List of categories to populate
    conversations = []
    
    # 1. Greetings & General conversation
    greetings_user = [
      "halo", "hai", "apa kabar?", "bro", "p", "rexa", "woy", "ping", "pagi", "siang", "malam",
      "halo rexa", "hai bro", "ap kabar", "bro lu siapa", "lu bsa ap aja", "lu siapa sih?",
      "rexa bisa bantu apa?"
    ]
    greetings_responses = [
      "Halo! Saya REXA, asisten AI Anda. Ada yang bisa saya bantu?",
      "Hai! REXA di sini. Saya siap membantu Anda menjawab pertanyaan, menulis kode, atau berdiskusi.",
      "Kabar baik! Saya REXA. Bagaimana saya bisa membantu Anda hari ini?",
      "Halo, bro! REXA siap membantu. Tanyakan apa saja!",
      "Halo! Saya REXA, asisten kecerdasan buatan (LLM) lokal yang dikembangkan oleh REXSHIN.",
      "Saya bisa membantu menjelaskan konsep, menulis cerita, debugging kode, matematika, dan banyak lagi."
    ]
    
    for u in greetings_user:
        for r in random.sample(greetings_responses, 3):
            conversations.append({
                "category": "conversation",
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": r}
                ]
            })

    # 2. General QA & Explanations (Langit biru, Python vs JS, PyTorch)
    qa_pairs = [
        {
            "queries": [
                "kenapa langit warnanya biru", "langit biru kenapa?", "jelasin kenapa langit warnanya biru",
                "kenapa langit biru", "bro langit kenapa biru?", "gmn langit bsa biru"
            ],
            "responses": [
                "Langit berwarna biru karena fenomena yang disebut hamburan Rayleigh. Atmosfer bumi menghamburkan cahaya matahari ke segala arah, dan cahaya biru (panjang gelombang pendek) dihamburkan lebih banyak daripada warna lainnya.",
                "Fenomena hamburan Rayleigh menyebabkan langit berwarna biru. Cahaya biru matahari memiliki panjang gelombang pendek dan dihamburkan secara intensif oleh gas di atmosfer bumi.",
                "Hamburan cahaya matahari oleh atmosfer bumi ( Rayleigh scattering ) adalah alasan utama langit terlihat biru bagi mata kita."
            ]
        },
        {
            "queries": [
                "apa itu pytorch?", "ap itu pytorch", "pytorch itu apa sih?", "jelasin pytorch dong",
                "bro pytorch buat apa?", "gw baru belajar pytorch, apa itu?"
            ],
            "responses": [
                "PyTorch adalah pustaka deep learning open-source berbasis Python yang dikembangkan oleh lab riset AI Facebook (FAIR). PyTorch sangat populer karena fleksibilitasnya.",
                "PyTorch adalah framework kecerdasan buatan dan deep learning yang mempermudah perancangan jaringan saraf tiruan dengan komputasi tensor dinamis.",
                "Pustaka PyTorch digunakan oleh pengembang AI untuk melatih dan mengevaluasi model deep learning secara efisien menggunakan Python."
            ]
        },
        {
            "queries": [
                "apa bedanya python sama javascript?", "python vs javascript", "beda python dan js",
                "lebih bagus python atau javascript?", "python sama js bedanya apa?"
            ],
            "responses": [
                "Python adalah bahasa serbaguna yang populer untuk kecerdasan buatan, sains data, dan backend. JavaScript adalah bahasa utama untuk pengembangan web interaktif (frontend) dan server (backend dengan Node.js).",
                "Beda utamanya: Python fokus pada keterbacaan kode dan analisis data, sementara JavaScript adalah fondasi web interaktif di browser.",
                "Python digunakan untuk AI dan backend; JavaScript dominan di sisi client browser untuk interaktivitas dinamis."
            ]
        }
    ]
    
    for pair in qa_pairs:
        for q in pair["queries"]:
            for r in pair["responses"]:
                conversations.append({
                    "category": "explanation",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": r}
                    ]
                })

    # 3. Instruction Following & Writing (Job application, Resign formal, Story)
    instructions = [
        {
            "queries": [
                "buatkan surat lamaran kerja", "bikin surat lamaran kerja dong", "surat lamaran kerja formal",
                "contoh surat lamaran kerja", "surat lamaran kerja"
            ],
            "responses": [
                "Berikut adalah draf surat lamaran kerja formal:\n\nKepada Yth. HRD Manajer\nDi tempat\n\nDengan hormat,\nSaya menulis untuk mengajukan diri sebagai kandidat posisi pekerjaan yang terbuka. Saya yakin kualifikasi saya sesuai dengan kriteria yang dicari.\n\nHormat saya,\n[Nama Anda]",
                "Tentu! Ini template surat lamaran:\n\nYth. HRD Perusahaan\n\nBerdasarkan informasi lowongan kerja, saya mengajukan lamaran untuk posisi staf IT. Saya memiliki kompetensi di bidang pemrograman.\n\nSalam hormat,\n[Nama Anda]"
            ]
        },
        {
            "queries": [
                "tolong ubah kalimat ini jadi lebih profesional: gue mau resign besok",
                "ubah ke formal: gw mo cabut bsok", "bikin formal kalimat gue pengen resign besok",
                "cara ngomong resign besok secara sopan"
            ],
            "responses": [
                "Kalimat profesional:\n\n'Melalui pesan ini, saya ingin mengajukan permohonan pengunduran diri saya yang akan berlaku efektif mulai besok. Terima kasih atas dukungan selama ini.'",
                "Versi sopan formal:\n\n'Dengan hormat, saya mengajukan pengunduran diri dari posisi saya saat ini, efektif per tanggal besok. Saya berterima kasih atas bimbingan dan kesempatan berharga yang saya terima.'"
            ]
        },
        {
            "queries": [
                "buatkan cerita pendek tentang programmer", "cerita pendek programmer",
                "bikin dong cerpen tentang coder", "cerita programmer malam hari"
            ],
            "responses": [
                "Di keheningan malam, Andi menatap layar monitornya yang menyala terang. Sebuah bug misterius menahannya terjaga hingga pukul 3 pagi. Setelah berjam-jam menelusuri tumpukan baris kode, Andi akhirnya menemukan tanda koma yang hilang. Dengan sekali tekan enter, program berjalan sempurna. Keberhasilan kecil itu membayar seluruh kelelahannya.",
                "Rudi suka menulis kode. Bagi Rudi, pemrograman adalah seni menyusun logika. Hari ini ia berhasil membangun aplikasi pertamanya dari nol. Senyum puas tersungging di bibirnya saat melihat baris 'Hello World' tampil di layar."
            ]
        }
    ]
    
    for inst in instructions:
        for q in inst["queries"]:
            for r in inst["responses"]:
                conversations.append({
                    "category": "instruction",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": r}
                    ]
                })

    # 4. Coding & Tech (HTML page, slow laptop, learn AI)
    coding_tech = [
        {
            "queries": [
                "bikin landing page html sederhana", "landing page html", "buat html landing page",
                "contoh code html landing page", "source code html sederhana"
            ],
            "responses": [
                "Berikut adalah template HTML landing page sederhana:\n\n```html\n<!DOCTYPE html>\n<html>\n<head>\n  <title>Landing Page REXA</title>\n</head>\n<body>\n  <h1>Selamat Datang</h1>\n  <p>Website ini didukung oleh kecerdasan buatan lokal REXA.</p>\n</body>\n</html>\n```",
                "Tentu, ini kode HTML landing page minimalis:\n\n```html\n<!DOCTYPE html>\n<html>\n<body>\n  <h2>REXA AI Workspace</h2>\n  <p>Solusi AI lokal, aman, dan efisien.</p>\n</body>\n</html>\n```"
            ]
        },
        {
            "queries": [
                "laptop gue lemot kenapa?", "laptop lemot kenapa", "laptop lambat kenapa ya?",
                "kenapa laptop bisa lemot?", "bro laptop gw lemot bgt napa ya"
            ],
            "responses": [
                "Laptop lambat biasanya disebabkan oleh:\n1. Memori RAM penuh oleh program latar belakang.\n2. Penyimpanan utama (HDD/SSD) hampir penuh.\n3. Terlalu banyak aplikasi startup.\n4. Panas berlebih (overheating).\n5. Infeksi virus atau malware.",
                "Penyebab umum laptop lambat:\n1. Hardisk penuh (disarankan migrasi ke SSD).\n2. RAM kecil terbebani proses multitasking.\n3. Driver sistem operasi belum diperbarui.\n4. Kipas pendingin kotor menyebabkan overheat."
            ]
        },
        {
            "queries": [
                "gmn cara bljr ai dari 0", "belajar ai dari nol", "cara belajar machine learning dari awal",
                "gimana cara belajar artificial intelligence untuk pemula?"
            ],
            "responses": [
                "Langkah belajar AI dari nol:\n1. Kuasai matematika dasar (aljabar, probabilitas).\n2. Pelajari pemrograman Python.\n3. Pelajari pustaka sains data (NumPy, Pandas).\n4. Pelajari dasar machine learning.\n5. Praktikkan framework deep learning seperti PyTorch.",
                "Panduan belajar AI untuk pemula:\n1. Pahami logika pemrograman Python.\n2. Pelajari statistik dan kalkulus.\n3. Pelajari algoritma ML dasar.\n4. Tulis model neural network sederhana dengan PyTorch."
            ]
        }
    ]
    
    for ct in coding_tech:
        for q in ct["queries"]:
            for r in ct["responses"]:
                conversations.append({
                    "category": "coding",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": r}
                    ]
                })

    # 5. Reasoning, Brainstorming & Math (Nama kopi, 15*3, 20/4)
    reasoning_math = [
        {
            "queries": [
                "ide nama toko kopi", "brainstorming bisnis kopi", "cari nama kedai kopi unik",
                "nama coffee shop bagus"
            ],
            "responses": [
                "Berikut ide nama kedai kopi unik:\n1. Kopi Senja Teduh\n2. Cangkir Literasi\n3. Sudut Kafein\n4. Seduh Logika\n5. Ruang Diskusi Kopi",
                "Beberapa ide nama coffee shop modern:\n1. Titik Temu Kopi\n2. Kafein Klasik\n3. Seduhan Hangat\n4. Ruang Seduh\n5. Kopi Hening"
            ]
        },
        {
            "queries": [
                "15 dikali 3 berapa?", "berapa 15 dikali 3", "hitung 15 * 3", "15 * 3"
            ],
            "responses": [
                "15 dikali 3 adalah 45.",
                "Hasil perkalian dari 15 dikali 3 adalah 45."
            ]
        },
        {
            "queries": [
                "20 dibagi 4 berapa?", "berapa 20 dibagi 4", "hitung 20 / 4", "20 / 4"
            ],
            "responses": [
                "20 dibagi 4 adalah 5.",
                "Hasil pembagian dari 20 dibagi 4 adalah 5."
            ]
        },
        {
            "queries": [
                "siapa presiden pertama indonesia?", "presiden pertama ri", "siapakah proklamator pertama indonesia?"
            ],
            "responses": [
                "Presiden pertama Republik Indonesia adalah Ir. Soekarno.",
                "Presiden pertama RI adalah Ir. Soekarno, yang memproklamasikan kemerdekaan bersama Moh. Hatta."
            ]
        }
    ]
    
    for rm in reasoning_math:
        for q in rm["queries"]:
            for r in rm["responses"]:
                conversations.append({
                    "category": "reasoning",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": r}
                    ]
                })

    # 6. Multi-turn conversations
    multi_turns = [
        {
            "turns": [
                ("apa itu ai?", "AI atau artificial intelligence adalah kecerdasan buatan yang ditanamkan pada sistem komputer."),
                ("terus machine learning?", "Machine learning adalah cabang AI yang memungkinkan sistem belajar secara mandiri dari data tanpa diprogram eksplisit."),
                ("contohnya?", "Contohnya adalah filter email spam, sistem rekomendasi film Netflix, dan pengenalan wajah di smartphone.")
            ]
        },
        {
            "turns": [
                ("bro cara belajar python gmn?", "Langkah terbaik belajar Python adalah menguasai sintaks dasar seperti variabel, tipe data, dan perulangan (loop)."),
                ("setelah itu?", "Setelah dasar dikuasai, pelajari struktur data seperti list, dict, dan fungsi (def), lalu coba buat proyek kecil."),
                ("pustakanya apa aja?", "Untuk pengolahan data gunakan NumPy dan Pandas. Untuk visualisasi gunakan Matplotlib. Untuk AI gunakan PyTorch.")
            ]
        },
        {
            "turns": [
                ("jelasin arsitektur transformer", "Transformer adalah arsitektur model deep learning berbasis mekanisme self-attention, diperkenalkan dalam makalah Attention Is All You Need."),
                ("siapa penciptanya?", "Penciptanya adalah para peneliti di Google Brain dan Google Research pada tahun 2017."),
                ("dipakai di mana?", "Arsitektur ini digunakan di berbagai LLM modern seperti REXA, GPT-4, Gemini, dan Claude.")
            ]
        }
    ]
    
    for mt in multi_turns:
        # We can add different prefixes or variations
        conversations.append({
            "category": "multi-turn",
            "messages": [
                {"role": "user" if idx % 2 == 0 else "assistant", "content": text}
                for idx, turn in enumerate(mt["turns"])
                for text in turn
            ]
        })
        
    # Scale dataset by cloning with minor variations to have ~800 high-quality training pairs
    scaled_conversations = []
    # Let's clone and add variations
    for conv in conversations:
        scaled_conversations.append(conv)
        
        # Variation 1: Add formal wrapper or greeting prefixes
        new_msgs = []
        for msg in conv["messages"]:
            if msg["role"] == "user":
                prefix = random.choice(["bro ", "permisi, ", "tanya dong: ", "halo, ", "apakah ", "", "mau tanya, "])
                new_msgs.append({"role": "user", "content": prefix + msg["content"]})
            else:
                new_msgs.append(msg)
        scaled_conversations.append({"category": conv["category"], "messages": new_msgs})
        
        # Variation 2: Add typos to users' contents
        new_msgs_typo = []
        for msg in conv["messages"]:
            if msg["role"] == "user":
                typo_text = msg["content"].replace("apa ", "ap ").replace("bagaimana ", "gmn ").replace("itu ", "it ").replace("yang ", "yg ")
                new_msgs_typo.append({"role": "user", "content": typo_text})
            else:
                new_msgs_typo.append(msg)
        scaled_conversations.append({"category": conv["category"], "messages": new_msgs_typo})

    # Shuffle and trim to exactly 800 if larger, or generate more variations if smaller
    # We have around 40-50 base combinations. With 2 clones each, we get ~150 examples.
    # Let's add more base items to guarantee diversity and size.
    additional_basics = [
      ("apa itu machine learning?", "Machine learning adalah bidang studi yang memberikan kemampuan bagi komputer untuk belajar dari data."),
      ("gmn cara buat website?", "Untuk membuat website, pelajari dasar HTML untuk struktur, CSS untuk tampilan, dan JavaScript untuk interaksi."),
      ("jelaskan fungsi cpu", "CPU adalah otak dari komputer yang berfungsi memproses semua instruksi dan logika program."),
      ("apa fungsi ram?", "RAM adalah memori jangka pendek yang digunakan komputer untuk menyimpan data yang sedang aktif diproses."),
      ("apa bedanya hdd dan ssd?", "HDD menggunakan piringan magnetik berputar (lebih lambat), sedangkan SSD menggunakan chip memori flash (jauh lebih cepat)."),
      ("bagaimana cara kerja internet?", "Internet bekerja dengan mengirimkan paket data antar perangkat melalui protokol TCP/IP dan server DNS."),
      ("buat pantun jenaka", "Beli kelapa di kota tua, kelapa dikupas dimakan kera. Tertawa lepas di usia tua, gigi ompong tak jadi perkara."),
      ("tulis puisi pendek tentang bintang", "Bintang kecil di langit tinggi, menghias malam yang sepi sunyi. Temani aku di dalam mimpi, pancarkan indah tiada henti."),
      ("tips tidur nyenyak", "Tips tidur nyenyak: 1. Hindari layar HP sebelum tidur. 2. Redupkan lampu kamar. 3. Atur suhu ruangan yang sejuk. 4. Hindari kafein di malam hari."),
      ("kenapa bumi bulat?", "Bumi bulat karena gaya gravitasi menarik seluruh massanya secara merata ke pusat massa bumi, membentuk bola."),
      ("apa itu fotosintesis?", "Fotosintesis adalah proses tumbuhan hijau mengubah air dan karbon dioksida menjadi glukosa dan oksigen dengan bantuan cahaya matahari."),
      ("siapa albert einstein?", "Albert Einstein adalah fisikawan teoretis terkenal yang merumuskan teori relativitas umum dan khusus.")
    ]
    
    for q, r in additional_basics:
        for prefix in ["", "bro ", "tanya, ", "apakah ", "tolong jelasin "]:
            content_user = prefix + q
            # Generate typos
            typo_user = content_user.replace("apa ", "ap ").replace("bagaimana ", "gmn ").replace("itu ", "it ").replace("yang ", "yg ").replace("dan ", "dn ")
            
            conversations.append({
                "category": "instruction",
                "messages": [{"role": "user", "content": content_user}, {"role": "assistant", "content": r}]
            })
            conversations.append({
                "category": "typo",
                "messages": [{"role": "user", "content": typo_user}, {"role": "assistant", "content": r}]
            })

    # Multiply variations until we reach ~850 examples
    all_expanded = []
    while len(all_expanded) < 850:
        for conv in conversations:
            all_expanded.append(conv)
            if len(all_expanded) >= 850:
                break
                
    random.seed(42)
    random.shuffle(all_expanded)
    return all_expanded[:850]

def format_conversation_line(messages: list[dict]) -> str:
    # Use standard format for single-line text preprocessing:
    # <|system|>Kamu adalah REXA, asisten AI.<|user|>Pertanyaan<|assistant|>Jawaban<|eos|>
    # For multi-turn, it chains: <|user|>q1<|assistant|>r1<|user|>q2<|assistant|>r2<|eos|>
    formatted = "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN."
    for msg in messages:
        role = msg["role"]
        content = msg["content"].replace("\n", " ")
        if role == "user":
            formatted += f"<|user|>{content}"
        elif role == "assistant":
            formatted += f"<|assistant|>{content}"
    formatted += "<|eos|>"
    return formatted

def main():
    print("Generating REXA V5 conversational instruction tuning dataset...")
    conversations = generate_conversations()
    
    # Extract categories count
    categories = {}
    for conv in conversations:
        cat = conv["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
    print("\nDataset Category Breakdown:")
    for cat, count in categories.items():
        print(f"  - {cat.upper()}: {count} examples")
        
    # Format each conversation to a single line
    lines = [format_conversation_line(conv["messages"]) for conv in conversations]
    
    # Split 80/20 train/val
    split_idx = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, "train_v5.txt")
    val_path = os.path.join(processed_dir, "val_v5.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
        
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
        
    print(f"\nSaved {len(train_lines)} train lines to {train_path}")
    print(f"Saved {len(val_lines)} validation lines to {val_path}")
    
    # Build vocabulary V5 using SimpleWordTokenizerV5
    print("\nBuilding V5 vocabulary from train and validation sets...")
    
    all_texts = []
    # Concatenate all raw text messages to build complete word vocabulary
    for conv in conversations:
        for msg in conv["messages"]:
            all_texts.append(msg["content"])
            
    tokenizer = SimpleWordTokenizerV5(lowercase=True)
    tokenizer.build_vocab(all_texts, min_freq=1)
    
    vocab_path = "data/tokenizer/vocab_v5.json"
    tokenizer.save_vocab(vocab_path)
    print(f"Saved V5 vocabulary containing {len(tokenizer.vocab)} tokens to {vocab_path}")

if __name__ == "__main__":
    main()
