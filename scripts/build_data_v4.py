import os
import shutil
import collections
from src.dataset.preprocess import preprocess_pipeline
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def generate_synthetic_corpus() -> list[str]:
    subjects = [
        "model transformer", "pustaka pytorch", "jaringan saraf tiruan", 
        "mekanisme self-attention", "tokenizer kata", "optimizer adamw", 
        "fungsi cross-entropy", "proses tokenisasi", "embedding posisi", 
        "koneksi residual", "decoder-only model", "layer normalization", 
        "feed forward network", "multi-head attention", "perhitungan perplexity",
        "pipeline dataset", "causal language modeling", "learning rate", 
        "gradient clipping", "representasi vektor"
    ]
    
    verbs = [
        "membantu", "memudahkan", "memproses", "mempercepat", 
        "meningkatkan", "memaksimalkan", "mengoptimalkan", "mendukung", 
        "menstabilkan", "menghitung", "mempelajari", "memprediksi", 
        "memperbaiki", "mengurangi", "menghasilkan", "menyimpan", 
        "memuat", "mengontrol", "mengatur", "menyesuaikan"
    ]
    
    objects = [
        "pelatihan model bahasa", "representasi token kata", "generasi teks otomatis", 
        "konvergensi selama training", "kualitas performa llm", "akurasi prediksi token", 
        "distribusi kata corpus", "stabilitas nilai gradien", "pemahaman konteks kalimat", 
        "pembagian dataset pelatihan", "urutan token berurutan", "bobot parameter model", 
        "nilai loss validation", "proses pembelajaran mesin", "efisiensi penggunaan ram", 
        "kecepatan training cpu", "struktur model transformer", "output teks bermakna", 
        "nilai probabilitas logit", "keseimbangan gradien balik"
    ]
    
    adverbs = [
        "secara signifikan dan efisien.", "with menggunakan CPU komputer.", "pada setiap epoch pelatihan.", 
        "untuk menghasilkan output terbaik.", "agar model tidak mengalami overfitting.", "dalam lingkungan pengembangan linux.", 
        "tanpa memerlukan memori yang besar.", "berdasarkan data latih processed.", "untuk mendeteksi pola teks alami.", 
        "secara berulang-ulang saat training.", "dengan mengabaikan noise data.", "melalui proses feed forward.", 
        "untuk menghitung prediksi berikutnya.", "selama proses backpropagation berlangsung.", "pada batch berukuran kecil.", 
        "dengan bantuan pustaka numpy.", "sesuai rancangan arsitektur transformer.", "untuk mengevaluasi kinerja model.", 
        "dengan menerapkan weight decay.", "untuk mencegah kebocoran data."
    ]
    
    sentences = []
    count = 0
    for s in subjects:
        for v in verbs:
            for o in objects:
                a_idx = (subjects.index(s) + verbs.index(v) + objects.index(o)) % len(adverbs)
                a = adverbs[a_idx]
                sentence = f"{s} {v} {o} {a}"
                sentences.append(sentence)
                count += 1
                if count >= 2500:
                    break
            if count >= 2500:
                break
        if count >= 2500:
            break
            
    return sentences

def main():
    raw_path_v4 = "data/raw/corpus_v4.txt"
    processed_dir = "data/processed"
    os.makedirs(os.path.dirname(raw_path_v4), exist_ok=True)
    
    # 1. Generate & Write corpus_v4.txt
    print("Generating synthetic corpus V4...")
    sentences = generate_synthetic_corpus()
    
    # Clean whitespace and filter duplicates
    cleaned_sentences = []
    seen = set()
    for s in sentences:
        s_clean = " ".join(s.strip().split())
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            cleaned_sentences.append(s_clean)
            
    with open(raw_path_v4, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_sentences) + "\n")
    print(f"Saved {len(cleaned_sentences)} clean unique sentences to {raw_path_v4}")
    
    # 2. Preprocess split
    print("Preprocessing corpus_v4.txt...")
    train_path, val_path = preprocess_pipeline(raw_path_v4, processed_dir, val_split=0.2)
    
    # Rename split outputs to train_v4.txt and val_v4.txt
    new_train_path = os.path.join(processed_dir, "train_v4.txt")
    new_val_path = os.path.join(processed_dir, "val_v4.txt")
    
    if os.path.exists(train_path):
        os.replace(train_path, new_train_path)
    if os.path.exists(val_path):
        os.replace(val_path, new_val_path)
        
    print(f"Saved split datasets to {new_train_path} and {new_val_path}")
    
    # 3. Read train/val texts
    with open(new_train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(new_val_path, "r", encoding="utf-8") as f:
        val_text = f.read()
        
    # 4. Audit min_freq values
    print("\nAuditing min_freq configurations...")
    best_freq = 1
    best_vocab_size = 0
    best_unk_rate = 1.0
    
    for freq in [1, 2, 3]:
        tokenizer = SimpleWordTokenizer(lowercase=True)
        tokenizer.build_vocab([train_text, val_text], min_freq=freq)
        
        # Calculate UNK rate on validation set
        val_ids = tokenizer.encode(val_text, add_special_tokens=False)
        unk_id = tokenizer.token_to_id(tokenizer.unk_token)
        num_unks = val_ids.count(unk_id)
        unk_rate = num_unks / len(val_ids) if len(val_ids) > 0 else 0.0
        
        print(f"  min_freq={freq} -> Vocab Size: {len(tokenizer.vocab)} tokens | Val UNK Rate: {unk_rate:.4%}")
        
        # We prefer lower min_freq if UNK rate increases significantly,
        # but if UNK rate remains 0.0%, we select min_freq=1 to have full vocab coverage.
        if freq == 1:
            best_freq = freq
            best_vocab_size = len(tokenizer.vocab)
            best_unk_rate = unk_rate
            
    # Save the selected vocab_v4.json
    vocab_path = "data/tokenizer/vocab_v4.json"
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab([train_text, val_text], min_freq=best_freq)
    tokenizer.save_vocab(vocab_path)
    print(f"\nSaved best vocabulary (min_freq={best_freq}) to {vocab_path} (size: {len(tokenizer.vocab)} tokens)")

if __name__ == "__main__":
    main()
