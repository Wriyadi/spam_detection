!pip install joblib spacy re
import streamlit as st
import joblib
import re
import spacy

# ==========================================
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Spam Email Detection", page_icon="📧", layout="centered")

# ==========================================
# 1. LOAD MODEL & SPACY
# ==========================================
# Menggunakan cache_resource agar model dan spacy tidak diload berulang kali
@st.cache_resource
def load_assets():
    try:
        # Load spaCy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        # Load Pipeline Model (Sudah termasuk TFIDF + SMOTE + LogReg)
        pipeline = joblib.load('spam_model_pipeline.pkl')
        return nlp, pipeline
    except FileNotFoundError:
        st.error("❌ File 'spam_model_pipeline.pkl' tidak ditemukan! Pastikan file tersebut ada di folder yang sama dengan app.py.")
        st.stop()
    except OSError:
         st.error("❌ Model spaCy 'en_core_web_sm' belum diinstall. Silakan buka terminal dan ketik: python -m spacy download en_core_web_sm")
         st.stop()

nlp, pipeline = load_assets()

# ==========================================
# 2. FUNGSI PREPROCESSING
# ==========================================
# Fungsi ini WAJIB SAMA PERSIS dengan yang ada di train_mlflow.py
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Menghapus URL & email
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Menghapus selain huruf alfabet
    text = re.sub(r'[^a-z\s]', '', text)
    
    doc = nlp(text)
    # Lemmatization & hapus stopwords
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
    return " ".join(tokens)

# ==========================================
# 3. ANTARMUKA STREAMLIT
# ==========================================
st.title("📧 Sistem Deteksi Spam Email")
st.markdown("""
Aplikasi web ini menggunakan **Natural Language Processing (NLP)** dan **Logistic Regression** yang telah dioptimasi dengan Pipeline untuk mengklasifikasikan apakah suatu teks email merupakan **Spam** atau **Bukan Spam (Ham)**.
""")

# Input teks dari pengguna
user_input = st.text_area("✍️ Masukkan isi teks email di bawah ini:", height=200, 
                          placeholder="Contoh: WIN BIG NOW!! Claim your limited offer cash guarantee...")

# Tombol Prediksi
if st.button("🔍 Analisis Email", use_container_width=True):
    if user_input.strip():
        with st.spinner('Menganalisis teks...'):
            
            # Langkah A: Preprocessing
            clean_text = preprocess_text(user_input)
            
            if not clean_text:
                st.warning("Teks tidak mengandung kata-kata yang valid untuk dianalisis setelah pembersihan (hanya berisi simbol/URL/Stopwords).")
            else:
                # Langkah B: Prediksi langsung menggunakan Pipeline
                # Perhatikan: input harus dalam bentuk list ([clean_text])
                prediction = pipeline.predict([clean_text])[0]
                proba = pipeline.predict_proba([clean_text])[0]
                
                # Langkah C: Tampilkan Hasil
                st.divider()
                # Berdasarkan asumsi label (biasanya 1 = Spam, 0 = Non-Spam)
                # SESUAIKAN JIKA LABEL ANDA BERBEDA
                if prediction == 1:
                    st.error(f"🚨 **PERINGATAN: EMAIL INI TERDETEKSI SEBAGAI SPAM**")
                    st.metric(label="Tingkat Keyakinan Model", value=f"{proba[1]:.2%}")
                else:
                    st.success(f"✅ **AMAN: EMAIL INI BUKAN SPAM (HAM)**")
                    st.metric(label="Tingkat Keyakinan Model", value=f"{proba[0]:.2%}")
                
                # Fitur tambahan untuk melihat detail pemrosesan
                with st.expander("🛠️ Lihat Detail Pemrosesan Teks"):
                    st.write("**Teks Asli:**")
                    st.info(user_input)
                    st.write("**Teks Bersih (Setelah Lemmatization & Hapus Stopwords):**")
                    st.info(clean_text)
    else:
        st.warning("⚠️ Silakan masukkan teks email terlebih dahulu sebelum melakukan analisis.")
