import pandas as pd
import numpy as np
import re
import spacy
import mlflow
import mlflow.sklearn
import joblib
import dagshub

# Evaluasi & Split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Pipeline, Ekstraksi Fitur & Model
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 0. KONFIGURASI MLFLOW DAGSHUB
# ==========================================
# ⚠️ GANTI BAGIAN INI DENGAN DATA REPOSITORY DAGSHUB ANDA
REPO_OWNER = "wriyadi5"
REPO_NAME = "spam-email-detection"

dagshub.init(
    repo_owner=REPO_OWNER,
    repo_name=REPO_NAME,
    mlflow=True
)

# Membuat atau menggunakan eksperimen yang sudah ada di DagsHub
mlflow.set_experiment("NLP_Spam_Email_Detection")

# ==========================================
# 1. LOAD SPACY & FUNGSI PREPROCESSING
# ==========================================
print("📥 Loading spaCy model...")
# Jika error, pastikan Anda telah menjalankan: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text):
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
# 2. LOAD DATASET
# ==========================================
def load_and_preprocess():
    print("📥 Loading dataset spam_email_dataset.csv ...")
    # Pastikan file ini ada di direktori yang sama dengan script
    df = pd.read_csv('spam_email_dataset.csv')
    df = df.dropna(subset=['email_text', 'label'])
    
    print("🧹 Membersihkan teks (ini memakan waktu bergantung spesifikasi PC Anda)...")
    df['clean_text'] = df['email_text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['label']
    
    # Split data stratify agar rasio spam/non-spam seimbang pada data latih & uji
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==========================================
# 3. MAIN EXECUTION & MLFLOW TRACKING
# ==========================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    print("🚀 Memulai proses Training & Remote Tracking ke DagsHub...")
    
    with mlflow.start_run(run_name="ImbPipeline_TFIDF_SMOTE_LogReg"):
        
        # --- PARAMETER MODEL ---
        params = {
            "tfidf__max_features": 5000,
            "logreg__C": 1.0,
            "logreg__solver": "liblinear",
            "logreg__random_state": 42
        }
        
        # Log konfigurasi/parameter ke MLflow DagsHub
        mlflow.log_params(params)
        
        # --- SUSUN IMBPIPELINE ---
        # Alur: Ekstrak Fitur (TF-IDF) -> Seimbangkan Kelas (SMOTE) -> Prediksi (Logistic Regression)
        pipeline = ImbPipeline([
            ('tfidf', TfidfVectorizer(max_features=params["tfidf__max_features"])),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(
                C=params["logreg__C"], 
                solver=params["logreg__solver"], 
                random_state=params["logreg__random_state"]
            ))
        ])
        
        # Latih Model Keseluruhan (Pipeline)
        print("⏳ Sedang melatih pipeline model...")
        pipeline.fit(X_train, y_train)
        
        # Prediksi & Evaluasi
        print("🎯 Melakukan evaluasi pada data test...")
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        
        # Log Metrik Hasil Evaluasi ke Cloud
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_weighted", f1_w)
        
        # Log keseluruhan model ke MLflow DagsHub & Simpan lokal sebagai cadangan
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="spam_email_pipeline")
        joblib.dump(pipeline, "spam_model_pipeline.pkl")
        
        print(f"\n✅ Training Selesai!")
        print(f"Accuracy: {acc:.4f} | F1 (Weighted): {f1_w:.4f}")
        print("\n📝 Classification Report:")
        print(classification_report(y_test, y_pred))
        print("💾 Model tersimpan secara lokal sebagai 'spam_model_pipeline.pkl'")
        print(f"📊 Cek hasil eksperimen dan model UI langsung di: https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")