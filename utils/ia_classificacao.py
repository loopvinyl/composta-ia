import pandas as pd
import numpy as np
import unicodedata
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------- FUNÇÕES AUXILIARES ----------
def normalizar_texto(txt):
    if pd.isna(txt):
        return ""
    txt = unicodedata.normalize("NFKD", str(txt))
    txt = txt.encode("ASCII", "ignore").decode("utf-8")
    return txt.upper().strip()

def classificar_destino_regra(texto):
    if pd.isna(texto):
        return "Indefinido"
    t = normalizar_texto(texto)
    if "COMPOSTAGEM" in t or "VERMICOMPOSTAGEM" in t:
        return "Compostagem"
    if "RECICLAGEM" in t or "TRIAGEM" in t or "COOPERATIVA" in t:
        return "Reciclagem"
    if "ATERRO SANITARIO" in t:
        return "Aterro Sanitario"
    if "ATERRO CONTROLADO" in t:
        return "Aterro Controlado"
    if "LIXAO" in t or "VAZADOURO" in t:
        return "Lixao"
    if "TRANSBORDO" in t:
        return "Transbordo"
    if "INCINERACAO" in t or "COPROCESSAMENTO" in t:
        return "Tratamento Termico"
    return "Outros"

# ---------- CLASSE PRINCIPAL ----------
class ClassificadorDestinoIA:
    def __init__(self, model_path="models/classificador_destino.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.classes_ = None

    def treinar_com_dados_snis(self, df, col_texto="DESTINO", col_target=None):
        print("🔄 Treinando modelo de IA com dados do SNIS...")
        if col_target not in df.columns or col_target is None:
            df['label_regra'] = df[col_texto].apply(classificar_destino_regra)
            y = df['label_regra']
        else:
            y = df[col_target]
        X = df[col_texto].fillna("").astype(str).apply(normalizar_texto)
        mask = X.str.strip() != ""
        X = X[mask]
        y = y[mask]
        min_samples = 5
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        mask_valid = y.isin(valid_classes)
        X = X[mask_valid]
        y = y[mask_valid]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))  # <--- CORREÇÃO DEFINITIVA
        ])
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.classes_
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Acurácia do classificador IA: {acc:.2%}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"💾 Modelo salvo em {self.model_path}")
        return self.pipeline

    def carregar_ou_treinar(self, df=None, col_texto="DESTINO"):
        if os.path.exists(self.model_path):
            print("📂 Carregando modelo de IA existente...")
            self.pipeline = joblib.load(self.model_path)
            self.classes_ = self.pipeline.classes_
            return self.pipeline
        else:
            if df is None:
                raise Exception("Modelo não encontrado e nenhum DataFrame fornecido para treino.")
            return self.treinar_com_dados_snis(df, col_texto)

    def prever(self, texto, threshold=0.5):
        if self.pipeline is None:
            return classificar_destino_regra(texto)
        texto_norm = normalizar_texto(str(texto))
        if texto_norm == "":
            return "Indefinido"
        probs = self.pipeline.predict_proba([texto_norm])[0]
        max_prob = max(probs)
        idx = list(probs).index(max_prob)
        classe_predita = self.pipeline.classes_[idx]
        if max_prob < threshold:
            return classificar_destino_regra(texto)
        return classe_predita
