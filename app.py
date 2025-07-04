from flask import Flask, request, jsonify
from flask_cors import CORS # CORS politikası izinleri (frontend erişimi için gerekli)
import joblib
import re
import numpy as np
from scipy.special import expit

# 1. Flask uygulaması oluşturuluyor
app = Flask(__name__)
# CORS middleware: farklı porttan gelen isteklerin kabul edilmesini sağlar
CORS(app)

# 2. URL tokenizer
def extractUrl(data):
    url = str(data).lower()
    extractSlash = url.split('/')
    result = []

    for i in extractSlash:
        extractDash = str(i).split('-')
        dotExtract = []

        for j in range(len(extractDash)):
            extractDot = str(extractDash[j]).split('.')
            dotExtract += extractDot

        result += extractDash + dotExtract

    suspicious_keywords = [
        "login", "secure", "verify", "account", "update",
        "password", "urgent", "click", "confirm", "win",
        "bonus", "gift", "bit.ly", "tinyurl", "reset", "bank"
    ]

    for keyword in suspicious_keywords:
        if keyword in url:
            result.append(f"keyword_{keyword}")

    return list(set(result))

# 3. Text modelleri (Email içeriği)
text_models = {
    "naive_bayes": joblib.load("naive_bayes_model.pkl"),
    "logistic_regression": joblib.load("logistic_regression_model.pkl"),
    "random_forest": joblib.load("random_forest_model.pkl"),
    "svm": joblib.load("svm_model.pkl"),
    "xgboost": joblib.load("xgboost_model.pkl")
}

text_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 4. URL modelleri
url_models = {
    "naive_bayes": joblib.load("naive_bayes_token_model.joblib"),
    "svm": joblib.load("svm_token_model.joblib"),
    "decision_tree": joblib.load("decision_tree_token_model.joblib")
}

url_vectorizer = joblib.load("token_vectorizer.joblib")

# 5. Metin temizleme
# E-mail içeriğinden sadece harfleri bırakıp temizler
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text) # Sadece harf ve boşluk bırak
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 6. Url çekimi
# E-mail içeriğindeki tüm URL’leri regex ile ayıklar
def extract_urls(text):
    return re.findall(r"https?://\S+|www\.\S+", text)

# 7. Text analizi API
@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json()
    # Gelen metin içeriği
    email_text = data.get("email_text", "")
    # Kullanıcının seçtiği model
    selected_model_name = data.get("model", "naive_bayes")

    # Temizlenmiş metin
    email_text_cleaned = clean_text(email_text)
    # TF-IDF vektöre dönüştür
    email_vec = text_vectorizer.transform([email_text_cleaned])

    scores = {}

    for model_name, model in text_models.items():
        if model_name == "svm":
            raw_score = model.decision_function(email_vec)[0]
            # Sigmoid uygulanarak olasılığa çevrilir
            score = 100 / (1 + np.exp(-raw_score))
        elif model_name == "xgboost":
            margin = model.predict(email_vec, output_margin=True)[0]
            # XGBoost'un margin çıktısına sigmoid uygulanır
            score = expit(np.clip(margin, -4, 4)) * 100
        else:
            # Spam sınıfı olasılığı
            score = model.predict_proba(email_vec)[0][1] * 100
        # Yüzdelik format
        scores[model_name] = round(score)

    # Tüm modellerin ortalama spam puanı
    average_score = round(np.mean(list(scores.values())))
    # Seçilen modelin skoru
    selected_score = scores.get(selected_model_name, 0)

    return jsonify({
        "spam_score": average_score,
        "model_scores": scores,
        "selected_score": selected_score
    })

# 8. URL analzi
@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json()
    email_text = data.get("email_text", "")
    selected_model_name = data.get("model", "naive_bayes")

    # Metin içindeki URL’leri çıkar
    urls = extract_urls(email_text)
    if not urls:
        return jsonify({"error": "No URLs found."})

    combined_url_text = ' '.join(urls)
    # Token tabanlı vektörizasyon
    vector = url_vectorizer.transform([combined_url_text])

    scores = {}
    for model_name, model in url_models.items():
        try:
            if hasattr(model, "decision_function"):
                raw_score = model.decision_function(vector)[0]
                score = 100 / (1 + np.exp(-raw_score))
            else:
                score = model.predict_proba(vector)[0][1] * 100
            scores[model_name] = round(score)
        except:
            scores[model_name] = 0 # Hata varsa skor sıfır ver

    avg_score = round(np.mean(list(scores.values())))
    selected_score = scores.get(selected_model_name, 0)

    return jsonify({
        "url_score": avg_score,
        "model_scores": scores,
        "selected_score": selected_score
    })

# 9. Sunucuyu başlat
if __name__ == "__main__":
    app.run(port=5000)