import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# 1. Veri setini yükle
df = pd.read_csv("url_spam_classification.csv")
df['is_spam'] = df['is_spam'].astype(int)

# 2. Özel tokenizer fonksiyonu
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
        # Tüm alt bölünmüş kelimeler listeye eklenir
        result += extractDash + dotExtract

    suspicious_keywords = [
        "login", "secure", "verify", "account", "update",
        "password", "urgent", "click", "confirm", "win",
        "bonus", "gift", "bit.ly", "tinyurl", "reset", "bank",
        "freestuff", "prize", "winner", "deal", "cheap", "limited"
    ]

    for keyword in suspicious_keywords:
        if keyword in url:
            result.append(f"keyword_{keyword}")

    # Aynı token tekrar etmesin diye set'e çevirilip listeye dönüştürülür
    return list(set(result))

# 3. Veri vektörizasyonu
urls = df['url']
labels = df['is_spam']
urls_train, urls_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)

# CountVectorizer, URL'leri özel tokenizer ile tokenlara ayırıp vektör haline getirir
vectorizer = CountVectorizer(tokenizer=extractUrl)
X_train = vectorizer.fit_transform(urls_train)
X_test = vectorizer.transform(urls_test)

# 4. Eğitilmiş modelleri yükle
models = {
    "Naive Bayes": joblib.load("naive_bayes_token_model.joblib"),
    "SVM": joblib.load("svm_token_model.joblib"),
    "Decision Tree": joblib.load("decision_tree_token_model.joblib")
}

# 5. Model performans değerlendirmesi
for name, model in models.items():
    print(f"\n📌 Model: {name}")

    # predict_proba sadece bazı modellerde vardır (örneğin Naive Bayes, Decision Tree)
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
    else:
        # SVM gibi predict_proba desteklemeyenlerde direkt predict
        y_pred = model.predict(X_test)

    # Performans metrikleri hesaplanır
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # Confusion matrix bileşenlerini ayır
    tn, fp, fn, tp = cm.ravel()

    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("Classification Report:")
    print(report)
