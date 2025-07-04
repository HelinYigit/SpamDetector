import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Veri setini yükle
df = pd.read_csv("url_spam_classification.csv")
df['is_spam'] = df['is_spam'].astype(int)

# 2. Tokenizer
def extractUrl(data):
    url = str(data).lower()  # URL küçük harfe çevrilir
    extractSlash = url.split('/') # Slash (/) karakterine göre parçalanır
    result = []

    for i in extractSlash:
        # Her parçayı '-' karakterine göre böl
        extractDash = str(i).split('-')
        dotExtract = []

        for j in range(len(extractDash)):
            # Ardından '.' karakterine göre böl ve nokta ile bölünmüş tüm parçaları ekle
            extractDot = str(extractDash[j]).split('.')
            dotExtract += extractDot

        result += extractDash + dotExtract

    # Şüpheli anahtar kelimeler URL içinde geçiyorsa özel keyword tokenları ekle
    suspicious_keywords = [
        "login", "secure", "verify", "account", "update",
        "password", "urgent", "click", "confirm", "win",
        "bonus", "gift", "bit.ly", "tinyurl", "reset", "bank",
        "freestuff", "prize", "winner", "deal", "cheap", "limited"
    ]

    for keyword in suspicious_keywords:
        if keyword in url:
            result.append(f"keyword_{keyword}")

    return list(set(result))



# 3. Eğitim/Test bölmesi
urls = df['url']
labels = df['is_spam']
urls_train, urls_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)

# 4. Vektörizer
# Özel tokenizer kullanılarak kelime frekanslarını çıkar
cv = CountVectorizer(tokenizer=extractUrl)
# Eğitim verisi üzerinden vektörizer öğrenilir ve dönüştürülür
X_train = cv.fit_transform(urls_train)

# 5. Modeller
models = {
    "naive_bayes_token_model.joblib": MultinomialNB(),
    "svm_token_model.joblib": LinearSVC(max_iter=5000),
    "decision_tree_token_model.joblib": DecisionTreeClassifier(random_state=42)
}

# 6. Eğitim & Kayıt
for filename, model in models.items():
    print(f">>> {filename} eğitiliyor...")
    model.fit(X_train, y_train)
    joblib.dump(model, filename)

# 7. Vektörizeri Kaydet
joblib.dump(cv, "token_vectorizer.joblib")
print("✅ Tüm modeller ve vektörizer kaydedildi.")
