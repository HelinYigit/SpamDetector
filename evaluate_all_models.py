import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Veri setini yükle
df = pd.read_csv("enron_processed_dataset.csv")
df = df.dropna(subset=["processed_text"])

# 2. Özellik ve etiket ayır
y = df["Spam/Ham"]
X = df["processed_text"]

# 3. Etiketleri sayısal hale getir
le = LabelEncoder() # "spam" → 1, "ham" → 0 gibi sayısal dönüştürme yapar
y_encoded = le.fit_transform(y)

# 4. Train/Test böl (aynı oranlarda)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. TF-IDF vektörizeri yükle
vectorizer = joblib.load("tfidf_vectorizer.pkl")
# Test verisi TF-IDF formatına dönüştürülüyor
X_test_vec = vectorizer.transform(X_test)

# 6. Model dosya isimleri ve etiketleri
model_files = {
    "Naive Bayes": "naive_bayes_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# 7. Performans skorları için tablo
results = []

# 8. Her modeli test et
for model_name, model_file in model_files.items():
    print(f"\n===== {model_name} =====")
    model = joblib.load(model_file)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred) # Doğruluk oranı
    cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
    tn, fp, fn, tp = cm.ravel() # Confusion matrix değerlerini aç

    # False Positive Rate (gerçek ham ama spam diye sınıflananların oranı)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # False Negative Rate (gerçek spam ama ham diye sınıflananların oranı)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")

    # Sonuçları bir sözlük olarak listeye ekle
    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    })

# 9. Pandas tablosu olarak özet göster
results_df = pd.DataFrame(results)
print("\n=== Karşılaştırmalı Skor Tablosu ===")
print(results_df.to_string(index=False))
