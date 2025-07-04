import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import joblib

# 1. Veriyi yükle
print("Veri yükleniyor...")
df = pd.read_csv("enron_processed_dataset_updated.csv")
df = df.dropna(subset=["processed_text"])
X = df["processed_text"]
y = df["Spam/Ham"].map({"ham": 0, "spam": 1})  # Sayısal hale getir

# 2. TF-IDF vektörizer
print("Vektörizer oluşturuluyor...")
vectorizer = TfidfVectorizer(
    lowercase=True,
  
    max_df=0.95,
    min_df=3,
 
)

X_vec = vectorizer.fit_transform(X)

# 3. Eğitim/Test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 4. Modellerin eğitilmesi
print("Naive Bayes eğitiliyor...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Logistic Regression eğitiliyor...")
# Max iterasyon artırıldı (default 100), çünkü convergence hatası olabilir
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

print("Random Forest eğitiliyor...")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

print("SVM (LinearSVC) eğitiliyor...")
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

print("XGBoost eğitiliyor...")
xgb_model = XGBClassifier(
    use_label_encoder=False, # Gereksiz uyarıları engeller
    eval_metric='logloss' # Eğitimde log loss metriğini kullanır
)

xgb_model.fit(X_train, y_train)

# 5. Kaydet
print("Modeller ve vektörizer kaydediliyor...")
# Tüm modellerde ortak kullanılacak vektörizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
# Her modelin kendi .pkl dosyasına kaydı
joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(log_model, "logistic_regression_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")

print("Tüm modeller başarıyla eğitildi ve kaydedildi.")
