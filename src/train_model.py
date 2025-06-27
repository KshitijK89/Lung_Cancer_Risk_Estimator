import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from preprocess import load_data, preprocess_data

# Load and preprocess
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# 1️⃣ Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("🔍 Logistic Regression")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# 2️⃣ Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("🌳 Random Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 3️⃣ Save the Better Model
best_model = rf_model if accuracy_score(y_test, rf_pred) > accuracy_score(y_test, lr_pred) else lr_model

model_path = os.path.join("..", "models", "model.pkl")
joblib.dump(best_model, model_path)
print(f"\n✅ Best model saved to {model_path}")
