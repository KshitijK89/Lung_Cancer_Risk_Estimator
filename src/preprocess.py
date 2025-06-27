import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data():
    path = os.path.join("..", "data", "survey lung cancer.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

def preprocess_data(df):
    df = df.copy()

    # Encode Gender and Target
    label_enc = LabelEncoder()
    df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # Male=1, Female=0
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    # Features and target
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # Feature Scaling (optional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Execute preprocessing
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("✅ Train/Test Split Complete")
    print("X_train shape:", X_train.shape)
    print("y_train distribution:\n", y_train.value_counts())
