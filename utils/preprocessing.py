import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data():
    # ✅ Load local dataset
    df = pd.read_csv("data/lung_cancer.csv")

    # =========================
    # CLEAN TARGET
    # =========================
    df['LUNG_CANCER'] = df['LUNG_CANCER'].astype(str).str.strip().str.upper()
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    # =========================
    # HANDLE CATEGORICAL DATA
    # =========================
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

    # Convert all remaining columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # =========================
    # REMOVE INVALID TARGET ROWS
    # =========================
    df = df[df['LUNG_CANCER'].notna()]

    # =========================
    # SPLIT FEATURES & TARGET
    # =========================
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # =========================
    # HANDLE MISSING VALUES
    # =========================
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)