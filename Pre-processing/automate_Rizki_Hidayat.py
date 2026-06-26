from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # ← Ganti LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.bounds_:
                lower, upper = self.bounds_[col]
                X_df[col] = np.clip(X_df[col], lower, upper)
        return X_df.values


def preprocess_data(data, target_column, save_path, file_path): 
    print(f"\n{'='*50}")
    print(f"  Dataset shape     : {data.shape}")
    print(f"  Target column     : {target_column}")
    print(f"  Missing values    : {data.isnull().sum().sum()}")
    print(f"{'='*50}\n")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"✅ Train size : {X_train.shape}")
    print(f"✅ Test size  : {X_test.shape}\n")

    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print(f"📊 Numeric features    ({len(numeric_features)})    : {numeric_features}")
    print(f"🏷️  Categorical features ({len(categorical_features)}) : {categorical_features}\n")

    column_names = X.columns
    df_header = pd.DataFrame(columns=column_names)
    df_header.to_csv(file_path, index=False)
    print(f"💾 Header kolom disimpan ke: {file_path}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            random_state=42
        )),
        ('outlier_capper', OutlierCapper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('imputer', IterativeImputer(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            initial_strategy='most_frequent',
            random_state=42
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


    print("⚙️  Menjalankan preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    dump(preprocessor, save_path)
    print(f"💾 Pipeline disimpan ke: {save_path}")

    print(f"\n✅ Preprocessing selesai!")
    print(f"   X_train_processed : {X_train_processed.shape}")
    print(f"   X_test_processed  : {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease UCI Preprocessing Pipeline")
    parser.add_argument('--input',  type=str, default='heart_disease_uci_raw.csv')
    parser.add_argument('--target', type=str, default='num')
    parser.add_argument('--model',  type=str, default='preprocessing/preprocessor.joblib')
    parser.add_argument('--header', type=str, default='preprocessing/heart_disease_uci_preprocessing.csv')
    args = parser.parse_args()

    print(f"\n📂 Membaca dataset: {args.input}")
    try:
        heart_data = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{args.input}' tidak ditemukan.")
        print("   Pastikan file CSV berada di direktori yang sama dengan script ini.\n")
        exit(1)

    X_train_clean, X_test_clean, y_train, y_test = preprocess_data(
        data          = heart_data,
        target_column = args.target,
        save_path     = args.model,
        file_path     = args.header
    )