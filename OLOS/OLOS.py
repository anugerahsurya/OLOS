# === Library Dasar Python ===
import os
import pickle
from collections import Counter

# === Library untuk Analisis dan Visualisasi Data ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

# === Library untuk Machine Learning & Preprocessing ===
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.tree import plot_tree, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_percentage_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from sklearn.preprocessing import label_binarize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# === Library untuk Model Tambahan ===
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# === Library untuk Hyperparameter Tuning ===
import optuna

# === Library untuk Interpretasi Model ===
import shap

# === Library untuk Penanganan Data Imbalanced ===
from imblearn.over_sampling import SMOTE

# === Library untuk Dataset Eksternal ===
from datasets import load_dataset

# === Library Tambahan untuk Visualisasi Pohon ===
import graphviz
from tqdm import tqdm
from itertools import combinations

# === Supresi Warning ===
import warnings

## Visualisasi Data

def visualisasiData(df):
    """
    Memvisualisasikan semua kolom numerik dan kategorik dalam sebuah DataFrame.

    Untuk setiap kolom numerik:
    - Histogram + KDE (sebaran)
    - Box plot (untuk deteksi outlier)

    Untuk setiap kolom kategorik:
    - Bar chart (frekuensi masing-masing kategori)

    Parameter:
    df (pd.DataFrame): DataFrame yang ingin divisualisasikan.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category','bool']).columns

    # Visualisasi kolom numerik
    if len(numeric_cols) > 0:
        print(f"Memvisualisasikan {len(numeric_cols)} kolom numerik...")
        for col in numeric_cols:
            plt.figure(figsize=(12, 5))

            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, color='skyblue')
            plt.title(f'Histogram dari {col}')
            plt.xlabel(col)
            plt.ylabel('Frekuensi')

            # Box plot
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[col], color='lightcoral')
            plt.title(f'Box Plot dari {col}')
            plt.ylabel(col)

            plt.tight_layout()
            plt.show()
    else:
        print("Tidak ada kolom numerik dalam DataFrame ini.")

    # Visualisasi kolom kategorik
    if len(categorical_cols) > 0:
        print(f"Memvisualisasikan {len(categorical_cols)} kolom kategorik...")
        for col in categorical_cols:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col, palette='pastel', order=df[col].value_counts().index)
            plt.title(f'Bar Chart dari {col}')
            plt.xlabel(col)
            plt.ylabel('Frekuensi')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    else:
        print("Tidak ada kolom kategorik dalam DataFrame ini.")

def preprocessingData(df, kolom, jenis=1, save_dir="Encoder Tersimpan", ordinal_order=None):
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = os.path.join(save_dir, f"{kolom}_encoder.pkl")

    if jenis == 1:  # Label Encoding
        le = LabelEncoder()
        df[kolom] = le.fit_transform(df[kolom])
        with open(encoder_path, 'wb') as file:
            pickle.dump(le, file)
        return df, le

    elif jenis == 2:  # One Hot Encoding
        # One Hot Encoding tidak memerlukan encoder saat "use"
        df = pd.get_dummies(df, columns=[kolom], prefix=kolom)
        return df, None

    elif jenis == 3:  # Ordinal Encoding
        if ordinal_order is None:
            raise ValueError(f"Untuk kolom '{kolom}', 'ordinal_order' harus disediakan untuk Ordinal Encoding.")

        df[kolom] = df[kolom].astype(str)
        oe = OrdinalEncoder(categories=[ordinal_order])
        df[kolom] = oe.fit_transform(df[[kolom]])
        with open(encoder_path, 'wb') as file:
            pickle.dump(oe, file)
            
        return df, oe
    else:
        raise ValueError("Jenis encoding tidak valid: gunakan 1 (Label), 2 (One Hot), atau 3 (Ordinal)")
          
## Hitung Korelasi dan VIF
def korelasiVIFTarget(df, target):
    """
    Menghitung korelasi Pearson terhadap kolom target dan VIF antar fitur numerik.

    Parameter:
        df (pd.DataFrame): DataFrame input.
        target (str): Nama kolom target.

    Return:
        pd.DataFrame: DataFrame berisi kolom:
            - 'Fitur': nama kolom fitur,
            - 'Korelasi': korelasi terhadap target,
            - 'VIF': nilai Variance Inflation Factor.
    """
    if target not in df.columns:
        raise ValueError(f"DataFrame tidak memiliki kolom target '{target}'.")

    # Pilih fitur numerik tanpa kolom target
    numeric_features = df.select_dtypes(include='number').drop(columns=[target])

    # Hitung korelasi Pearson terhadap target
    korelasi = df[numeric_features.columns].corrwith(df[target])

    # Hitung VIF
    X = add_constant(numeric_features)
    vif_values = [variance_inflation_factor(X.values, i + 1) for i in range(len(numeric_features.columns))]

    # Gabungkan ke dalam satu DataFrame
    result_df = pd.DataFrame({
        'Fitur': numeric_features.columns,
        'Korelasi': korelasi.values,
        'VIF': vif_values
    })

    # Urutkan berdasarkan nilai absolut korelasi
    result_df = result_df.reindex(result_df['Korelasi'].abs().sort_values(ascending=False).index)

    return result_df.reset_index(drop=True)

def EliminasiOutlier(X_train, y_train, contamination=0.05, case=1):
    # Fungsi untuk mendeteksi outlier menggunakan Isolation Forest
    def detect_outliers_isolation_forest(X, contamination):
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(X)
        df = pd.DataFrame(X, columns=X.columns)
        df['Outlier'] = outliers == -1
        return df

    # Fungsi untuk mengeliminasi outlier dari dataset
    def eliminate_outliers(df):
        cleaned_df = df[~df['Outlier']].drop(columns=['Outlier'])
        return cleaned_df

    # Standardisasi fitur X
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    if case == 1:
        # CASE 1: Deteksi outlier secara keseluruhan
        outlier_df = detect_outliers_isolation_forest(X_scaled, contamination)
        outlier_df[y_train.name] = y_train.values
        cleaned_data_scaled = eliminate_outliers(outlier_df)

    elif case == 2:
        # CASE 2: Deteksi outlier per kelas
        cleaned_parts = []

        for class_label in y_train.unique():
            class_mask = y_train == class_label
            X_class = X_scaled[class_mask]
            y_class = y_train[class_mask].reset_index(drop=True)

            outlier_df = detect_outliers_isolation_forest(X_class, contamination)
            outlier_df[y_train.name] = y_class
            cleaned_class_data = eliminate_outliers(outlier_df)
            cleaned_parts.append(cleaned_class_data)

        # Gabungkan kembali semua kelas yang telah dibersihkan
        cleaned_data_scaled = pd.concat(cleaned_parts, ignore_index=True)
    else:
        raise ValueError("Parameter 'case' hanya bisa bernilai 1 (umum) atau 2 (klasifikasi per kelas).")

    # Kembalikan data ke skala asli
    cleaned_X = scaler.inverse_transform(cleaned_data_scaled.drop(columns=[y_train.name]))
    cleaned_X = pd.DataFrame(cleaned_X, columns=X_train.columns)
    cleaned_y = cleaned_data_scaled[y_train.name].reset_index(drop=True)

    # Output info
    print(f"Jumlah data sebelum eliminasi outlier: {len(X_train)}")
    print(f"Jumlah data setelah eliminasi outlier: {len(cleaned_X)}")

    return cleaned_X, cleaned_y

def eksplorasiBestFitur(fixed_features, target, data, case=1):
    """
    Menguji semua kombinasi fitur tambahan dengan fixed features untuk regresi atau klasifikasi.

    Parameters:
    - fixed_features (list): Fitur yang selalu digunakan.
    - available_features (list): Fitur tambahan yang akan dikombinasikan.
    - target (str): Nama kolom target.
    - df (pd.DataFrame): Dataset utama.
    - case (int): 1 untuk regresi (RMSE), 2 untuk klasifikasi (F1 macro).

    Returns:
    - pd.DataFrame: Hasil kombinasi fitur, skor metrik terbaik, dan model terbaik.
    """
    results = []
    # Fitur tambahan
    available_features = [col for col in data.columns if col != target and col not in fixed_features]
    
    for i in tqdm(range(1, len(available_features) + 1), desc="Menguji kombinasi fitur tambahan"):
        for combo in combinations(available_features, i):
            features = fixed_features + list(combo)

            X = data[features]
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            if case == 1:  # Regresi
                models = {
                    'KNN': KNeighborsRegressor(),
                    'CatBoost': CatBoostRegressor(verbose=0, random_state=1),
                    'RandomForest': RandomForestRegressor(random_state=1),
                    'SVR': SVR(),
                    'LightGBM': LGBMRegressor(random_state=1, verbose=-1)
                }
                metrics = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    metrics[name] = root_mean_squared_error(y_test, preds)

                best_model = min(metrics, key=metrics.get)
                best_score = round(metrics[best_model], 4)

            elif case == 2:  # Klasifikasi
                models = {
                    'KNN': KNeighborsClassifier(),
                    'CatBoost': CatBoostClassifier(verbose=0, random_state=1),
                    'RandomForest': RandomForestClassifier(random_state=1),
                    'SVC': SVC(),
                    'LightGBM': LGBMClassifier(random_state=1, verbose=-1)
                }
                metrics = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    metrics[name] = f1_score(y_test, preds, average='macro')

                best_model = max(metrics, key=metrics.get)
                best_score = round(metrics[best_model], 4)

            else:
                raise ValueError("Case harus bernilai 1 (regresi) atau 2 (klasifikasi).")

            results.append({
                'features': features,
                'score': best_score,
                'best_model': best_model
            })

    hasil_df = pd.DataFrame(results)
    hasil_df = hasil_df.sort_values(by='score', ascending=(case == 1)).reset_index(drop=True)
    return hasil_df

def klasifikasikan_indeks_gabungan(nilai):
    if nilai < 1: return 0
    elif nilai < 2: return 1
    elif nilai < 3: return 2
    elif nilai < 4: return 3
    else: return 4


def get_model_and_params(model_type, trial):
    if model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        model_cls = RandomForestRegressor
    elif model_type == "knn":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2)
        }
        model_cls = KNeighborsRegressor
    elif model_type == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "verbose": 0
        }
        model_cls = CatBoostRegressor
    else:
        raise ValueError(f"Model {model_type} tidak dikenali.")
    
    return model_cls, params


def optimasiIndeksGabungan(
    X_A_train, X_A_val, yA_train, yA_val,
    X_B_train, X_B_val, yB_train, yB_val,
    X_C_train, X_C_val, yC_train, yC_val,
    n_trials=50,
    random_state=42
):
    def objective(trial):
        model_A_type = trial.suggest_categorical("model_A", ["rf", "knn", "catboost"])
        model_B_type = trial.suggest_categorical("model_B", ["rf", "knn", "catboost"])
        model_C_type = trial.suggest_categorical("model_C", ["rf", "knn", "catboost"])

        model_A_cls, params_A = get_model_and_params(model_A_type, trial)
        model_B_cls, params_B = get_model_and_params(model_B_type, trial)
        model_C_cls, params_C = get_model_and_params(model_C_type, trial)

        model_A = model_A_cls(**params_A)
        model_B = model_B_cls(**params_B)
        model_C = model_C_cls(**params_C)

        model_A.fit(X_A_train, yA_train)
        model_B.fit(X_B_train, yB_train)
        model_C.fit(X_C_train, yC_train)

        pred_A = model_A.predict(X_A_val)
        pred_B = model_B.predict(X_B_val)
        pred_C = model_C.predict(X_C_val)

        gabungan_pred = (pred_A + pred_B + pred_C) / 3
        gabungan_true = (yA_val + yB_val + yC_val) / 3

        klasifikasi_pred = [klasifikasikan_indeks_gabungan(x) for x in gabungan_pred]
        klasifikasi_true = [klasifikasikan_indeks_gabungan(x) for x in gabungan_true]

        return f1_score(klasifikasi_true, klasifikasi_pred, average='macro')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    # Ambil model terbaik
    model_A_type = best_params["model_A"]
    model_B_type = best_params["model_B"]
    model_C_type = best_params["model_C"]

    # Bangun ulang model dengan best params
    trial_fixed = optuna.trial.FixedTrial(best_params)
    model_A_cls, config_A = get_model_and_params(model_A_type, trial_fixed)
    model_B_cls, config_B = get_model_and_params(model_B_type, trial_fixed)
    model_C_cls, config_C = get_model_and_params(model_C_type, trial_fixed)

    model_A = model_A_cls(**config_A)
    model_B = model_B_cls(**config_B)
    model_C = model_C_cls(**config_C)

    model_A.fit(X_A_train, yA_train)
    model_B.fit(X_B_train, yB_train)
    model_C.fit(X_C_train, yC_train)

    pred_A = model_A.predict(X_A_val)
    pred_B = model_B.predict(X_B_val)
    pred_C = model_C.predict(X_C_val)

    gabungan_pred = (pred_A + pred_B + pred_C) / 3
    gabungan_true = (yA_val + yB_val + yC_val) / 3

    klasifikasi_pred = [klasifikasikan_indeks_gabungan(x) for x in gabungan_pred]
    klasifikasi_true = [klasifikasikan_indeks_gabungan(x) for x in gabungan_true]

    # Evaluasi lengkap
    f1_macro = f1_score(klasifikasi_true, klasifikasi_pred, average='macro')
    balanced_acc = balanced_accuracy_score(klasifikasi_true, klasifikasi_pred)
    cls_report = classification_report(klasifikasi_true, klasifikasi_pred, output_dict=True)
    conf_matrix = confusion_matrix(klasifikasi_true, klasifikasi_pred)

    # ROC AUC (binarize label)
    labels = sorted(set(klasifikasi_true + klasifikasi_pred))
    y_true_bin = label_binarize(klasifikasi_true, classes=labels)
    y_pred_bin = label_binarize(klasifikasi_pred, classes=labels)
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')

    # Feature importance helper
    def extract_importance(model, model_type, feature_names):
        if model_type == "rf":
            return dict(zip(feature_names, model.feature_importances_))
        elif model_type == "catboost":
            importances = model.get_feature_importance()
            return dict(zip(feature_names, importances))
        else:  # knn tidak memiliki feature importance
            return None

    def safe_colnames(X, fallback_prefix="X"):
        if hasattr(X, "columns"):
            return X.columns
        return [f"{fallback_prefix}{i}" for i in range(X.shape[1])]

    importance_A = extract_importance(model_A, model_A_type, safe_colnames(X_A_train, "A"))
    importance_B = extract_importance(model_B, model_B_type, safe_colnames(X_B_train, "B"))
    importance_C = extract_importance(model_C, model_C_type, safe_colnames(X_C_train, "C"))

    return {
        "model_A": model_A_type,
        "params_A": config_A,
        "importance_A": importance_A,

        "model_B": model_B_type,
        "params_B": config_B,
        "importance_B": importance_B,

        "model_C": model_C_type,
        "params_C": config_C,
        "importance_C": importance_C,

        "f1_macro": f1_macro,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": conf_matrix,
        "classification_report": cls_report,
        "roc_auc_macro": roc_auc
    }





