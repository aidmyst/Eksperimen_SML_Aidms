"""
==========================================================
Script Otomatisasi Data Preprocessing
==========================================================
Nama        : Aidms
Dataset     : Diabetes Dataset (Scikit-learn)
Deskripsi   : Script ini mengotomatiskan seluruh pipeline
              data preprocessing sesuai Template Eksperimen MSML.

Tahapan Preprocessing:
  1. Memuat dataset
  2. Menangani Missing Values
  3. Menghapus Data Duplikat
  4. Deteksi dan Penanganan Outlier (IQR Capping)
  5. Standarisasi Fitur (StandardScaler)
  6. Split Train/Test
  7. Menyimpan hasil preprocessing
==========================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys


def load_data(file_path):
    """
    Tahap 1: Memuat dataset dari file CSV.

    Parameters:
        file_path (str): Path ke file CSV dataset.

    Returns:
        pd.DataFrame: DataFrame yang berisi dataset.
    """
    print("=" * 60)
    print("TAHAP 1: MEMUAT DATASET")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"[ERROR] File tidak ditemukan: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print(f"  Dataset berhasil dimuat dari: {file_path}")
    print(f"  Dimensi dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"  Kolom: {df.columns.tolist()}")
    print(f"  Tipe data:\n{df.dtypes.to_string()}")
    print()
    return df


def handle_missing_values(df):
    """
    Tahap 2: Menangani Missing Values.
    Strategi: Imputasi menggunakan median untuk kolom numerik.

    Parameters:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa missing values.
    """
    print("=" * 60)
    print("TAHAP 2: MENANGANI MISSING VALUES")
    print("=" * 60)

    missing_total = df.isnull().sum().sum()
    print(f"  Total missing values: {missing_total}")

    if missing_total > 0:
        print("  Melakukan imputasi dengan median...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                count = df[col].isnull().sum()
                df[col].fillna(median_val, inplace=True)
                print(f"    - Kolom '{col}': {count} missing -> diisi median ({median_val:.4f})")
        print(f"  Missing values setelah imputasi: {df.isnull().sum().sum()}")
    else:
        print("  Tidak ada missing values. Data sudah lengkap.")

    print()
    return df


def remove_duplicates(df):
    """
    Tahap 3: Menghapus Data Duplikat.

    Parameters:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa data duplikat.
    """
    print("=" * 60)
    print("TAHAP 3: MENGHAPUS DATA DUPLIKAT")
    print("=" * 60)

    duplicates = df.duplicated().sum()
    print(f"  Jumlah data duplikat: {duplicates}")

    if duplicates > 0:
        rows_before = df.shape[0]
        df = df.drop_duplicates()
        rows_after = df.shape[0]
        print(f"  Baris sebelum: {rows_before} -> Baris setelah: {rows_after}")
        print(f"  {duplicates} baris duplikat berhasil dihapus.")
    else:
        print("  Tidak ada data duplikat.")

    print()
    return df


def handle_outliers(df, target_col='target'):
    """
    Tahap 4: Deteksi dan Penanganan Outlier menggunakan IQR Method.
    Strategi: Capping (mengganti outlier dengan batas IQR).

    Parameters:
        df (pd.DataFrame): DataFrame input.
        target_col (str): Nama kolom target (tidak di-cap).

    Returns:
        pd.DataFrame: DataFrame dengan outlier yang sudah ditangani.
    """
    print("=" * 60)
    print("TAHAP 4: DETEKSI DAN PENANGANAN OUTLIER (IQR)")
    print("=" * 60)

    features = [col for col in df.columns if col != target_col]
    total_outliers = 0

    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outlier_count > 0:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"  - Fitur '{col}': {outlier_count} outlier di-capping "
                  f"ke [{lower_bound:.4f}, {upper_bound:.4f}]")
            total_outliers += outlier_count

    if total_outliers == 0:
        print("  Tidak ada outlier yang terdeteksi.")
    else:
        print(f"  Total outlier yang ditangani: {total_outliers}")

    print()
    return df


def standardize_and_split(df, target_col='target', test_size=0.2, random_state=42):
    """
    Tahap 5: Standarisasi Fitur dan Split Data Train/Test.

    Parameters:
        df (pd.DataFrame): DataFrame input.
        target_col (str): Nama kolom target.
        test_size (float): Proporsi data testing.
        random_state (int): Random seed untuk reprodusibilitas.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("=" * 60)
    print("TAHAP 5: STANDARISASI FITUR & SPLIT TRAIN/TEST")
    print("=" * 60)

    # Memisahkan fitur dan target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    print(f"  Shape fitur (X): {X.shape}")
    print(f"  Shape target (y): {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")

    # Standarisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Standarisasi selesai (StandardScaler)")
    print(f"  Mean training (mendekati 0): {np.mean(X_train_scaled, axis=0).round(6)}")
    print(f"  Std training (mendekati 1): {np.std(X_train_scaled, axis=0).round(6)}")

    print()
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_preprocessed_data(df, output_path):
    """
    Tahap 6: Menyimpan hasil preprocessing ke file CSV.

    Parameters:
        df (pd.DataFrame): DataFrame yang sudah dipreprocess.
        output_path (str): Path untuk menyimpan file output.
    """
    print("=" * 60)
    print("TAHAP 6: MENYIMPAN HASIL PREPROCESSING")
    print("=" * 60)

    # Pastikan direktori output ada
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, index=False)
    print(f"  File disimpan ke: {output_path}")
    print(f"  Dimensi dataset akhir: {df.shape[0]} baris x {df.shape[1]} kolom")
    print()


def print_summary(df_original, df_final, output_path):
    """Menampilkan ringkasan preprocessing."""
    print("=" * 60)
    print("RINGKASAN DATA PREPROCESSING")
    print("=" * 60)
    print(f"  Dataset Awal     : {df_original.shape[0]} baris x {df_original.shape[1]} kolom")
    print(f"  Dataset Akhir    : {df_final.shape[0]} baris x {df_final.shape[1]} kolom")
    print(f"  Missing Values   : Ditangani (imputasi median)")
    print(f"  Duplikat         : Ditangani (dihapus)")
    print(f"  Outlier          : Ditangani (IQR capping)")
    print(f"  Standarisasi     : StandardScaler")
    print(f"  Train/Test Split : 80/20 (random_state=42)")
    print(f"  Output File      : {output_path}")
    print("=" * 60)
    print("\n[SUCCESS] Preprocessing selesai!")


def main():
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing.
    """
    print()
    print("##########################################################")
    print("#  OTOMATISASI DATA PREPROCESSING - TEMPLATE MSML        #")
    print("#  Dataset: Diabetes (Scikit-learn)                       #")
    print("##########################################################")
    print()

    # Konfigurasi path
    raw_data_path = os.path.join('..', 'namadataset_raw', 'diabetes.csv')
    output_path = 'diabetes_preprocessing.csv'

    # ============================
    # Pipeline Preprocessing
    # ============================

    # Tahap 1: Memuat dataset
    df = load_data(raw_data_path)
    df_original = df.copy()

    # Tahap 2: Menangani Missing Values
    df = handle_missing_values(df)

    # Tahap 3: Menghapus Data Duplikat
    df = remove_duplicates(df)

    # Tahap 4: Deteksi dan Penanganan Outlier
    df = handle_outliers(df, target_col='target')

    # Tahap 5: Standarisasi dan Split Data
    X_train, X_test, y_train, y_test, scaler = standardize_and_split(
        df, target_col='target', test_size=0.2, random_state=42
    )

    # Tahap 6: Menyimpan hasil preprocessing
    save_preprocessed_data(df, output_path)

    # Ringkasan
    print_summary(df_original, df, output_path)


if __name__ == '__main__':
    main()
