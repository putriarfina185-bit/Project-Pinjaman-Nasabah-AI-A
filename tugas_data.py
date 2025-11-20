import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_excel('dataset-pinjaman-nasabah.xlsx')
    print("Dataset berhasil dimuat dari file Excel.")
except FileNotFoundError:
    print("ERROR: File 'dataset-pinjaman-nasabah.xlsx' tidak ditemukan.")
    exit()
except ImportError:
    print("ERROR: Library 'openpyxl' tidak terinstal. Harap jalankan: pip install openpyxl")
    exit()

cat_cols = ['JenisKelamin', 'StatusPernikahan', 'Wiraswasta', 'Credit_History', 
            'JumTanggungan', 'Pendidikan', 'IncomeNasabah', 'IncomePasangan', 
            'JangkaWaktuPinjaman', 'WilayahTempatTinggal', 'StatusPinjaman']

num_cols = ['JumlahPinjaman']

print("\n--- Memproses Imputasi Nilai Hilang ---")
for col in cat_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

print("Imputasi selesai. Jumlah nilai hilang:", df.isnull().sum().sum())

print("\n--- Melakukan Feature Engineering ---")
df['Total_Income'] = df['IncomeNasabah'] + df['IncomePasangan']
df['Total_Income'] = df['Total_Income'].replace(0, np.nan)
df['Total_Income'] = df['Total_Income'].fillna(df['Total_Income'].mean())
df['Rasio_Pinjaman_Pendapatan'] = df['JumlahPinjaman'] / df['Total_Income']
print("Fitur Rasio_Pinjaman_Pendapatan telah dibuat.")

print("\n--- Menangani Outlier (Winsorization) ---")
outlier_cols = ['JumlahPinjaman']

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
    print(f"Outlier pada kolom {col} dikoreksi.")

# --- KODE SELANJUTNYA DITAMBAHKAN DI SINI ---

## 5. Penanganan Data Kategorikal (One-Hot Encoding)
print("\n--- Melakukan One-Hot Encoding ---")
cols_to_encode = ['JenisKelamin', 'StatusPernikahan', 'Wiraswasta',
                  'WilayahTempatTinggal', 'JumTanggungan', 'Pendidikan']

df = df.drop(['ID_Nasabah', 'IncomeNasabah', 'IncomePasangan'], axis=1, errors='ignore')
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
print("One-Hot Encoding selesai. Jumlah kolom saat ini:", df.shape[1])

## 6. Penskalaan Fitur (Min-Max Scaler)
print("\n--- Melakukan Min-Max Scaling ---")
scaler = MinMaxScaler()
scale_cols = df.drop(columns=['StatusPinjaman'], errors='ignore').select_dtypes(include=np.number).columns.tolist()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("Min-Max Scaling selesai.")

## 7. Pembagian Data Latih dan Uji & Penyimpanan
print("\n--- Membagi Data dan Menyimpan Hasil ---")
X = df.drop('StatusPinjaman', axis=1, errors='ignore')
y = df['StatusPinjaman']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('data_latih_scaled.csv', index=False)
test_df.to_csv('data_uji_scaled.csv', index=False)

print("Data telah dibagi menjadi Latih (80%) dan Uji (20%).")
print("File data latih dan uji tersimpan di 'data_latih_scaled.csv' dan 'data_uji_scaled.csv'.")
print("\n=== PROSES PREPROCESSING & FEATURE ENGINEERING SELESAI ===")
