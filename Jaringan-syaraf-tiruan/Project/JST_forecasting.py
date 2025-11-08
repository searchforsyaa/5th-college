# ----------------------------------------------------------------------
# SKRIP ANALISIS FORECASTING HARGA SAHAM MENGGUNAKAN LSTM
# ----------------------------------------------------------------------
#
# Deskripsi:
# Skrip ini melakukan prediksi time series (forecasting) pada data harga
# saham (menggunakan kolom 'Close') dari file CSV yang disediakan.
#
# Model: Long Short-Term Memory (LSTM) Neural Network
# Data: 'dataset_bbri.xlsx - Sheet1.csv'
#
# Langkah-langkah:
# 1. Import library yang diperlukan
# 2. Definisikan parameter model
# 3. Muat dan proses data
# 4. Normalisasi data (MinMax Scaling)
# 5. Buat dataset sequence (helper function)
# 6. Pisahkan data training dan testing
# 7. Bentuk ulang (reshape) data untuk input LSTM
# 8. Bangun arsitektur model LSTM
# 9. Kompilasi dan latih model
# 10. Lakukan prediksi
# 11. Kembalikan data ke skala semula (Inverse Transform)
# 12. Hitung Root Mean Squared Error (RMSE)
# 13. Visualisasikan hasil prediksi
# 14. Visualisasikan loss training vs validasi
#
# ----------------------------------------------------------------------

# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------------------------------------------------
# 2. DEFINISIKAN PARAMETER
# Nama file CSV Anda
FILE_PATH = r"D:\semester-5\Jaringan-syaraf-tiruan\Project\dataset_bbri.xlsx"

# Berapa hari ke belakang yang digunakan untuk memprediksi 1 hari ke depan
LOOK_BACK = 60

# Proporsi pembagian data training (80% training, 20% testing)
TRAIN_SPLIT_RATIO = 0.8

# Parameter untuk Neural Network
EPOCHS = 100
BATCH_SIZE = 32

# ----------------------------------------------------------------------
# 3. FUNGSI HELPER UNTUK MEMBUAT SEQUENCE
def create_dataset(dataset, look_back=1):
    """
    Mengubah array nilai menjadi format dataset sequence.
    Contoh:
    dataset = [1, 2, 3, 4, 5]
    look_back = 2
    Maka hasilnya:
    dataX = [[1, 2], [2, 3], [3, 4]]
    dataY = [3, 4, 5]
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# ----------------------------------------------------------------------
# 4. MUAT DAN PROSES DATA
print(f"Memuat data dari {FILE_PATH}...")
try:
    df = pd.read_csv(
        FILE_PATH,
        usecols=['Date', 'Close'],
        parse_dates=['Date'],
        index_col='Date'
    )
except FileNotFoundError:
    print(f"Error: File '{FILE_PATH}' tidak ditemukan.")
    print("Pastikan file CSV berada di direktori yang sama dengan skrip ini.")
    exit()

# Menghapus baris dengan data NaN (jika ada)
df = df.dropna()
print("Data berhasil dimuat.")
print(df.head())

# Ambil nilai 'Close' dan ubah menjadi numpy array
dataset = df['Close'].values.reshape(-1, 1)

# ----------------------------------------------------------------------
# 5. NORMALISASI DATA
# Mengubah skala data menjadi antara 0 dan 1
# Ini penting untuk performa model neural network
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(dataset)

# ----------------------------------------------------------------------
# 6. PISAHKAN DATA TRAINING DAN TESTING
# Pisahkan data sebelum membuat sequence
train_size = int(len(scaled_dataset) * TRAIN_SPLIT_RATIO)
test_size = len(scaled_dataset) - train_size

train_data = scaled_dataset[0:train_size, :]
test_data = scaled_dataset[train_size:len(scaled_dataset), :]

print(f"Ukuran data training: {len(train_data)}")
print(f"Ukuran data testing: {len(test_data)}")

# Buat sequence
X_train, y_train = create_dataset(train_data, LOOK_BACK)
X_test, y_test = create_dataset(test_data, LOOK_BACK)

# ----------------------------------------------------------------------
# 7. BENTUK ULANG (RESHAPE) DATA
# Input LSTM memerlukan format [samples, time_steps, features]
# Saat ini data kita [samples, time_steps], jadi kita tambah 1 dimensi untuk features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Shape X_train (setelah reshape): {X_train.shape}")
print(f"Shape X_test (setelah reshape): {X_test.shape}")

# ----------------------------------------------------------------------
# 8. BANGUN ARSITEKTUR MODEL LSTM
print("Membangun model LSTM...")
model = Sequential()
# Layer LSTM pertama dengan 50 unit.
# return_sequences=True karena kita akan menambah layer LSTM lagi
model.add(LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
# Layer LSTM kedua
model.add(LSTM(50, return_sequences=False))
# Output layer (Dense) dengan 1 unit untuk prediksi harga
model.add(Dense(1))

# ----------------------------------------------------------------------
# 9. KOMPILASI DAN LATIH MODEL
# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Latih model
print(f"Mulai training model (Epochs={EPOCHS}, Batch Size={BATCH_SIZE})...")
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)
print("Training selesai.")

# ----------------------------------------------------------------------
# 10. LAKUKAN PREDIKSI
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# ----------------------------------------------------------------------
# 11. KEMBALIKAN DATA KE SKALA SEMULA (INVERSE TRANSFORM)
# Prediksi masih dalam skala 0-1, kita kembalikan ke skala harga asli
train_predict = scaler.inverse_transform(train_predict)
y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1))

test_predict = scaler.inverse_transform(test_predict)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

# ----------------------------------------------------------------------
# 12. HITUNG ROOT MEAN SQUARED ERROR (RMSE)
train_rmse = math.sqrt(mean_squared_error(y_train_orig, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test_orig, test_predict))

print(f"\n--- HASIL EVALUASI ---")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE:  {test_rmse:.2f}")

# ----------------------------------------------------------------------
# 13. VISUALISASIKAN HASIL PREDIKSI
print("Membuat plot hasil prediksi...")

# Siapkan data untuk plot
# Buat array kosong seukuran data asli, isi dengan NaN
train_predict_plot = np.empty_like(scaled_dataset)
train_predict_plot[:, :] = np.nan
# Isi bagian data training dengan hasil prediksi
train_predict_plot[LOOK_BACK:len(train_predict) + LOOK_BACK, :] = train_predict

# Lakukan hal yang sama untuk data testing
test_predict_plot = np.empty_like(scaled_dataset)
test_predict_plot[:, :] = np.nan
# Indeks awal plot testing adalah setelah data training + look_back
test_plot_start_index = train_size + LOOK_BACK
test_predict_plot[test_plot_start_index : test_plot_start_index + len(test_predict), :] = test_predict


# Plot
plt.figure(figsize=(16, 8))
# Plot data asli
plt.plot(df.index, scaler.inverse_transform(scaled_dataset), label='Data Asli (Close)')
# Plot prediksi training
plt.plot(df.index, train_predict_plot, label='Prediksi Training')
# Plot prediksi testing
plt.plot(df.index, test_predict_plot, label='Prediksi Testing')

plt.title(f'Prediksi Harga Saham (LSTM) - Test RMSE: {test_rmse:.2f}')
plt.xlabel('Tanggal')
plt.ylabel('Harga Close')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# 14. VISUALISASIKAN LOSS TRAINING VS VALIDASI
print("Membuat plot loss model...")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

print("--- Skrip Selesai ---")