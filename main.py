import streamlit as st
import pandas as pd
import sklearn
import joblib

# Muat model dari file .pkl
loaded_model = joblib.load('model.pkl')

# Judul halaman
st.title("Prediksi Penjualan Produk")

# Teks sederhana
st.write("Ini adalah aplikasi untuk memprediksi penjualan produk menggunakan model regresi linear.")

# Input teks dari pengguna
number_input = st.number_input("Harga Produk:")
jabatan_input = st.number_input("Jabatan (encoded):")
user_input = st.text_input("Masukkan nama Anda:")

# Contoh data baru dengan 3 fitur
new_data = [[ 1,1,number_input]]  # Misalnya: harga_produk = 500, jabatan_encoded = 3

# Menampilkan pesan sesuai input
if user_input:
    st.write(f"Halo, {user_input}! Selamat datang di Streamlit.")

# Tombol interaktif
if st.button("Prediksi"):
    # Lakukan prediksi menggunakan model yang dimuat
    prediction = loaded_model.predict(new_data)
    st.write(f"Prediksi Penjualan:{prediction}")