import pickle

# Muat model dari file .pkl
with open('model.pkl', 'rb') as file: 
    loaded_model = pickle.load(file)    

# mempredisi data baru
new_data = [[ 1,1,200]]  # Misalnya: harga_produk = 500, jabatan_encoded = 3
prediction = loaded_model.predict(new_data)     
print(f"Prediksi Penjualan: {prediction}")
