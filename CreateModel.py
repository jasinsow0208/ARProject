import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle


# Contoh dataset
data = {
    'kategori_produk': ['Elektronik', 'Pakaian', 'Makanan', 'Elektronik', 'Pakaian', 'Makanan'],
    'harga_produk': [500, 300, 200, 400, 250, 150],
    'penjualan': [100, 80, 120, 110, 90, 130]
}

# Konversi ke DataFrame
df = pd.DataFrame(data)
 
# Pisahkan fitur dan target
X = df[['kategori_produk', 'harga_produk']] 
y = df['penjualan']

# Inisialisasi BinaryEncoder
binary_encoder = BinaryEncoder(cols=['kategori_produk'])

# Transformasi data
X_encoded = binary_encoder.fit_transform(X)

print(X_encoded)

# Bagi data menjadi training set dan test set
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Inisialisasi model regresi linear
model = LinearRegression()  
# Latih model
model.fit(X_train, y_train)
# Prediksi pada test set
y_pred = model.predict(X_test)
# Evaluasi model
mse = mean_squared_error(y_test, y_pred)    
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Simpan model ke file .pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

