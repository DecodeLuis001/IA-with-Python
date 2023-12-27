import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generar datos de ejemplo
np.random.seed(42)
X = np.random.rand(100, 2)  # Dos características para cada ejemplo
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Clasificación binaria (suma de características)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo de red neuronal
model = Sequential()
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Precisión del modelo: {accuracy}")
