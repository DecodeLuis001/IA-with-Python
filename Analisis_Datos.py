import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def regresion_logistica_y_probabilidad(X, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las características (opcional pero recomendado para regresión logística)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Construir y entrenar el modelo de regresión logística con scikit-learn
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred_prob = modelo.predict_proba(X_test)[:, 1]

    # Mostrar la probabilidad de ocurrencia del suceso
    for prob in y_pred_prob:
        print(f"Probabilidad de ocurrencia del suceso: {prob:.4f}")

if __name__ == "__main__":
    # Solicitar al usuario que ingrese datos para realizar la regresión logística
    num_muestras = int(input("Ingrese el número de muestras: "))
    num_caracteristicas = int(input("Ingrese el número de características: "))

    # Generar datos de ejemplo
    X = np.random.rand(num_muestras, num_caracteristicas)
    y = np.random.randint(2, size=num_muestras)  # Generar etiquetas binarias (0 o 1)

    # Realizar regresión logística y mostrar la probabilidad de ocurrencia del suceso
    regresion_logistica_y_probabilidad(X, y)
