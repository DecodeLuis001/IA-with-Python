import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def regresion_logistica_y_estadisticas(X, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las características (opcional pero recomendado para regresión logística)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Añadir constante para el término independiente
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Construir y entrenar el modelo de regresión logística con statsmodels
    modelo = sm.Logit(y_train, X_train)
    resultados = modelo.fit()

    # Imprimir las estadísticas del modelo
    print(resultados.summary())

    # Hacer predicciones en el conjunto de prueba
    y_pred = resultados.predict(X_test)

    # Convertir probabilidades a etiquetas binarias (0 o 1)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calcular la precisión del modelo
    precision = np.mean(y_pred_binary == y_test)
    print(f"Precisión del modelo: {precision}")

if __name__ == "__main__":
    # Solicitar al usuario que ingrese datos para realizar la regresión logística
    num_muestras = int(input("Ingrese el número de muestras: "))
    num_caracteristicas = int(input("Ingrese el número de características: "))

    # Generar datos de ejemplo
    X = np.random.rand(num_muestras, num_caracteristicas)
    y = np.random.randint(2, size=num_muestras)  # Generar etiquetas binarias (0 o 1)

    # Realizar regresión logística y mostrar estadísticas
    regresion_logistica_y_estadisticas(X, y)
