import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def regresion_lineal(x, y):
    # Dividir los datos en conjunto de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Crear el modelo de regresión lineal
    modelo = LinearRegression()

    # Entrenar el modelo
    modelo.fit(x_train.reshape(-1, 1), y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = modelo.predict(x_test.reshape(-1, 1))

    # Imprimir los coeficientes de la regresión
    print("Coeficiente (pendiente):", modelo.coef_[0])
    print("Intercepto:", modelo.intercept_)

    # Graficar los datos y la línea de regresión
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Variable independiente')
    plt.ylabel('Variable dependiente')
    plt.title('Regresión Lineal')
    plt.show()

if __name__ == "__main__":
    # Generar datos de ejemplo
    np.random.seed(42)
    x = np.random.rand(100, 1) * 10  # Variable independiente
    y = 2 * x + 1 + np.random.randn(100, 1) * 2  # Relación lineal con ruido

    # Calcular regresión lineal
    regresion_lineal(x, y)
