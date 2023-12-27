import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def regresion_logistica(X, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construir y entrenar el modelo de regresión logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy}")

    # Mostrar informe de clasificación
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Solicitar al usuario que ingrese datos para realizar la regresión logística
    num_muestras = int(input("Ingrese el número de muestras: "))
    num_caracteristicas = int(input("Ingrese el número de características: "))

    # Generar datos de ejemplo
    X = np.random.rand(num_muestras, num_caracteristicas)
    y = np.random.randint(2, size=num_muestras)  # Generar etiquetas binarias (0 o 1)

    # Realizar regresión logística
    regresion_logistica(X, y)
