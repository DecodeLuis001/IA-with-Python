from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Datos ficticios de ejemplo (altura, peso, género)
# 0: Hombre, 1: Mujer
datos = np.array([
    [175, 70, 0],
    [160, 55, 1],
    [180, 80, 0],
    [155, 50, 1],
    [165, 65, 0],
    [150, 45, 1],
    # ... puedes agregar más datos aquí
])

# Separar los datos en características (X) y etiquetas (y)
X = datos[:, :-1]  # Altura y peso
y = datos[:, -1]   # Género

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de árbol de decisiones
modelo_arbol = DecisionTreeClassifier()

# Entrenar el modelo
modelo_arbol.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predicciones = modelo_arbol.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print(f'Precisión del modelo: {precision}')

# Ejemplo de predicción individual
nueva_medida = np.array([[170, 75]])  # Inserta tus propias medidas aquí
prediccion_nueva_medida = modelo_arbol.predict(nueva_medida)
genero_predicho = "Hombre" if prediccion_nueva_medida == 0 else "Mujer"
print(f'Predicción para la nueva medida: {genero_predicho}')
