from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Iris como ejemplo
iris = load_iris()
X = iris.data
y = iris.target

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
nueva_muestra = [[5.0, 3.0, 4.0, 1.5]]  # Inserta tus propios valores aquí
prediccion_nueva_muestra = modelo_arbol.predict(nueva_muestra)
print(f'Predicción para la nueva muestra: {prediccion_nueva_muestra}')
