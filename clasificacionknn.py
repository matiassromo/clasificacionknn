import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


''' Interpretación: El modelo K-NN aplicado con las variables radius_mean y 
texture_mean permitió clasificar de forma visual y precisa los tumores 
entre benignos y malignos. La frontera de decisión generada muestra 
una separación clara entre ambas clases, con pocos errores en los datos de prueba. 
Esto evidencia que K-NN es eficaz para problemas de clasificación médica con 
variables representativas. Sin embargo, su desempeño puede variar si no se 
escogen bien las características o si los datos no están normalizados. '''


# Cargar dataset
df = pd.read_csv(r'C:\Users\Usuario\Desktop\PC\UDLA\SEMESTRE 8\INTELIGENCIA ARTIFICIAL I\clasificacionknn\data.csv')

# Preprocesamiento
df = df.drop(columns=['id', 'Unnamed: 32'])  # Eliminar columnas innecesarias
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])  # M=1, B=0

# Selección de 2 características para graficar (puedes cambiarlas)
X = df[['radius_mean', 'texture_mean']]
y = df['diagnosis']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento con KNN
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, y_train)

# Crear malla para graficar fondo de decisión
h = 0.1
x_min, x_max = X['radius_mean'].min() - 1, X['radius_mean'].max() + 1
y_min, y_max = X['texture_mean'].min() - 1, X['texture_mean'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Graficar frontera de decisión + puntos
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ListedColormap(["#FF0000", "#00AA00"])
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Datos de entrenamiento y prueba
plt.scatter(X_train['radius_mean'], X_train['texture_mean'], c=y_train, cmap=cmap_bold, edgecolor='k', label='Entrenamiento')
plt.scatter(X_test['radius_mean'], X_test['texture_mean'], c=y_test, cmap=cmap_bold, edgecolor='k', marker='*', s=150, label='Prueba')

# Etiquetas
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.title("Frontera de Decisión - KNN (k=5)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
