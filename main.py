import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Lectura de datos
data = pd.read_csv('datos2.csv', header=None).values
print(f'Datos obternidos: \n{data} \n🤩🤩🤩🤩🤩🤩🤩')
print(f'El tamaño de la matriz es: {data.shape} 😱')
print(f'El tipo de dato es: {data.dtype} 🤔')


# Aplicamos la descomposición de valores singulares (SVD)
U, S, V = np.linalg.svd(data)
_, valores_singulares, _ = np.linalg.svd(data)

# Visualizamos los vectores singulares izquierdos y derechos en un gráfico 2D
plt.figure(figsize=(8, 8))

# Ajustamos el factor de escala, puesto que los valores son muy cercanos a cero
escala = 15

# Visualizamos los vectores singulares izquierdos y derechos en un gráfico 2D
plt.figure(figsize=(8, 8))

# Vectores singulares izquierdos
plt.plot([0, U[0, 0]*escala], [0, U[1, 0]*escala], color='r', label='U[:, 0]')
plt.plot([0, U[0, 1]*escala], [0, U[1, 1]*escala], color='b', label='U[:, 1]')

# Vectores singulares derechos
plt.plot([0, V[0, 0]*escala], [0, V[1, 0]*escala], color='g', label='V[0, :]')
plt.plot([0, V[0, 1]*escala], [0, V[1, 1]*escala], color='y', label='V[1, :]')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title('Vectores Singulares Izquierdos y Derechos')
plt.show()

# Crear el boxplot de los valores singulares
plt.boxplot(valores_singulares)

# Añadir título y etiquetas a los ejes
plt.title('Boxplot de Valores Singulares')
plt.xlabel('Valores Singulares')
plt.ylabel('Valores')

# Mostrar la gráfica
plt.show()

