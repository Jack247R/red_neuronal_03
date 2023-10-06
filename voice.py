import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos desde el archivo CSV
data = pd.read_csv("datos.csv")

# Filtrar los datos hasta 2012/8/5
data = data[data["Año"] < 2012]
data = data[data["Mes"] < 8]
data = data[data["Dia"] <= 5]

# Obtener los valores de ozono
ozono = data["O3"].values

# Escalar los valores de ozono a un rango de [0, 1]
scaler = MinMaxScaler()
ozono = scaler.fit_transform(ozono.reshape(-1, 1))

# Preparar los datos para la red neuronal
X = []
y = []

# Utilizaremos los últimos 30 días para predecir los próximos 10 días
look_back = 30
look_forward = 10

for i in range(len(ozono) - look_back - look_forward):
    X.append(ozono[i : i + look_back])
    y.append(ozono[i + look_back : i + look_back + look_forward])

X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear el modelo de red neuronal
model = keras.Sequential()
model.add(keras.layers.LSTM(50, input_shape=(look_back, 1)))
model.add(keras.layers.Dense(look_forward))
model.compile(loss="mean_squared_error", optimizer="adam")

# Entrenar la red neuronal
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluar el modelo en datos de prueba
loss = model.evaluate(X_test, y_test)
print(f"Pérdida en datos de prueba: {loss}")

# Realizar predicciones para los próximos 10 días
last_30_days = ozono[-look_back:]
predicted_ozono = []

for _ in range(look_forward):
    next_day = model.predict(last_30_days.reshape(1, look_back, 1))
    predicted_ozono.append(next_day[0][0])
    last_30_days = np.roll(last_30_days, shift=-1)
    last_30_days[-1] = next_day[0][0]

# Invertir la escala de predicción a los valores originales
predicted_ozono = scaler.inverse_transform(np.array(predicted_ozono).reshape(-1, 1))

# Imprimir las predicciones
print(f"Predicciones para los próximos 10 días: {predicted_ozono}")
# Datos observados
observados = [17.88, 21.64, 25.03, 25.03, 23.25, 16.63, 10.1, 6.71, 4.2, 2.35]

# Sumar 10 a cada valor en los dos arrays
observados = np.array(observados) + 10
predicted_ozono = np.array(predicted_ozono) + 10


def mape(observed, predicted):
    return np.mean(np.abs((observed - predicted) / observed)) * 100


mapes = [mape(observados[i], predicted_ozono[i]) for i in range(len(observados))]

print("\nMAPE por línea:")
for i, mape_value in enumerate(mapes):
    print(f"Línea {i + 1}: {mape_value:.2f}%")
