#Aditya singh
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.title("🌤 KNN Weather Prediction App")

# Dataset
X = np.array([[30, 70],
              [25, 80],
              [27, 60],
              [31, 65],
              [23, 85],
              [28, 75],[30,81],[35,79],[20,40],[35,75],[36,82],[29,80],[21,65],[24,64],[35,89]])

y = np.array([0, 1, 0, 0, 1, 1,0,0,1,0,0,1,1,1,0])

# Train Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# User Inputs
temperature = st.slider("Select Temperature (°C)", 20, 40, 26)
humidity = st.slider("Select Humidity (%)", 50, 100, 78)

new_point = np.array([[temperature, humidity]])
prediction = knn.predict(new_point)[0]

# Plot Graph
fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(X[y == 0, 0], X[y == 0, 1], s=100, label="Sunny")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=100, label="Rainy")
ax.scatter(temperature, humidity, marker='*', s=300, color='red', label="New Prediction")

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
ax.set_title("KNN Weather Classification")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)

# Show Prediction
if prediction == 0:
    st.success("Predicted Weather: ☀️ Sunny")
else:

    st.info("Predicted Weather: 🌧️ Rainy")
