#aditya singh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Dataset (Temperature, Humidity)
X = np.array([[30, 70],
              [25, 80],
              [27, 60],
              [31, 65],
              [23, 85],
              [28, 75],[30,81],[35,79],[20,40],[35,75],[36,82],[29,80],[21,65],[24,64],[35,89]])


y = np.array([0, 1, 0, 0, 1, 1])


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)


new_point = np.array([[26, 78]])
prediction = knn.predict(new_point)[0]


plt.figure(figsize=(7, 5))


plt.scatter(X[y == 0, 0], X[y == 0, 1], s=100)

plt.scatter(X[y == 1, 0], X[y == 1, 1], s=100)

plt.scatter(new_point[0, 0],
            new_point[0, 1],
            marker='*',
            s=300,
            color='red',
            label="New Prediction")

plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("KNN Weather Classification")
plt.grid(alpha=0.3)
plt.show()

if prediction == 0:
    print("Predicted Weather: Sunny ☀️")
else:

    print("Predicted Weather: Rainy 🌧️")
