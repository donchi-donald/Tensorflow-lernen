import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

layers = keras.layers

#define Parameters


#prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.


#create Model, (wie unsere Neuron aussieht, wie viel Schichte es gibt)
model = keras.models.Sequential([#28*28
    layers.Reshape((28,28,1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',#Loss/Fehler-Funktion
    metrics=['accuracy']#Accuracy Metric (Was wollen wir haben)
)

#Gradient descent/Optimization:( Der Lernalgorithmus/Wie optimiere mein code damit er lernt)
model.fit(x_train, y_train, epochs=10)

#predict test data
model.evaluate(x_test, y_test)


#Plot
prediction = model(x_test[:1])
plt.imshow(np.reshape(x_test[0], [28,28]), cmap='gray')
plt.show()
print("Prediction", np.argmax(prediction.numpy()))