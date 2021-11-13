import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

fashion_mnist = keras.datasets.fashion_mnist

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Reshape((28,28,1)),
    keras.layers.Conv2D(32,3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10) #denn wir 10 Klasse haben
])
#bugs resolve
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=10)

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)

#Plot
prediction = model(test_x[:1])
plt.imshow(np.reshape(test_x[0], [28,28]), cmap='gray')
plt.show()
print("Prediction", np.argmax(prediction.numpy()))



