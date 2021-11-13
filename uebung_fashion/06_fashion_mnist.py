import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(train_x[0], cmap='gray')
plt.show()