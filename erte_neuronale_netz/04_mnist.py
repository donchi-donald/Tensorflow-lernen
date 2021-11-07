import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import Model, layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plst


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#define Parameters
#datenparameter
num_classes = 10 #wir haben ziffer von 0 bis 9
num_features = 28*28

#jedes bild hat sein egenes Fehler
#trainingsparameter
batch_size = 256
lerning_rate = 0.1 # oder etwas kleinerer
training_steps = 2560
step = 100

n_hidden_1 = 128
n_hidden_2 = 256

#prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data() #die trainingsdaten sind daten, die wir trainieren. Die neuronen dürfen die testdaten nie sehen, nicht trainingsdaten mit testdaten mischen
x_train = np.array(x_train, np.float32).reshape([-1, num_features]) #wir löschen eine dimension, damit die bilder gut gespeichert wurde
x_test = np.array(x_test, np.float32).reshape([-1, num_features])
x_train, x_test = x_train/255., x_test/255. #neurone arbeite besser im bereich 0 bis 1, bilder sind von 0 bis 255 pixel
#print(x_train[0]) #0 0 die weiße pixel 28*28pixel, 0-255, wie viel Prozent davon sind farbig?

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat()
train_data = train_data.shuffle(1100).batch(batch_size).prefetch(1) #mit shuffle wollen wir die Daten auseinande mischen




#create Model, (wie unsere Neuron aussieht, wie viel Schichte es gibt)
class Mnist_Neural_Network(Model):
    def __init__(self):
        super(Mnist_Neural_Network, self).__init__()
        #schichten, wir benutzen die dense schicht, weil es einfach ist, was trainiert neurone und sucht man das beste aus
        self.fc1 = layers.Dense(n_hidden_1,
                                activation=tf.nn.relu)
        self.fc2 = layers.Dense(n_hidden_2,
                                activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)

    #training=none wir sind am Anfang nicht in Trainingsmodus, wir evaluieren
    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.out(x)
        if not training:
            x = tf.nn.softmax(x)
        return x

network = Mnist_Neural_Network()


#Loss/Fehler-Funktion
def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

#Accuracy Metric (Wie gut die Daten sind)
def accuracy(y_predicted, y_true):
    y_true = tf.cast(y_true, tf.int64)
    correct = tf.equal(tf.argmax(y_predicted, -1), y_true)
    return tf.reduce_mean(tf.cast(correct, tf.float32), axis=-1)


#Gradient descent/Optimization:( Der Lernalgorithmus/Wie optimiere mein code damit er lernt)
optimizer = tf.optimizers.SGD(lerning_rate)
def gradient_descent(x, y):
    with tf.GradientTape() as grad:
        prediction = network(x, training=True)
        loss = cross_entropy_loss(prediction, y)
    weights = network.trainable_variables
    gradient = grad.gradient(loss, weights)
    optimizer.apply_gradients(zip(gradient, weights))
    return loss, prediction

#Main
for i, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    loss, pred = gradient_descent(batch_x, batch_y)

    if i % step == 0:
        print(f"step: {i}, Loss:{loss}, Accuracy: {accuracy(pred, batch_y)}")

#PRedict test data
prediction = network(x_test, training=False)
print(f"Accuracy on test-data: {accuracy(prediction, y_test)}")

#Plot
for i in range(10):
    plt.imshow(np.reshape(x_test[i], [28,28]), cmap='gray')
    plt.show()
    print("Prediction", np.argmax(prediction.numpy()[i]))
