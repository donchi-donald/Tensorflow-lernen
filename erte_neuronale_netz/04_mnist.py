import tensorflow as tf
from tensorflow.python.keras import Model, layers
from tensorflow.keras.datasets import mnist
import numpy as np

#define Parameters

num_classes = 10 #wir haben ziffer von 0 bis 9
num_featues = 28*28
n_hidden_1 = 128
n_hidden_2 = 256

#prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data() #die trainingsdaten sind daten, die wir trainieren. Die neuronen dürfen die testdaten nie sehen, nicht trainingsdaten mit testdaten mischen
x_train = np.array(x_train, np.float32)
x_test = np.array(x_test, np.float32)
print(len(x_train[0][0])) #0 0 die weiße pixel 28*28pixel
exit()



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




#Loss/Fehler-Funktion


#Accuracy Metric (Wie gut die Daten sind)



#Gradient descent/Optimization:( Der Lernalgorithmus/Wie optimiere mein code damit er lernt)


#Main


#Plot 