import tensorflow as tf

hello_world = tf.constant("hello tensorflow")
print(hello_world.numpy())
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())