import tensorflow as tf

x = tf.constant(42)
y = tf.constant(1337)

add = tf.add(x, y)

print(add.numpy())
a = tf.constant([1., 2., 3., 4.])
a1 = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [-1, 1, -1, 1]])
a2 = tf.constant([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
z = tf.constant([[[1], [2]], [[2], [3]]])
print(a)
print(a1)
print(a2)
print(z)
print(tf.reduce_mean(a))
print(tf.matmul(a2, a1))

# command: strg + alt + L
