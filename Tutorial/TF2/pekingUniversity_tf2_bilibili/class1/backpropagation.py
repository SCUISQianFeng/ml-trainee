import tensorflow as tf
import numpy as np

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epochs = 40
LR_BASE = 0.2
LR_DECAY = 0.99
LR_STEP = 0.1

for epoch in range(epochs):
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square( w +1)
    grads = tape.gradient(target=loss, sources=w)
    w.assign_sub(lr * grads)
    print("After {} epoch. w is {} , lr is {}".format(epoch, w.numpy(), lr))

