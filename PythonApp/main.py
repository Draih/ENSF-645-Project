import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load our dataset here. Need to find out if we need to cut down to subsets of the dataset or which datasets we should / shouldn't do.

# Training parameters
learning_rate = 1e-4
num_steps = 10
batch_size = 5

# Network parameters
num_hidden_layers = 0 # Todo: Real value
latent_d = 0 # Todo: Real value
num_input = 6 # This could be totally wrong, I think our vector space is 6

# Something about an encoder.
# Something about a decoder.

# Something about a loss function.
# Something about an optimizer.

# Something about an initializer.

with tf.Session() as sess:
    sess.run(init)

    # Something about training.

    # Something about testing.

# Visualization stuff should go here.