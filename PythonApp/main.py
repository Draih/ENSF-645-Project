import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load our dataset here. Need to find out if we need to cut down to subsets of the dataset or which datasets we should / shouldn't do.

# Training parameters
learning_rate = 1e-4
num_steps = 10
batch_size = 5

# Network parameters
conv_layers=6
fully_connected_layers=3
latent_d = 0 # Todo: Real value
input_features = 70  # I think the input is 70, like written on page 3
input_features_length=1014
dropout=2
p_dropout=0.5 # dropout probability

######## init weights --> init with gaussian distribution and mean=0 and standard deviation=0.02
def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.random_normal(shape, stddev=0.02)
    return tf.Variable(initial)

######## init biases --> do we have any biases?
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

######## defining the convolution ?????
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

######## defining the pooling --> no overlapping?
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')



######## constructing the neural net ----- only the structure is right, the conv2d and maxpool2d don't make any sense yet
def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)
    conv4=
    conv4=
    conv5=
    conv5=
    conv6=
    conv6=


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc2=
    fc2=
    fc2=
    fc3=
    fc3=
    fc3=



    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

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