import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load our dataset here. Need to find out if we need to cut down to subsets of the dataset or which datasets we should / shouldn't do.

# Training parameters
learning_rate = 1e-4
num_steps = 10
batch_size = 5

##### conv net ### parameters have to get adjusted --> what means N/A at table 1 for Pool
input_image = Input(70)
c1 = Conv2D(32,kernel_size=(7, 7),
                 activation='relu',name='conv1')(input_image)                
p1 = MaxPooling2D(pool_size=(3, 3))(c1)
c2 = Conv2D(32,kernel_size=(7, 7),
                 activation='relu',name='conv1')(p1)                
p2 = MaxPooling2D(pool_size=(3, 3))(c2)
c3 = Conv2D(32,kernel_size=(3, 3),
                 activation='relu',name='conv1')(p2)                
p3 = MaxPooling2D(pool_size=(2, 2))(c3)
c4 = Conv2D(32,kernel_size=(3, 3),
                 activation='relu',name='conv1')(p3)                
p4 = MaxPooling2D(pool_size=(2, 2))(c4)
c5 = Conv2D(32,kernel_size=(3, 3),
                 activation='relu',name='conv1')(p4)                
p5 = MaxPooling2D(pool_size=(2, 2))(c5)
c6 = Conv2D(32,kernel_size=(3, 3),
                 activation='relu',name='conv1')(p5)                
p6 = MaxPooling2D(pool_size=(3, 3))(c6)
#### fully connected layers with dropout ##### parameters must be adjusted
fc1 = Dense(1024, activation='relu',name='fully1')(p6)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(1024, activation='relu',name='fully2')(d1)
d2 = Dropout(0.5)(fc2)
y = Dense(1024, activation='relu',name='fully3')(d2)
#y = Dense(num_classes, activation='softmax',name='softout')(fc3)

model = keras.models.Model(inputs=[input_image], outputs=[y])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

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