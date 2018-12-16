#from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras import backend as K
from keras.layers import Input
import numpy as np
from organize_input import YelpSpecificJSON

train_size = 200000
test_size = 40000

# init of weights I am not sure if this is right
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# Load our dataset here. Need to find out if we need to cut down to subsets of the dataset or which datasets we should / shouldn't do.
x_train, y_train, x_test, y_test = YelpSpecificJSON("yelp_academic_dataset_review.json", train_size, test_size)

print (x_train[0])

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Training parameters
learning_rate = 1e-3
epochs = 1
batch_size = 128
classes = 1 # ?
input_size = 1014

x_train = x_train.reshape(x_train.shape[0], input_size, 1)
x_test = x_test.reshape(x_test.shape[0], input_size, 1)

##### Input
input_data = Input(shape=(input_size, 1 ))

##### conv net 
# parameters adjusted like shown in table, used the small feature --> N/A means no implementation of MaxPooling layer, Christoph 11 Dec 2018
c1 = Convolution1D(256,7,activation='relu')(input_data)
p1 = MaxPooling1D(3)(c1)
c2 = Convolution1D(256,7,activation='relu')(p1)
p2 = MaxPooling1D(3)(c2)
c3 = Convolution1D(256,3,activation='relu')(p2)
c4 = Convolution1D(256,3,activation='relu')(c3)
c5 = Convolution1D(256,3,activation='relu')(c4)
c6 = Convolution1D(256,3,activation='relu')(c5)
p6 = MaxPooling1D(3)(c6)

fl1 = Flatten()(p6)

##### fully connected layers with dropout
fc1 = Dense(1024,activation='relu')(fl1)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(1024,activation='relu')(d1)
d2 = Dropout(0.5)(fc2)

##### Output
y = Dense(classes, activation='softmax')(d2)

model = keras.models.Model(inputs=[input_data], outputs=[y])

model.compile(loss=keras.losses.mean_squared_error,
              optimizer='sgd')

# Training
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('ProjectModel.h5')

# Testing

# Visualization stuff should go here.