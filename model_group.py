# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:20:13 2021

Script for two different model configurations, which can be
trained with automatic (augmentation layer) and 
manuel data augmentation (data which is already augmented).
The scenes need to can be entered for example as an array in x_train of size
(number of scene, 224,224, 1). The number of pixels can be varied also.
In this case the test data should have the same size as the training data!
If the scenes are already augmented and ready for the training,
AUGMENTATION_GIVEN = 1, otherwise use AUGMENTATION_GIVEN = 0.

Both model configurations can be used for any data.
For this work were these two configuration used,
because they showed the best results for the respective dataset.


@author: Christopher Reichel
"""
# %%
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from pathlib import Path


# %% Choose type of training data and change DATAPATH
# TODO: Change DATAPATH and set if Augmentation is given or not

DATAPATH = os.path.join('C:/Users/chris/Desktop/Studium/Masterarbeit/bs-data/')
# DATAPATH = os.path.join('/gpfs/possnerhsmfs/danker/home/Prediction')
DATAPATH_SAVE = os.path.join("C:/Users/chris/Desktop/Studium/Masterarbeit/results/jessi")
FILENAME = os.path.join("test")
# if training data is already augmented = 1, else = 0
Augmentation_GIVEN = 1

# %% Define constant variables
# Open = 0, Closed = 1, NoMCC = 2
CATEGORIES = ['Open', 'Closed', 'NoMCC']
FIG_FOR = 'png'

# %% load input data

# load the test dataset
pickle_in = open(os.path.join(DATAPATH, 'X_og.pickle'), 'rb')
X = pickle.load(pickle_in)

# load the categories for the training data set
pickle_in = open(os.path.join(DATAPATH, 'y_og.pickle'), 'rb')
y = pickle.load(pickle_in)


# %% set values

FILLVALUES = -999

X[X == FILLVALUES] = 0

y = np.array(y)

y = to_categorical(y)

# %% Set data augmentations

if Augmentation_GIVEN == 0:
    data_augmentation = tf.keras.Sequential([
                        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                        layers.experimental.preprocessing.RandomRotation(0.4),
                        layers.experimental.preprocessing.RandomContrast(factor=0.7),
                        layers.experimental.preprocessing.RandomZoom(height_factor=(-0.1, -0.2), width_factor=None, fill_mode="reflect"),
                        layers.experimental.preprocessing.RandomCrop(height=163, width=163),
                        layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2)
                                           ])

#%% Model configurations for the augmented data
# %% Model configurations for the augmented data


if Augmentation_GIVEN == 0:
    model = Sequential()

    # add data augmentation layer
    model.add(data_augmentation)

    # add Conv2D layer with 32 filtersize and 3x3 kernel
    model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
    # add  Activation function "relu" after the Convolutional layer
    model.add(Activation('relu'))
    # add MaxPooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # add second Conv2D layer with 64 filtersize
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # add third Conv2D layer with 128 filtersize
    model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # add FLatten layer with with 2 dense layer (64 and 3) and activation function "relu" and "softmax"
    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # add "Adam optimizer" with learningrate = 0.001 and loss function "categorical_crossentropy"
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # set filepaths and name to save the model as hdf5 at a certain point

    # set file path
    filepath = Path(DATAPATH_SAVE)
    # set filename
    filename = Path(FILENAME)
    # set file format
    file = Path(filepath/(str(filename) + ".hdf5"))

    # add checkpoint where the model will be saved
    # for monitor = "val_accuracy" and mode = "max" it will be saved at each point where the validation accuracy has been the highest
    checkpoint = ModelCheckpoint(filepath=file,
                                 monitor="val_accuracy",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="max")

    callbacks = [checkpoint]
    print("hi")


#%% Model configurations for already augmented dataset

if Augmentation_GIVEN == 1:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:], kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:], kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())  

    model.add(Dense(64,))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(3))

    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    filepath = Path(DATAPATH_SAVE)
    filename = Path(FILENAME)
    file = Path(filepath/(str(filename) + ".hdf5"))

    print("bye")

    checkpoint = ModelCheckpoint(filepath=file,
                                 monitor="val_accuracy",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="max")

    callbacks = [checkpoint]

# %% model fit

# start training the model
history = model.fit(X, y, batch_size=20, epochs=2, validation_split=0.3, callbacks=callbacks)


# %% set history and plot Paths and save the history of the model

# set full file path for save the history (accuracy and loss data)
file_history_full = Path(filepath/(str(filename) + ".npy"))
np.save(file_history_full, history.history)

# set full file path to save the plots
file_plot_full = Path(filepath/(str(filename) + ".PNG"))

# use this to load the history
history = np.load(file_history_full, allow_pickle='TRUE').item()

# %% Plot accuracy and loss

plt.subplot(221)
plt.title("Loss")
plt.plot(history["loss"], label="training")
plt.legend(loc="best")
plt.plot(history["val_loss"], label="validation")
plt.legend(loc="best")
plt.legend
plt.xlabel("Epochs")

plt.subplot(222)
plt.title("Accuracy")
plt.plot(history["accuracy"], label="training")
plt.legend(loc="best")
plt.plot(history["val_accuracy"], label="validation")
plt.legend(loc="best")
plt.legend
plt.xlabel("Epochs")

plt.show
plt.savefig(file_plot_full)
