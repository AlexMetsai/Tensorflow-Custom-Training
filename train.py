# Alexandros I. Metsai
# alexmetsai@.gmail.com

import tensorflow as tf
import numpy as np
import time

from tensorflow import keras
from tensorflow.keras import layers


# Model definition.
inputs = keras.Input(shape=(784,), name='digits')
x1 = layers.Dense(64, activation='relu')(inputs)
x2 = layers.Dense(64, activation='relu')(x1)
outputs = layers.Dense(10, name='predictions')(x2)
model = keras.Model(inputs, outputs=outputs)

# Define optimizer and loss function.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Load training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Create validation set from last 10,000 samples.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare datasets for tensorflow.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
