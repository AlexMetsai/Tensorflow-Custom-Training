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
