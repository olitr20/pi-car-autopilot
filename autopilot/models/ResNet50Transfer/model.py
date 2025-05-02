import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model as kerasModel

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

batch_size = 16
res = 128
initial_learn_rate = 1e-4
transfer_learn_rate = 1e-6

def build_initial_model(verbose = False):
    # Instantiate base model
    base_model = keras.applications.ResNet50V2(
        include_top = False,
        weights = 'imagenet',
        input_tensor = None,
        input_shape = (res, res, 3),
        pooling = None,
        name = 'resnet50v2_base',
    )
    
    # Freeze model weights
    base_model.trainable = False

    # Summarise base model
    if verbose == True:
        print('\nSummary of Base Model:\n')
        base_model.summary(show_trainable = True)
    
    # Build custom input layer
    inputs = Input(shape=(res, res, 3))
    
    # Ensure model runs forward passes in inference mode
    x = base_model(inputs, training=False)
    
    # Convert features of `base_model.output_shape[1:]` to vectors
    x = GlobalAveragePooling2D()(x)
    
    # Cerate two output heads for steering angle and speed
    steering_output = Dense(1, activation = 'sigmoid', dtype='float32', name = 'steering')(x)
    speed_output = Dense(1, activation = 'sigmoid', dtype='float32', name = 'speed')(x)
    
    # Build the model
    model = kerasModel(inputs = inputs,
                  outputs = [steering_output, speed_output],
                  name = 'resnet50v2')
    
    # Compile model
    model.compile(optimizer = tf.keras.optimizers.Adam(initial_learn_rate),
                  loss = {'steering': 'mse',
                          'speed': 'binary_crossentropy'},
                  metrics = {'steering': ['mae',
                                          tf.keras.metrics.R2Score(name='r2')],
                             'speed': ['accuracy']})
    
    return model

class Model:

    saved_weights = 'resnet50v2_transfer.h5'

    def __init__(self):
        self.model = build_initial_model()
        weights_path = os.path.join(os.path.dirname(__file__), self.saved_weights)
        self.model.load_weights(weights_path)
        self.model.summary()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize(image, (res, res))
        image = tf.cast(image, tf.float32) / 255.0
        image = np.array([image])
        image = image[0, :, :, :]
        return image

    def predict(self, image):
        image = self.preprocess(image)
        batch = np.expand_dims(image, axis=0)
        steering_arr, speed_arr = self.model.predict(batch)
        angle = float(steering_arr[0][0])
        speed = float(speed_arr[0][0])
        # Training data was normalised so convert back to car units
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed