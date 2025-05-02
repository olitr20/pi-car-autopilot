import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model as kerasModel
from keras.optimizers import AdamW

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

batch_size = 16
res = 256
initial_learn_rate = 1e-4

def build_initial_model():
    # Instantiate base model
    base_model = keras.applications.EfficientNetV2B2(
        include_top = False,
        weights = "imagenet",
        input_tensor = None,
        input_shape = (res, res, 3),
        pooling = 'avg',
        include_preprocessing=True,
        name="efficientnetv2-b2",
    )
    
    # Freeze model weights
    base_model.trainable = False
    
    # Build custom input layer
    inputs = Input(shape=(res, res, 3))
    
    # Ensure model runs forward passes in inference mode
    x = base_model(inputs, training=False)
        
    # Cerate two output heads for steering angle and speed
    s = Dense(128, activation='relu')(x)
    s = BatchNormalization()(s)
    s = Dropout(0.3)(s)
    speed_output = Dense(1, activation = 'sigmoid', dtype='float32', name = 'speed')(s)

    h = Dense(128, activation='relu')(x)
    h = BatchNormalization()(h)
    h = Dropout(0.3)(h)
    steering_output = Dense(1, activation = 'sigmoid', dtype='float32', name = 'steering')(h)


    # Build the model
    model = kerasModel(inputs = inputs,
                       outputs = [steering_output, speed_output],
                       name = 'efficientnetv2-b2-full')
    
    # Reweight output heads
    λ_steering = 1.0
    λ_speed = 0.1

    # Compile model
    model.compile(optimizer = AdamW(initial_learn_rate,
                                    weight_decay=1e-4),
                  loss = {'steering': 'mse',
                          'speed': 'binary_crossentropy'},
                  loss_weights={'steering': λ_steering,
                                'speed': λ_speed},
                  metrics = {'steering': ['mae',
                                          tf.keras.metrics.R2Score(name='r2')],
                             'speed': ['accuracy']})
    
    return model

class Model:

    saved_weights = 'EfficientNet_256.h5'

    def __init__(self):
        self.model = build_initial_model()
        weights_path = os.path.join(os.path.dirname(__file__), self.saved_weights)
        self.model.load_weights(weights_path)
        self.model.summary()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize(image, (res, res))
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