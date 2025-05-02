import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.models import Model as KerasModel

import numpy as np
import cv2
import os


def identity_block(x, filter_size, kernel_size):
    # Copy input tensor to x_residual
    x_residual = x

    # Block Layer 1
    x = Conv2D(filter_size, kernel_size, padding='same')(x_residual)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Block Layer 2
    x = Conv2D(filter_size, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    # Add Residue
    x = Add()([x, x_residual])
    x = Activation('relu')(x)
    return x

def conv_block(x, filter_size, kernel_size, strides=(2, 2)):
    # Copy input tensor to x_residual
    x_residual = x

    # Block Layer 1
    x = Conv2D(filter_size, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Block Layer 2
    x = Conv2D(filter_size, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    # Add Residue
    x_residual = Conv2D(filter_size, (1, 1), strides=strides)(x_residual)

    x = Add()([x, x_residual])
    x = Activation('relu')(x)
    return x

def build_ResNet34(input_shape=(224, 224, 3)):
    # Setup Input Layer
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(x_input)

    # Initial convolution and pooling
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64

    # Add ResNet Blocks
    for i in range(4):
      if i == 0:
        x = conv_block(x, filter_size=filter_size, kernel_size=3, strides=(2,2))
        for j in range(block_layers[i]):
          x = identity_block(x, filter_size=filter_size, kernel_size=3)
      else:
        # One Residual/Convolutional block followed by Identity blocks
        # Filter size will increase by a factor of 2
        filter_size = filter_size * 2
        x = conv_block(x, filter_size=filter_size, kernel_size=3, strides=(2,2))
        for j in range(block_layers[i] - 1):
          x = identity_block(x, filter_size=filter_size, kernel_size=3)

    x = GlobalAveragePooling2D()(x)

    # Add a common dense layer
    fc = Dense(2048, activation='relu')(x)

    # Two output heads:
    # Steering angle head (regression: value between 0 and 1)
    steering_output = Dense(1, activation='sigmoid', name='steering')(fc)

    # Speed head (binary classification: 0 or 1)
    speed_output = Dense(1, activation='sigmoid', name='speed')(fc)

    model = KerasModel(inputs=x_input, outputs=[steering_output, speed_output], name = "ResNet34")
    return model


class Model:

    saved_weights = 'ResNet34.h5'

    def __init__(self):
        self.model = build_ResNet34()
        weights_path = os.path.join(os.path.dirname(__file__), self.saved_weights)
        self.model.load_weights(weights_path)
        self.model.summary()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
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