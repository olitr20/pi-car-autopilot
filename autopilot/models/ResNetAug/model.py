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
    AveragePooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model as KerasModel

import numpy as np
import cv2
import os


def identity_block(x, k, F, stage, block):
    conv_base_name = 'Conv_' + str(stage) + block + '_branch'
    bn_base_name = 'BN_' + str(stage) + block + '_branch'

    x_residual = x

    x = Conv2D(filters = F, kernel_size = (1, 1), strides = (1, 1),
               padding='valid', name = conv_base_name + '2a',
               kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis=3, name = bn_base_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = F, kernel_size = (k, k), strides = (1, 1),
               padding='same', name = conv_base_name + '2b',
               kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis=3, name = bn_base_name + '2b')(x)

    x = Add()([x_residual, x])
    x = Activation('relu')(x)
    return x

def conv_block(x, k, F, stage, block, s = 2):
    conv_base_name = 'Conv_' + str(stage) + block + '_branch'
    bn_base_name = 'BN_' + str(stage) + block + '_branch'

    x_residual = x

    x = Conv2D(filters = F, kernel_size = (1, 1), strides = (s, s),
               name = conv_base_name + '2a',
               kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis=3, name = bn_base_name + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters = F, kernel_size = (k, k), strides = (1, 1),
               padding = 'same', name = conv_base_name + '2b',
               kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis=3, name = bn_base_name + '2b')(x)

    x_residual = Conv2D(filters = F, kernel_size = (1, 1), strides = (s, s),
               padding = 'valid', name = conv_base_name + '1',
               kernel_initializer = glorot_uniform(seed = 0))(x_residual)
    x_residual = BatchNormalization(axis=3, name = bn_base_name + '1')(x_residual)

    x = Add()([x_residual, x])
    x = Activation('relu')(x)
    return x

def build_ResNet50(input_shape = (224, 224, 3)):
    x_input = Input(shape = input_shape)
    x = ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2),
               name = 'Conv_1', kernel_initializer = glorot_uniform(seed = 0))(x)
    x = BatchNormalization(axis = 3, name = 'BN_1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides = (2, 2))(x)

    x = identity_block(x, k = 3, F = 64, stage = 2, block = 'a')
    x = identity_block(x, k = 3, F = 64, stage = 2, block = 'b')
    x = identity_block(x, k = 3, F = 64, stage = 2, block = 'c')

    x = conv_block(x, k = 3, F = 128, stage = 3, block = 'a', s = 2)
    x = identity_block(x, k = 3, F = 128, stage = 3, block = 'b')
    x = identity_block(x, k = 3, F = 128, stage = 3, block = 'c')
    x = identity_block(x, k = 3, F = 128, stage = 3, block = 'd')

    x = conv_block(x, k = 3, F = 256, stage = 4, block = 'a', s = 2)
    x = identity_block(x, k = 3, F = 256, stage = 4, block = 'b')
    x = identity_block(x, k = 3, F = 256, stage = 4, block = 'c')
    x = identity_block(x, k = 3, F = 256, stage = 4, block = 'd')
    x = identity_block(x, k = 3, F = 256, stage = 4, block = 'e')
    x = identity_block(x, k = 3, F = 256, stage = 4, block = 'f')

    x = conv_block(x, k = 3, F = 512, stage = 5, block = 'a', s = 2)
    x = identity_block(x, k = 3, F = 512, stage = 5, block = 'b')
    x = identity_block(x, k = 3, F = 512, stage = 5, block = 'c')

    x = AveragePooling2D(pool_size = (4, 4), name = 'avg_pool')(x)

    x = Flatten()(x)
    fc = Dense(1000, activation = 'relu', name = 'fc1000',
               kernel_initializer = glorot_uniform(seed = 0))(x)

    steering_output = Dense(1, activation = 'sigmoid', name = 'steering')(fc)

    speed_output = Dense(1, activation = 'sigmoid', name = 'speed')(fc)

    model = KerasModel(inputs = x_input, outputs = [steering_output, speed_output], name = 'ResNet50')
    return model

class Model:

    saved_weights = 'ResNet34_Aug_Bright_fc1000.h5'

    def __init__(self):
        self.model = build_ResNet50()
        weights_path = os.path.join(os.path.dirname(__file__), self.saved_weights)
        self.model.load_weights(weights_path)
        self.model.summary()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = tf.image.random_brightness(image, max_delta=0.2)
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
