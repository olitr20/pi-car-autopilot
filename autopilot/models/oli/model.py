from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os


class Model:

    saved_model = 'ResNet34.keras'

    def __init__(self):
        self.model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model))
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
        angle, speed = self.model.predict(np.array([image]))[0]
        # Training data was normalised so convert back to car units
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed


# class Model:

#     saved_model = 'ResNet34_Aug_Bright_fc1000.keras'

#     def __init__(self):
#         self.model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model))
#         self.model.summary()

#     def preprocess(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = tf.convert_to_tensor(image, dtype=tf.uint8)
#         image = tf.image.resize(image, (224, 224))
#         image = image / 255.0
#         image = tf.image.random_brightness(image, max_delta=0.2)
#         image = np.array([image])
#         image = image[0, :, :, :]
#         return image

#     def predict(self, image):
#         image = self.preprocess(image)
#         angle, speed = self.model.predict(np.array([image]))[0]
#         # Training data was normalised so convert back to car units
#         angle = 80 * np.clip(angle, 0, 1) + 50
#         speed = 35 * np.clip(speed, 0, 1)
#         return angle, speed
