import numpy as np
import cv2
import os

import tensorflow as tf
from keras.models import load_model
    

class Model:

    #saved_model = 'EfficientNetB0_128.h5'
    saved_model = 'EfficientNetB0_128_init.h5'

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__),
                                  self.saved_model)
        self.model = load_model(model_path)
        self.model.summary()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize(image, (128, 128))
        return image

    def predict(self, image):
        image = self.preprocess(image)
        batch = np.expand_dims(image, axis=0)
        steering_arr, speed_arr = self.model.predict(batch)
        angle = float(steering_arr[0][0])
        speed = float(speed_arr[0][0])
        # angle = (5 * np.round((80 * np.clip(angle, 0, 1) + 50) / 5)).astype(int)
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * int(round(np.clip(speed, 0, 1)))
        return angle, speed
