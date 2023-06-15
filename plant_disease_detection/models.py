import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json


class DiseasePrediction:
    def __init__(self,
                model_path="20230612_01/plant_disease",
                class_indices_path="class_indices.json"):
        self.model_path = model_path
        self.class_indices_path = class_indices_path
        
        # load model
        self.model = tf.keras.models.load_model(self.model_path)
        
        # import indices
        with open(self.class_indices_path, "r") as f:
            self.class_indices = json.load(f)
            
    def predict(self, 
                image_path,
                image_size=(180, 180)):
        # load image
        image = tf.keras.utils.load_img(image_path, target_size=image_size)
        plt.imshow(image)
        input_arr = tf.expand_dims(tf.keras.utils.img_to_array(image), axis=0)

        # predict image
        prediction = self.model.predict(input_arr)
        prediction_idx = np.argmax(prediction)

        # get the value
        return self.class_indices[str(prediction_idx)]