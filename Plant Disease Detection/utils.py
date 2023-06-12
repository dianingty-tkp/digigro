import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def predict(image_path,
            model,
            indices,
            image_size=(180, 180)):
    
    # load image
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    plt.imshow(image)
    input_arr = tf.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
    
    # predict image
    prediction = model.predict(input_arr)
    prediction_idx = np.argmax(prediction)
    
    # get the value
    return indices[str(prediction_idx)]
