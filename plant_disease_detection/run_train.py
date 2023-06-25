import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

import os

# Define parameters
image_size = (180, 180)
input_shape = (180, 180, 3)
batch_size = 32
n_cat=38

epochs = 25
output_dir = "20230625_01"
steps_per_epoch = None #1758
validation_steps = None #440
base_dir = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
test_dir = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

print("START")

# prepare output dir
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# save params
params = {
    "image_size" : image_size,
    "batch_size" : batch_size,
    "epochs" : epochs,
    "steps_per_epoch" : steps_per_epoch,
    "validation_steps" : validation_steps,
    "base_dir" : base_dir,
    "test_dir" : test_dir
}

params_path = os.path.join(output_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params_path, f)
    
dataset = pd.read_csv("dataset.csv")
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Import data
print("Load image...")
training_set = train_datagen.flow_from_dataframe(dataframe=dataset,
                                             x_col="file_path", y_col="label",
                                             class_mode="categorical",
                                             target_size=image_size, batch_size=batch_size)

# Create classes index file
# print("Classes index file...")
# categories = list(train_data.class_indices.keys())
# print(train_data.class_indices)

# with open('class_indices.json','w') as f:
#     json.dump(train_data.class_indices, f)

# Train model
print("Train model...")

# base model = MobileNet
base_model = tf.keras.applications.MobileNet(weights = "imagenet",
                                             include_top = False,
                                             input_shape = input_shape)

base_model.trainable = False

# Prepare model
inputs = keras.Input(shape = input_shape)

x = base_model(inputs, training = True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(n_cat, 
                          activation="softmax")(x)

model = keras.Model(inputs = inputs, 
                    outputs = x, 
                    name="LeafDisease_MobileNet")

optimizer = tf.keras.optimizers.Adam()
# earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(optimizer = optimizer,
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
              metrics=[keras.metrics.CategoricalAccuracy(), 
                       'accuracy'])

# history = model.fit(training_set,
#                     validation_data=val_data,
#                     epochs=epochs,
#                     steps_per_epoch=steps_per_epoch,
#                     callbacks=[earlystop],
#                     validation_steps=validation_steps)

history = model.fit(training_set,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch)

# test model
print("Evaluate model...")
test_data = tf.keras.utils.image_dataset_from_directory(test_dir,
    label_mode="categorical",
    shuffle=False,
    image_size=image_size,
    batch_size=batch_size,
)

model.evaluate(test_data)

# Save model
print("Save model...")
model_path = os.path.join(output_dir, 'plant_disease')
model.save(model_path)

# save to tflite
print("Save tflite model...")
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_path = os.path.join(output_dir, 'model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print("END!")
