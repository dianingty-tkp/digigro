import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

N_SPLIT = 5
image_size = (180, 180)
input_shape = (180, 180, 3)
batch_size = 32
n_cat=38

epochs = 25
# steps_per_epoch = 1758
# validation_steps = 440
steps_per_epoch = None
validation_steps = None

print("START!")

# files = tf.io.gfile.glob("dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/*/*")

# df = {"file_path": [],
#      "label" : []}
# for f in files:
#     df["file_path"].append(f)
#     df["label"].append(f.split("/")[-2])
    
# df = pd.DataFrame(df)

# kfold = StratifiedKFold(n_splits=N_SPLIT,shuffle=True,random_state=42)
# train_x = df["file_path"]
# train_y = df["label"]

df = pd.read_csv("cv_group.csv")

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

result_df = {"k" : [],
            "loss" : [],
            "cat_accuracy" : [],
            "accuracy" : []}


j=1
while j <= N_SPLIT:
    print(f"Start cross validation {j+1}")
    x_train_df = df[df["group"] != j]
    x_valid_df = df[df["group"] == j]

    print("Load image...")
    training_set = train_datagen.flow_from_dataframe(dataframe=x_train_df,
                                                 x_col="file_path", y_col="label",
                                                 class_mode="categorical",
                                                 target_size=image_size, batch_size=batch_size)

    validation_set = validation_datagen.flow_from_dataframe(dataframe=x_valid_df,
                                                 x_col="file_path", y_col="label",
                                                 class_mode="categorical",
                                                 target_size=image_size, batch_size=batch_size)

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
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.compile(optimizer = optimizer,
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
                  metrics=[keras.metrics.CategoricalAccuracy(), 
                           'accuracy'])

    history = model.fit(training_set,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch)
    
    print("Evaluate...")
    result = model.evaluate(validation_set, steps=validation_steps)
    
    result_df["k"].append(j)
    result_df["loss"].append(result[0])
    result_df["cat_accuracy"].append(result[1])
    result_df["accuracy"].append(result[2])
    j+=1
    
    print("Save result...")
    result_df_cache = pd.DataFrame(result_df)
    result_df_cache.to_csv("cv_result_20230623.csv", index=False)

print("END!")