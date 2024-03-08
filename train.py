import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# Load the dataset
PATH = Path('Vegetable Images')
train_dir = PATH / 'train'
validation_dir = PATH / 'validation'

# Create a dataset
train_dataset = image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    shuffle=True,
    image_size=(160, 160),
    batch_size=(32)
)

validation_dataset = image_dataset_from_directory(
    validation_dir,
    label_mode='categorical',
    shuffle=True,
    image_size=(160, 160),
    batch_size=(32)
)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
class_names = train_dataset.class_names


# load data in advance to prevent IO blocking
AUTOTUNE = tf.data.AUTOTUNE
# Data loading optimization
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# Visualize the data
# plt.figure(figsize=(10, 10))
# for images, labels in test_dataset.take(1):  # labels[0,1,0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[np.argmax(labels[i])])
#         plt.axis("off")
# plt.show()
# plt.close("off")

# photo augmentation for the training set
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# normalizing the data to [-1, 1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# Create the base model from the pre-trained model MobileNet V2
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160) + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

prediction_layer = tf.keras.layers.Dense(
    len(class_names), activation='softmax')

# test the basd model
# images, labels = next(iter(train_dataset))
# feature_batch = base_model(images)
# print(feature_batch.shape)

# Visualize the augmented datacle
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):  # labels [0,1,0]
#     first_image = images[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#         plt.imshow(augmented_image[0] / 255)
#         # plt.title(class_names[np.argmax(labels[i])])
#         plt.axis("off")
# plt.show()
# plt.close("off")

# feature extraction
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)  # dropout layer to avoid overfitting
outputs = prediction_layer(x)  # Ensure the layer is called with input

# train the model
base_learning_rate = 0.0001
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              # use the binary cross-entropy loss function bc the model outputs logits
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.summary()

# train the model
initial_epochs = 10
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset
)
loss, accuracy = model.evaluate(test_dataset)
print("Test accurary:", accuracy)
save_model_path = './vegetable_model_v1.h5'
model.save(save_model_path)
