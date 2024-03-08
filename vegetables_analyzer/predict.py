import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import requests
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# SAVE_MODEL_PATH = "./vegetable_model_version_categori_loss_v1.h5"
SAVE_MODEL_PATH = "./vegetable_model_v1.h5"
TMP_IMAGE_PATH = "./tmp.png"

class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd',
               'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot']


# get image data from the internet
def get_image(img_url):
    res = requests.get(img_url, stream=True)
    if res.status_code == 200:
        with open(TMP_IMAGE_PATH, "wb") as f:
            f.write(res.content)
        return True
    else:
        return False


def predict():
    test_image_src = image.load_img(TMP_IMAGE_PATH, target_size=(160, 160))
    test_image_arr = image.img_to_array(test_image_src)
    test_image = tf.expand_dims(test_image_arr, axis=0)
    model = tf.keras.models.load_model(SAVE_MODEL_PATH)
    predictions = model.predict_on_batch(test_image)  # [0.2,0.5,0,3]

    plt.imshow(test_image_arr.astype("uint8"))
    plt.title(class_names[np.argmax(predictions[0])])
    plt.axis("off")
    plt.show()
    plt.close("all")


# test
if __name__ == "__main__":
    # img_url = "https://www.skinnytaste.com/wp-content/uploads/2009/05/String-Beans-with-Garlic-and-Oil-6.jpg"
    img_url = "https://peppergeek.com/wp-content/uploads/2021/12/Capsicum-baccatum.jpg.webp"
    if (get_image(img_url)):
        predict()
