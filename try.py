import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from nst_model.base import NSTModel

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

style_image = Image.open(style_path)
style_image.show()
style_array = np.array(style_image)
style_tensor = tf.expand_dims(tf.cast(style_array, tf.float32), 0)
style_tensor = tf.constant(style_tensor / 255)

content_image = Image.open(content_path)
content_image.show()
content_array = np.array(content_image)
content_tensor = tf.expand_dims(tf.cast(content_array, tf.float32), 0)
content_tensor = tf.constant(content_tensor / 255)

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

nst = NSTModel(style_tensor, content_tensor, style_layers, content_layers)
nst.generate(epochs=1000, lambda_=30)
