import tensorflow as tf
import numpy as np
from PIL import Image
import time

from .image import tensor2image

def load_image(name, PATH, from_url=False, low_memory=True):
    if from_url:
        PATH = tf.keras.utils.get_file(name, PATH)

    image = Image.open(PATH)
    if low_memory:
        size = list(image.size)
        MAX_RES = 512
        if max(size) > 512:
            aspect_ratio = max(size) / min(size)
            if size[0] > size[1]:
                size[0] = MAX_RES
                size[1] = MAX_RES // aspect_ratio
            else:
                size[0] = MAX_RES // aspect_ratio
                size[1] = MAX_RES
        image = image.resize((*size))
    array = np.array(image)
    tensor = tf.expand_dims(tf.cast(array, tf.float32), 0)
    tensor = tf.constant(tensor / 255)

    return tensor

def save_image(tensor, name=None):
    image = tensor2image(tensor)

    if not name:
        name = 'generated'
    template = "{}_{}.jpg".format(name, time.time())
    image.save(template)
