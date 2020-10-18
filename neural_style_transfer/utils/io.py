import tensorflow as tf
import numpy as np
from PIL import Image
import time

from .image import tensor2image

def load_image(PATH, from_url=False, low_memory=True):
    """Function to load and/or rescale image.

    Parameters
    ----------
    PATH : string
           Path of the image to be loaded.

    from_url : bool
               Whether or not the path provided is a URL.

    low_memory : bool
                 Whether or not to restrict low memory usage.
                 - If `True` limits the maximum scale of largest dimension
                   to be 512px, and scales the other according to the aspect
                   ratio of the original image.
                 - If `False` uses the original resolution of image.
                 - Example : If image size (1200, 800) and low_memory
                             is `True`. Aspect ratio is 1.5 and the
                             new image size will be (512, 341).

    Returns
    -------
    tensor : {tensor}, shape (1, None, None, 3)
             float, Tensor of the loaded image.
    """
    name = PATH.split('/')[-1]
    if from_url:
        PATH = tf.keras.utils.get_file(name, PATH)

    image = Image.open(PATH)
    if low_memory:
        size = list(image.size)
        MAX_RES = 512
        if max(size) > 512:
            aspect_ratio = max(size) / min(size)
            if size[0] > size[1]:
                size[0] = int(MAX_RES)
                size[1] = int(MAX_RES // aspect_ratio)
            else:
                size[0] = int(MAX_RES // aspect_ratio)
                size[1] = int(MAX_RES)
        image = image.resize(tuple(size))

    array = np.array(image)
    tensor = tf.expand_dims(tf.cast(array, tf.float32), 0)
    tensor = tf.constant(tensor / 255)

    return tensor

def save_image(tensor, name=None):
    """Function to save given tensor as image.

    Parameters
    ----------
    tensor : {tensor}, shape (1, None, None, 3)
             float, Tensor of the image to be saved.

    name : string
           Name by which the image should be saved.
           If None, `generated` is the default name along with timestamp.

    Returns
    -------
    None.
    """
    image = tensor2image(tensor)

    if not name:
        name = 'generated'
    template = "{}_{}.jpg".format(name, time.time())
    image.save(template)
