import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(tensor, standardize=False):
    """
    Function to preprocess given image tensor based on inception
    preprocessing (not including standardize).

    Parameters
    ----------
    tensor :  {tensor}, shape (1, None, None, 3)
              float. Tensor of image.

    standardize : bool, default=False
                  Whether or not to standardize the image tensor.

    Returns
    -------
    tensor : {tensor}, shape (1, None, None, 3)
             float. Tensor of preprocessed image tensor.
    """
    if standardize:
        tensor = tensor * 255

        tensor = tensor - tf.reduce_mean(tensor)
        tensor = tensor / tf.math.reduce_std(tensor)

        tensor = tensor * 64
        tensor = tensor + 128

        tensor = tf.clip_by_value(tensor, 0, 255)

    # channe-wise mean of imagenet dataset
    mean = np.array([103.939, 116.779, 123.68])

    # Reverse channels RGB -> BGR
    tensor = tf.reverse(tensor, axis=[-1])

    tensor = tensor - mean

    tensor = tf.clip_by_value(tensor, 0, 255)

    return tensor

def tensor2image(tensor, rescale=255):
    """
    Function to convert tensor to image.

    Parameters
    ----------
    tensor : {tensor}, shape (1, None, None, 3)
             float. Tensor to be converted to image.

    Returns
    -------
    image : Image.
    """
    if len(tf.shape(tensor)) > 3:
        tensor = tf.squeeze(tensor, axis=0)
    tensor = tensor * rescale
    image = np.array(tensor, dtype='uint8')
    image = Image.fromarray(image)
    return image
