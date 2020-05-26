import tensorflow as tf
from PIL import Image

def preprocess_image(tensor):
    """
    Function to preprocess given image tensor based on inception
    preprocessing (not including standardize).

    Parameters
    ----------
    tensor :  {tensor}, shape (1, None, None, 3)
              float. Tensor of image.

    standardize : bool. Whether or not to standardize the image tensor.

    Returns
    -------
    tensor : {tensor}, shape (1, None, None, 3)
             float. Tensor of preprocessed image tensor.
    """
    tensor = tensor * 255

    tensor = tensor - tf.reduce_mean(tensor)
    tensor = tensor / tf.math.reduce_std(tensor)

    tensor = tensor * 64
    tensor = tensor + 128

    tensor = tf.clip_by_value(tensor, 0, 255)
    return tensor

def tensor2image(tensor):
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
    image = np.array(tensor, dtype='uint8') * 255
    image = Image.fromarray(image)
    return image
