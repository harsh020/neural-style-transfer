import tensorflow as tf
from PIL import Image

def preprocess_image(tensor):
    tensor = tensor * 255

    tensor = tensor - tf.reduce_mean(tensor)
    tensor = tensor / tf.math.reduce_std(tensor)

    tensor = tensor * 64
    tensor = tensor + 128

    tensor = tf.clip_by_value(tensor, 0, 255)
    return tensor

def tensor2image(tensor):
    img = Image.fromarray(tensor.numpy())
    return img
