import tensorflow as tf

def get_gram_matrix(input_tensor):
    """Function to find gram matrix to given tensor.

    Parameters
    ----------
    input_tensor : {tensor}, shape (1, None, None, 3)
                   float, Tensor of input image.

    Returns
    -------
    gram : {tensor}, shape (1, None, None, 3)
           float, Tensor of gram matrix of input tensor.
    """
    gram = tf.linalg.einsum('gijc, gijd->gcd', input_tensor, input_tensor)
    shape = tf.shape(input_tensor)
    IJ = tf.cast(shape[0]*shape[1], tf.float32)
    gram = gram / IJ
    return gram
