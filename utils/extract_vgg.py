import tensorflow as tf

def extract_vgg(layer_names):
    """Function to extract and build inception model. The input of this model
    will be input layer of original model and output will be the list of
    activations from the given layers.

    Parameters
    ----------
    layer_names : {array-like}
                  string, The name of layers from which the which to use to
                  extract semantic image representation.

    Returns
    -------
    model : object, The extracted model.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(nme).output for nme in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model
