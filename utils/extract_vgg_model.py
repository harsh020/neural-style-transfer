import tensorflow as tf

def extract_vgg_model(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(nme).output for nme in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model
