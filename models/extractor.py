import tensorflow as tf
import numpy as np

from utils.extract_vgg_model import extract_vgg
from utils.gram_matrix import get_gram_matrix
from utils.image import preprocess_image


class ExtractorModel(tf.keras.Model):
    def __init__(self, style_layers, content_layers):
        super(ExtractorModel, self).__init__()
        self.model = extract_vgg(style_layers+content_layers)
        self.model.trainable = False
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.n_style_layers = len(style_layers)
        self.n_content_layers = len(content_layers)

    def call(self, input):
        unscaled_input = input * 255
        proc_input = preprocess_image(unscaled_input)

        outputs = self.model(proc_input)

        style_outputs, content_outputs = (outputs[:self.n_style_layers],
                                          outputs[self.n_style_layers:])

        style_outputs = [get_gram_matrix(style_output)
                         for style_output in style_outputs]

        style_dict = {style_key: style_value
                      for style_key, style_value in zip(self.style_layers,
                                                        style_outputs)}

        content_dict = {content_key: content_value
                      for content_key, content_value in zip(self.content_layers,
                                                            content_outputs)}

        return {'style': style_dict, 'content': content_dict}
