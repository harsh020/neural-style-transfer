import tensorflow as tf
import numpy as np
import regex as re

from model.base import NeuralStyleTransfer
from utils.io import load_image, save_image
from utils.image import tensor2image

class Generator:
    def __init__(self, style_path, content_path, style_layers, content_layers):
        self.style_path = style_path
        self.content_path = content_path

        self.style_layers = style_layers
        self.content_layers = content_layers
        if not style_layers:
            self.style_layers = ['block1_conv1',
                                 'block2_conv1',
                                 'block3_conv1',
                                 'block4_conv1',
                                 'block5_conv1']
        if not content_layers:
            self.content_layers = ['block5_conv2']

        return self

    def _fit(self):
        '''Load images as tensors from given path.'''
        url_template = 'http[s]?://([a-zA-Z]|[0-9]|[@#$&\(\)]|[+-=&@!]|[/,.:\"\"])+'
        from_url = False
        if re.match(url_template, self.style_path):
            from_url = True
        style_tensor = load_image(style_path[0], style_path[1], from_url=from_url)

        from_url = False
        if re.match(url_template, self.content_path):
            from_url = True
        content_tensor = load_image(content_path[0], content_path[1], from_url=from_url)

        return style_tensor, content_tensor

    def fit(self, epochs=1, lr=0.02, lambda_=0):
        style_tensor, content_tensor = self._fit()
        nst = NSTModel(style_tensor, content_tensor, self.style_layers,
                       self.content_layers)
        tensor = nst.generate(epochs, lambda_)

        return tensor

    def transform(self, tensor, save=False, display=True):
        image = tensor2image(tensor)
        if save:
            save_image(tensor)
        if display:
            image.show()
        return image
