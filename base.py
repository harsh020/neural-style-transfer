import tensorflow as tf
import numpy as np
import regex as re

from model.base import NeuralStyleTransfer
from utils.io import load_image, save_image
from utils.image import tensor2image

class Generator:
    """Class to perfrom Neural Style Transfer. Runs on top of
    `NeuralStyleTransfer` class.
    ::See also : model.base.NeuralStyleTransfer

    Parameters
    ----------
    style_path : string
                 Path of style image. Either on disk or a URL.

    content_path : string
                   Path of content image. Either on disk or a URL.

    style_layers : {array-like}, shape (None, )
                   string, Names of layers of `VGG19` which model should
                   use to extract style image.

    content_layers : {array-like}, shape (None, )
                     string, Names of layers of `VGG19` which model should
                     use to extract content image.

    Attribures
    ----------
    style_path : string
                 Path of style image. Either on disk or a URL.

    content_path : string
                   Path of content image. Either on disk or a URL.

    style_layers : {array-like}, shape (None, )
                   string, Names of layers of `VGG19` which model should
                   use to extract style image.

    content_layers : {array-like}, shape (None, )
                     string, Names of layers of `VGG19` which model should
                     use to extract content image.

    Returns
    -------
    self : object.
    """
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
        """Load images as tensors from given path."""
        url_template = r'http[s]?://([a-zA-Z]|[0-9]|[@#$&\(\)]|[+-=&@!]|[/,.:\"\"])+'
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
        """Fit the `NeuralStyleTransfer` class.

        Parameters
        ----------
        epochs : int
                 Number of epochs to train for.

        lr : float
             Step size.

        lambda_ : int or float
                  Regularization coefficient.

        Returns
        -------
        tensor : {tensor}, shape (1, None, None, 3)
                 float, Tensor of dreamt image.
        """
        style_tensor, content_tensor = self._fit()
        nst = NSTModel(style_tensor, content_tensor, self.style_layers,
                       self.content_layers)
        tensor = nst.generate(epochs, lambda_)

        return tensor

    def transform(self, tensor, save=False, display=True):
        """Transform tensor to image.

        Parameters
        ----------
        tensor : {tensor}, shape (1, None, None, 3)
                 float Tensor of dreamt image.

        save : bool, default=False
               Whether or not to save the image to disk.

        display : bool, default=True
                  Whether or not to display the image.

        Returns
        -------
        image : Image.
        """
        image = tensor2image(tensor)
        if save:
            save_image(tensor)
        if display:
            image.show()
        return image
