import tensorflow as tf
from PIL import Image
import numpy as np

from .extractor import ExtractorModel
from ..utils.image import tensor2image


class BaseModel:
    """Base Model to perform gradient descent."""
    def __init__(self, initializer):
        # super(BaseModel, self).__init__()
        self.image = tf.Variable(initializer)

    def compile(self, model=None, optimizer=None, loss=None,
                regularizer=None, lambda_=0):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.lambda_ = lambda_

    @tf.function
    def _train_step(self):
        """Perform one gradient descent step."""
        with tf.GradientTape() as tape:
            outputs = self.model(self.image)
            loss = self.loss(outputs)
            reg = self.lambda_ * self.regularizer(self.image)
            loss += reg
        grad = tape.gradient(loss, self.image)
        self.optimizer.apply_gradients([(grad, self.image)])
        self.image.assign(tf.clip_by_value(self.image, 0, 1))

    def fit(self, epochs=1):
        """Fit `NeuralStyleTransfer` model.

        Parameters
        ----------
        epochs : int. Number of epochs to train for.

        Returns
        -------
        image : {tensor}, shape (1, None, None, 3)
                float, Transformed tensor of input image, after
                entire training.
        """
        for i in range(epochs):
            self._train_step()
            print("Epoch: {:4d}/{:4d}".format(i+1, epochs), end='\r')
        return self.image


class NeuralStyleTransfer(BaseModel):
    """Model to perform Neural Style Transfer.

    Inherits
    --------
    BaseModel : BaseModel to perform gradient descent.
    tf.keras.Model : Tensorflow.Keras.Model class.

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

    style_weight : float, default=1e-2
                   Coefficient of loss from style image.

    content_weight : float, default=1e4
                     Coefficient of loss from content image.

    Attribures
    ----------
    extractor : object
                `VGG19` custom model to extract symantic image information.

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

    n_style : float
              Number of layers in style image.

    n_content : float
                Number of layers in content image.

    style_weight : float
                   Coefficient of loss from style image.

    content_weight : float
                     Coefficient of loss from content image.

    Returns
    -------
    self : object.
    """
    def __init__(self, style_tensor, content_tensor, style_layers,
                 content_layers, style_weight=1e-2, content_weight=1e4):
        super(NeuralStyleTransfer, self).__init__(initializer=content_tensor)
        self.extractor = ExtractorModel(style_layers, content_layers)
        outputs = self.extractor(style_tensor)
        self.style_targets = outputs['style']
        outputs = self.extractor(content_tensor)
        self.content_targets = outputs['content']

        self.n_style = self.extractor.n_style_layers
        self.n_content = self.extractor.n_style_layers

        self.style_weight = style_weight
        self.content_weight = content_weight

    def _style_loss(self, style_outputs):
        """Function to calculate loss from style activations."""
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2)
                                              for name in style_outputs.keys()])
        style_loss = style_loss * self.style_weight / self.n_style

        return style_loss

    def _content_loss(self, content_outputs):
        """Function to calculate loss from content activations."""
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2)
                                              for name in content_outputs.keys()])
        content_loss = content_loss * self.content_weight / self.n_content

        return content_loss

    def style_content_loss(self, outputs):
        """Function to calculate total loss.

        Parameters
        ----------
        outputs : {array-like}, shape (None, )
                  Outputs of the custom `VGG19` extraction model.

        Returns
        -------
        loss : float
               Total loss for a given output.
        """
        style_loss = self._style_loss(outputs['style'])
        content_loss = self._content_loss(outputs['content'])

        loss = style_loss + content_loss

        return loss

    def total_variation_regularizer(self, tensor, beta=1.2):
        """Calculate Total Variation Regularization value."""
        reg = (tf.reduce_sum((tensor[:,:,1:,:] - tensor[:,:,:-1,:])**2) +
               tf.reduce_sum((tensor[:,1:,:,:] - tensor[:,:-1,:,:])**2))**(beta/2.)
        return reg

    def generate(self, lr=0.02, epochs=1, lambda_=0):
        """Generate style transfer image.

        lr : float
             Step size.

        epochs : int
                 Number of epochs to train for.

        lambda_ : float
                  Coefficent for total variational regularizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.compile(self.extractor, optimizer, self.style_content_loss,
                     self.total_variation_regularizer, lambda_)
        tensor = self.fit(epochs)

        return tensor
