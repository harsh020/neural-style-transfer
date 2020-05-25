import tensorflow as tf
from PIL import Image
import numpy as np

from .extractor import ExtractorModel
from .utils.image import tensor2image


class BaseModel:
    '''Base Model to perform gradient descent on an image.'''
    def __init__(self, initializer):
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
        with tf.GradientTape() as tape:
            outputs = self.model(self.image)
            loss = self.loss(outputs)
            reg = self.lambda_ * self.regularizer(self.image)
            loss += reg
        grad = tape.gradient(loss, self.image)
        self.optimizer.apply_gradients([(grad, self.image)])
        self.image.assign(tf.clip_by_value(self.image, 0, 1))

    def fit(self, epochs=1):
        for i in range(epochs):
            self._train_step()
            print("Epoch: {:4d}/{:4d}".format(i+1, epochs), end='\r')
        return self.image


class NeuralStyleTransfer(BaseModel):
    def __init__(self, style_tensor, content_tensor, style_layers,
                 content_layers, style_weight=1e-2, content_weight=1e2):
        super(NSTModel, self).__init__(content_tensor)
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
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2)
                                              for name in style_outputs.keys()])
        style_loss = style_loss * self.style_weight / self.n_style

        return style_loss

    def _content_loss(self, content_outputs):
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2)
                                              for name in content_outputs.keys()])
        content_loss = content_loss * self.content_weight / self.n_content

        return content_loss

    def style_content_loss(self, outputs):
        style_loss = self._style_loss(outputs['style'])
        content_loss = self._content_loss(outputs['content'])

        loss = style_loss + content_loss

        return loss

    def total_variation_regularizer(self, tensor, beta=1.2):
        reg = (tf.reduce_sum((tensor[:,:,1:,:] - tensor[:,:,:-1,:])**2) +
               tf.reduce_sum((tensor[:,1:,:,:] - tensor[:,:-1,:,:])**2))**(beta/2.)
        return reg

    def generate(self, lr=0.02, epochs=1, lambda_=0):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.compile(self.extractor, optimizer, self.style_content_loss,
                     self.total_variation_regularizer, lambda_)
        tensor = self.fit(epochs)

        return tensor
