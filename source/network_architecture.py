"""
This class defines the architecture of the neural network model. I consists of the input preprocessing to a format that the neural network
understands. The neural network is made up of three components The convolutional layer, pooling layer and output layers.
"""

#Import Built-Ins

#Import Third-Party
import tensorflow as tf
#Import Homebrew
from source.utility import Utility


class NetworkArchitecture:

    def __init__(self, network_input):
        self.input = network_input

    def preprocess_image(self):
        """
        Consistent preprocessing of all batch images.
        Reshapes matrix of input image of shape(batch size, height, width), to a 4D tensor, compatible
        with the convolutional layer
        :return:
        """
        print("[INFO] loading and preprocessing image....")
        Utility.loadImage()

    def convolution_layer(self, layer_input, filters=32, kernel_size=(5, 5), stride=1, padding="same"):
        """
        A function that takes an input and convoles it against a set of filers to create a feature map
        :param conv_input: a numpy array of size [input_height, input_width, # of colour channels] represents tensor inpt
        :param filters: the number of filters used in convolution
        :param kernel_size: represents the size of each filter
        :param stride:
        :param padding: padding scheme
        :return:
        """
        tf.layers.conv2d(inputs=layer_input,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=stride,
                         padding=padding,
                         activation=tf.nn.relu,
                         trainable=True)


    def relu_layer(self):
        pass

    def maxpool_layer(self,layer_input, pool_size=(2,2), stride=2):
        """
        Reduces the dimensionality of the feature map created by the convolutional layer. In order to reduces the number of parameters
        used in weighting the hidden layer of the neural network.

        Does this by extracting a sub region of the feature map by taking the highest value in that region A.K.A highest detected feature.
        :param pool_size:
        :return:
        """
        tf.layers.max_pooling2d(
            inputs=layer_input,
            pool_size=pool_size,
            strides=stride
        )

    def fully_connected_layer(self):
        pass

