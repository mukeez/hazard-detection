"""

"""

#Import Built-In
import os
#Import Third-Party
from PIL import Image
#Import Homebrew
from source.network_architecture import NetworkArchitecture


class NeuralNetworkModel:

    def __init__(self):
        self.model = NetworkArchitecture()

    def convolutional_neural_network(self, channel=3):
        print("[INFO]: Preparing Convolutional Neural Network....")

        # Convolutional Layer #1
        conv_layer_1 = self.model.convolution_layer(filters=16, kernel_size=(5,5), stride=1, padding='same')

        # Pooling Layer #2
        maxpool_layer_1 = self.model.maxpool_layer(layer_input=conv_layer_1, pool_size=2)

        #Convolutional Layer #2
        conv_layer_2 = self.model.convolution_layer(layer_input= maxpool_layer_1, filters=32, kernel_size=(5,5), stride=1, padding="same")

        # Pooling Layer #2
        maxpool_layer_2 = self.model.maxpool_layer(layer_input=conv_layer_2, pool_size=2)