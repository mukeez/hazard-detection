"""
Utility Script contiaining some useful commands
"""

#Import Built-In
import os
#Import Third-Party
from PIL import Image
#Import Homebrew

class Utility:

    @staticmethod
    def load_image(directory):
        return [os.path.join(directory, file) for file in os.listdir(directory)]