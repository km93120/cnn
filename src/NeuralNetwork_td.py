import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        self.model = None
    
    def createModel(self):
        """Create and compile the keras model. See layers-18pct.cfg and 
           layers-params-18pct.cfg for the network model, 
           and https://code.google.com/archive/p/cuda-convnet/wikis/LayerParams.wiki 
           for documentation on the layer format.
        """

    def train(self, train_data, train_labels, epochs):
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """

        pass

    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """
        pass

    def test(self, test_data):
        """Make predictions for a list of images and display the results
        
        Arguments:
            test_data {np.array} -- The test images
        """
        pass

    ## Exercise 7 Save and load a model using the keras.models API
    def saveModel(self, saveFile="model.h5"):
        """Save a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        pass

    def loadModel(self, saveFile="model.h5"):
        """Load a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        pass