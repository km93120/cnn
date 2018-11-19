import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        self.model = None
    

    def createModel(self):

        inputLayer      = keras.layers.Flatten(input_shape=(32,32,3))
        conv1           = keras.layers.Conv2D(32,5,activation='relu', input_shape=(32, 32,3)) 
        pool1           = keras.layers.MaxPooling2D(3, strides= 2, data_format="channels_last")
        pool1Activation = keras.layers.Activation('relu')
        rnorm1          = keras.layers.BatchNormalization(axis=1)
        conv2           = keras.layers.Conv2D(32,5,1,"same","channels_last")
        conv2Activation = keras.layers.Activation('relu')
        pool2           = keras.layers.AveragePooling2D(3,2)
        rnorm2          = keras.layers.BatchNormalization(axis=1)
        conv3           = keras.layers.Conv2D(64,5,1,"same","channels_last")
        conv3Activation = keras.layers.Activation('relu')
        pool3           = keras.layers.AveragePooling2D(3,2)
        flattening      = keras.layers.Flatten(input_shape=(32,32,3))   
        fc10            = keras.layers.LocallyConnected2D(32,5,input_shape=(32,3))
        outputLayer     = keras.layers.Dense(10)
        
        

        self.model = keras.Sequential([
                
                
                
            
                conv1,
                pool1,
                pool1Activation,
                rnorm1,
                conv2,
                conv2Activation,
                pool2,
                rnorm2,
                conv3,
                conv3Activation,
                pool3,
                flattening,
                fc10,
                outputLayer   ])

        self.model.compile(optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

      

        """Create and compile the keras model. See layers-18pct.cfg and
           layers-params-18pct.cfg for the network model,
           and https://code.google.com/darchive/p/cuda-convnet/wikis/LayerParams.wiki
           for documentation on the layer format.
        """
        
    
    def train(self, train_data, train_labels, epochs):
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """

        self.model.fit(train_data,train_labels)

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