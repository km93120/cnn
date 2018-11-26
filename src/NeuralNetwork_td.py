import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        self.model = None
    

    def createModel(self):

       conv1 = keras.layers.Conv2D(filters= 32,kernel_size = 5,strides = 1, padding = "same",
                                   data_format = 'channels_last',activation='relu', 
                                   input_shape= (32, 32,3))

       pool1 = keras.layers.MaxPooling2D(pool_size=3,strides = 2,data_format = "channels_last")

       pool1Activation = keras.layers.Activation('relu')
       
       rnorm1 = keras.layers.BatchNormalization(axis = 3)
 
       conv2  = keras.layers.Conv2D(filters= 32,kernel_size= 5,strides = 1,padding = "same",
                                    data_format = "channels_last",activation='relu');
       
       pool2  = keras.layers.AveragePooling2D(pool_size = (3,3),strides = 2,data_format = "channels_last")

       rnorm2 = keras.layers.BatchNormalization(axis=3)

       conv3  = keras.layers.Conv2D(filters= 64,kernel_size= 5,strides = 1,padding = "same",
                                    data_format = "channels_last",activation='relu');
       
       pool3  = keras.layers.AveragePooling2D(pool_size = (3,3),strides = 2,data_format = "channels_last")

       flattening = keras.layers.Flatten(data_format = "channels_last");

       fc10  = keras.layers.Dense(10);

       softmax = keras.layers.Softmax()
       self.model = keras.Sequential([
                conv1,
                pool1,
                pool1Activation,
                rnorm1,
                conv2,
                pool2,
                rnorm2,
                conv3,
                pool3,
                flattening,
                fc10,
                softmax])

        

        
    



        

      

    
        
    
    def train(self, datagen ,train_data, train_labels, epochss):
        
        self.model.compile(optimizer=keras.optimizers.Adam(lr = 1.0),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """
        #self.model.fit_generator(datagen.flow(train_labels, train_labels, batch_size=32),
         #                   steps_per_epoch=len(train_data) / 32, epochs=epochss)
        self.model.fit(train_data,train_labels,epochs = epochss,batch_size = 32)

        pass

    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """
        return self.model.evaluate(eval_data, eval_labels, batch_size=32)
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