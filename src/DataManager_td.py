import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.datasets import cifar10


class DataManager(object):

    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.eval_data = []
        self.eval_labels = []
        self.loadData()

    def loadData(self):
        (train_data, train_labels), (eval_data, eval_labels) = cifar10.load_data()
        self.train_data = train_data/255.0
        self.train_labels = train_labels
        self.eval_data = eval_data/255.0
        self.eval_labels = eval_labels    

        """Load the data from cifar-10-batches. 
           See http://www.cs.toronto.edu/~kriz/cifar.html for instructions on 
           how to do so.
        """
        """a = 5

        for k in range(1,a+1):
            dictionary = self.unpickle("./cifar-10-batches-py/data_batch_" + str(a))
            dataArr = dictionary.get(b'data')
            labelArr = dictionary.get(b'labels')
            for i in range(0,len(dataArr)):
                self.train_data.append(dataArr[i]/255.0)
                self.train_labels.append(labelArr[i])

        testDictionary = self.unpickle("./cifar-10-batches-py/test_batch")
        evalDataArr   = testDictionary.get(b'data')
        evalLabelsArr = testDictionary.get(b'labels')
        for i in range(0, len(dataArr)):
            self.eval_data.append(evalDataArr[i]/255.0)
            self.eval_labels.append(evalLabelsArr[i])


        print(len(self.train_data))
        #self.train_data = self.train_data / 255.0
        # for k in range(0,len(dictionary)):
        #print(dictionary.keys())
        #print(self.train_data)

        a = 5
        for i in range(0,a):
            self.train_data.append(self.unpickle("./cifar-10-batches-py/data_batch_1"))
        """
        print(self.train_data.__len__())
        print(self.eval_data.__len__()) 
        print(self.eval_data[5])

    # keras.layers.BatchNormalization    seule normalisation disponible.
    # fc : fully connected
    # parametres du reseau dans params.18cfg
    # couches convolutionnelles dans keras
    # quand le modele est construit : augmentation de données : elle consite en la modification(rotation) + duplication. on peut utiliser keras pour cette operation

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict

        pass
