from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np


def main():
    dm = DataManager()
    nn = NeuralNetwork()
    nn.createModel()
    

if __name__ == "__main__":
    main()