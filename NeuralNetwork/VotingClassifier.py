from typing import List, Tuple

import numpy as np
from torch import nn

from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO2


class VotingClassifier:
    def __init__(self, classifiers: List[nn.Module]):
        self.classifiers: List[nn.Module] = classifiers

    def predict(self, inputs: List[List[float]]):
        outputWeights: List[List[float]] = [self.classifiers[i](inputs[i]) for i in range(len(inputs))]
        outputs: List[float] = np.average(outputWeights, dim=1)
        _, prediction = np.max(outputs)
        return prediction


if __name__ == '__main__':
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]

    # Create dataset



    # Get the different classifiers
    cl1 = NeuralNetworkTKEO2(len(participants), sFeatures)
