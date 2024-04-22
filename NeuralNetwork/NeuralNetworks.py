import math

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from utils import getDataRoot
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):
    def __init__(self, nPersons: int):
        super().__init__("NeuralNetworkTKEO")
        self.fs1 = 64
        self.fs2 = 64
        # These layers are responsible for extracting features and fixing offsets
        # General formula to detect size:
        # Conv1d: (Size + 2*Padding - Filter )/Stride + 1
        # MaxPool: [(Size + 2*Padding) - (dilation x (kernel - 1)) - 1]/stride + 1
        self.fLayers = nn.Sequential(
            # Dimensions: [bSize, 1, 176319]
            nn.Conv1d(1, 5, self.fs1, 1),
            nn.ReLU(),
            # Dimensions: [bSize, 5, 176256]
            nn.AvgPool1d(4, 4),
            # Dimensions: [bSize, 5, 44064]
            nn.Conv1d(5, 10, self.fs2, 1),
            nn.ReLU(),
            # Dimensions: [bSize, 10, 88097]
            nn.AvgPool1d(4, 4)
            # Dimensions: [bSize, 10, 44048]
        )
        # These layers are responsible for classification after being passed through the fLayers

        self.cInput = self.calcSizePool(self.calcSizeConv(self.calcSizePool(self.calcSizeConv(176319, self.fs1), 4, 4), self.fs2), 4, 4)
        self.cLayers = nn.Sequential(
            nn.Linear(10 * self.cInput, 1024),
            nn.ReLU(),
            nn.Linear(1024, nPersons),
            nn.Softmax(dim=1)

        )

    @staticmethod
    def calcSizeConv(inputSize, filterSize: int, stride: int = 1, padding: int = 0):
        return math.floor((inputSize + 2 * padding - filterSize) / stride) + 1

    @staticmethod
    def calcSizePool(inputSize, filterSize: int, stride: int, dilation: int = 1, padding: int = 0):
        return math.floor(((inputSize + 2 * padding) - (dilation * (filterSize - 1)) - 1) / stride) + 1

    @property
    def fs1(self):
        return self._fs1

    @fs1.setter
    def fs1(self, value):
        self._fs1 = value

    @property
    def fs2(self):
        return self._fs2

    @fs2.setter
    def fs2(self, value):
        self._fs2 = value

    def trainOnData(self, trainingData: Dataset = None, folds: int = None, epochs: int = None, batchSize: int = None, lr: float = None, fs1: int = None, fs2: int = None, verbose: bool = False):
        if fs1 is not None:
            self.fs1 = fs1
        if fs2 is not None:
            self.fs2 = fs2
        return super().trainOnData(trainingData, folds, epochs, batchSize, lr, verbose)

    def forward(self, x: Tensor):
        # Add a dimension in front of feature --> "1 channel"
        x = x.unsqueeze(1)
        # Run through feature layers
        x = self.fLayers.forward(x)
        # Flatten en discard the different channels
        x = torch.flatten(x, start_dim=1)
        # Run through classification layers
        x = self.cLayers.forward(x)
        return x


if __name__ == '__main__':
    path = getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "ann", "walter", "jan", "lieve"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO"))
    testPath = getDataRoot().joinpath("testData")
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEOtest"))
    labels = dataset.labelArray
    batchSize = 4
    network = NeuralNetworkTKEO(len(participants))
    bounds = {"lr": (1e-5, 1e-3), "fs1": (32, 256), "fs2": (32, 256)}

    results = network.optimizeParams(bounds=bounds, trainingData=dataset)

    network.trainOnData(trainingData=dataset, folds=5, epochs=5, batchSize=batchSize, verbose=True, lr=results.get("lr"), fs1=math.floor(results.get("fs1")), fs2=math.floor(results.get("fs2")))
    network.printResults(fullReport=True)
    network.testOnData(testData=testDataset)
    network.printResults(testResult=True)
    network.saveModel(getDataRoot().joinpath("models"), idNr=2)
