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
        # Params layer1
        self.fs1 = 64
        self.ch1 = 5
        self.st1 = 1

        # Params layer 2
        self.fs2 = 64
        self.ch2 = 10
        self.st2 = 1

        # Params layer 3
        self.fs3 = 64
        self.ch3 = 10
        self.st3 = 1

        # Params layer 4
        self.fs4 = 64
        self.ch4 = 10
        self.st4 = 1

        # Params layer 5
        self.fs5 = 64
        self.ch5 = 10
        self.st5 = 1

        # These layers are responsible for extracting features and fixing offsets
        self.fLayers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.ch1, kernel_size=self.fs1, stride=self.st1, padding=round(self.fs1/2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.fs2, stride=self.st2,padding=round(self.fs2/2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.fs3, stride=self.st3,padding=round(self.fs3/2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.fs4, stride=self.st4,padding=round(self.fs4/2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.ch4, out_channels=self.ch5, kernel_size=self.fs5, stride=self.st5,padding=round(self.fs5/2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )
        # These layers are responsible for classification after being passed through the fLayers

        self.cInput = self.ch5*500

        self.cLayers = nn.Sequential(

            nn.Linear(self.cInput, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(128, nPersons),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def calcSizeConv(inputSize, filterSize: int, stride: int = 1, padding: int = 0):
        return math.floor((inputSize + 2 * padding - filterSize) / stride) + 1

    @staticmethod
    def calcSizePool(inputSize, filterSize: int, stride: int, dilation: int = 1, padding: int = 0):
        return math.floor(((inputSize + 2 * padding) - (dilation * (filterSize - 1)) - 1) / stride) + 1

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

    def getTransformedSample(self, sample: torch.Tensor):
        print(sample)
        tSample = sample.unsqueeze(1)
        sample = self.fLayers.forward(tSample)
        print(sample)


if __name__ == '__main__':
    path = getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "ann", "walter", "jan", "lieve"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO"))
    testPath = getDataRoot().joinpath("testData")
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEOtest"))
    labels = dataset.labelStrings
    batchSize = 32
    network = NeuralNetworkTKEO(len(participants))

    # bounds = {"lr": (1e-4, 1), "dr": (0.2, 0.8)}
    # results = network.optimizeParams(bounds=bounds, trainingData=dataset)

    # network.trainOnData(trainingData=dataset, folds=5, epochs=35, batchSize=batchSize, verbose=True, lr=results.get("lr"), dr=results.get("dr"))
    network.trainOnData(trainingData=dataset, folds=5, epochs=50, lr=0.001, dr=0.6, batchSize=batchSize, verbose=True)
    network.printResults(fullReport=True)
    network.testOnData(testData=testDataset)
    network.printResults(testResult=True)
    network.printLoss()

    savePrompt = input("Do you want to save? (Y or N) ")
    if savePrompt.capitalize() == "Y":
        saveName = input("Which name do you want to give it?: ")
        network.saveModel(getDataRoot().joinpath("models"), name=saveName)
