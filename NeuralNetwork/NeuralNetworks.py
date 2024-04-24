import math

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from utils import getDataRoot
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):

    def __init__(self, nPersons: int, sFeatures: int):
        super().__init__("NeuralNetworkTKEO")
        # Params layer1
        self.l1 = [10, 64, 1]

        # Params layer 2
        self.l2 = [10, 64, 1]

        # Params layer 3
        self.l3 = [10, 64, 1]

        # Params layer 4
        self.l4 = [10, 64, 1]

        # Params layer 5
        self.l5 = [10, 64, 1]

        # These layers are responsible for extracting features and fixing offsets
        self.fLayers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.l1[0], kernel_size=self.l1[1], stride=self.l1[2], padding=round(self.l1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.l1[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.l2[0], out_channels=self.l3[0], kernel_size=self.l3[1], stride=self.l3[2], padding=round(self.l3[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.l3[0], out_channels=self.l4[0], kernel_size=self.l4[1], stride=self.l4[2], padding=round(self.l4[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.l1[0], out_channels=self.l5[0], kernel_size=self.l5[1], stride=self.l5[2], padding=round(self.l5[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )
        # These layers are responsible for classification after being passed through the fLayers

        self.cInput = self.l5[0] * self.calcInputSize(sFeatures)

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

    def calcInputSize(self, nInput):
        inputSize = nInput
        for layers in self.children():
            if isinstance(layers, nn.Sequential):
                for subLayer in layers.children():
                    if isinstance(subLayer, nn.Conv1d):
                        inputSize = self.calcSizeConv(inputSize, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                    if isinstance(subLayer, nn.AvgPool1d):
                        inputSize = self.calcSizePool(inputSize, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
        return inputSize

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
    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO400"))
    testPath = getDataRoot().joinpath("testData")
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEOtest400"))
    labels = dataset.labelStrings
    batchSize = 32
    network = NeuralNetworkTKEO(len(participants), dataset.featureSize)

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
