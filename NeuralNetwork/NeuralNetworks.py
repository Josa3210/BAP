import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import utils
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):
    def __init__(self, nPersons: int):
        super().__init__("NeuralNetworkTKEO")
        # These layers are responsible for extracting features and fixing offsets
        # General formula to detect size:
        # Conv1d: (Size + 2*Padding - Filter )/Stride + 1
        # MaxPool: [(Size + 2*Padding) - (dilation x (kernel - 1)) - 1]/stride + 1
        self.fLayers = nn.Sequential(
            # Dimensions: [bSize, 1, 176319]
            nn.Conv1d(1, 10, 5, 1),
            nn.ReLU(),
            # Dimensions: [bSize, 10, 176315]
            nn.MaxPool1d(2, 2),
            # Dimensions: [bSize, 10, 88157]
            nn.Conv1d(10, 20, 5, 1),
            nn.ReLU(),
            # Dimensions: [bSize, 20, 88153]
            nn.MaxPool1d(2, 2)
            # Dimensions: [bSize, 10, 44076]
        )
        # These layers are responsible for classification after being passed through the fLayers
        self.cLayers = nn.Sequential(
            nn.Linear(20*44076, 1024),
            nn.ReLU(),
            nn.Linear(1024, nPersons),
            nn.Softmax(dim=1)
        )

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
    path = utils.getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "ann", "walter", "jan", "lieve"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=utils.getDataRoot().joinpath(r"cache\TKEO"))
    testPath = utils.getDataRoot().joinpath("testData")
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=utils.getDataRoot().joinpath(r"cache\TKEOtest"))
    labels = dataset.labelArray
    batchSize = 4
    network = NeuralNetworkTKEO(len(participants))
    # bounds = {"lr": (1e-5, 1e-3), "dr": (0.2, 0.7)}

    # network.optimizeParams(bounds=bounds, trainingData=dataset)

    network.trainOnData(trainingData=dataset, folds=5, epochs=5, batchSize=batchSize, verbose=True, lr=network.bestLR, dr=network.bestDR)
    network.printResults(fullReport=True)
    network.testOnData(testData=testDataset)
    network.printResults(testResult=True)
    network.saveModel(utils.getDataRoot().joinpath("models"), idNr=2)
