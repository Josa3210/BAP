from torch import Tensor, nn
from torch.utils.data import DataLoader

import utils
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):
    def __init__(self, nPersons: int):
        super().__init__("NeuralNetworkTKEO")
        self.layers = nn.Sequential(
            nn.Linear(176319, 1024),
            nn.Dropout(self.dropoutRate),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(self.dropoutRate),
            nn.ReLU(),
            nn.Linear(256, nPersons),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor):
        return self.layers.forward(x)


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
    bounds = {"lr": (1e-5, 1e-3), "dr": (0.2, 0.7)}

    # network.optimizeParams(bounds=bounds, trainingData=dataset)

    network.trainOnData(trainingData=dataset, folds=5, epochs=5, batchSize=batchSize, verbose=True, lr=network.bestLR, dr=network.bestDR)
    network.printResults(fullReport=True)
    network.testOnData(testData=testDataset)
    network.printResults(testResult=True)
    network.saveModel(utils.getDataRoot().joinpath("models"), idNr=2)
