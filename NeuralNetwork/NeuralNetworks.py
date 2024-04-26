import torch
from torch import Tensor, nn

from utils import getDataRoot
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):

    def __init__(self, nPersons: int, sFeatures: int, method: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkTKEO", method)
        # Params layer1
        self.l1 = [5, 128, 1]
        # Params layer 2
        self.l2 = [5, 128, 1]
        # Params layer 3
        self.l3 = [5, 64, 1]
        # Params layer 4
        self.l4 = [5, 64, 1]
        # Params layer 5
        self.l5 = [5, 64, 1]
        # Params layer 5
        self.l6 = [5, 64, 1]
        # Params layer 5
        self.l7 = [5, 64, 1]

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
            nn.Conv1d(in_channels=self.l4[0], out_channels=self.l5[0], kernel_size=self.l5[1], stride=self.l5[2], padding=round(self.l5[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.l5[0], out_channels=self.l6[0], kernel_size=self.l6[1], stride=self.l6[2], padding=round(self.l6[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.l6[0], out_channels=self.l7[0], kernel_size=self.l7[1], stride=self.l7[2], padding=round(self.l7[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )

        # These layers are responsible for classification after being passed through the fLayers
        self.cInput = self.l7[0] * self.calcInputSize(sFeatures)

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
        self.apply(self.initWeightsZero)

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


class NeuralNetworkSTFT(InterfaceNN):

    def __init__(self, nPersons: int, sFeatures: list[int, int], initMethod: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkSTFT", initMethod)

        # Params layer1
        self.l1 = [10, 128, 1]
        # Params layer 2
        self.l2 = [10, 128, 1]
        # Params layer 3
        self.l3 = [10, 64, 1]
        # Params layer 4
        self.l4 = [10, 64, 1]
        # Params layer 5
        self.l5 = [10, 64, 1]
        # Params layer 5
        self.l6 = [10, 64, 1]
        # Params layer 5
        self.l7 = [10, 64, 1]

        # These layers are responsible for extracting features and fixing offsets
        self.fLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.l1[0], kernel_size=self.l1[1], stride=self.l1[2], padding=round(self.l1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l1[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l2[0], out_channels=self.l3[0], kernel_size=self.l3[1], stride=self.l3[2], padding=round(self.l3[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l3[0], out_channels=self.l4[0], kernel_size=self.l4[1], stride=self.l4[2], padding=round(self.l4[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l4[0], out_channels=self.l5[0], kernel_size=self.l5[1], stride=self.l5[2], padding=round(self.l5[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l5[0], out_channels=self.l6[0], kernel_size=self.l6[1], stride=self.l6[2], padding=round(self.l6[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l6[0], out_channels=self.l7[0], kernel_size=self.l7[1], stride=self.l7[2], padding=round(self.l7[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

        # These layers are responsible for classification after being passed through the fLayers
        self.cInput = self.l7[0] * self.calcInputSize(sFeatures)

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
        self.apply(self.initWeightsZero)

    def calcInputSize(self, nInput: list[int, int]):
        inputSize = nInput
        for layers in self.children():
            if isinstance(layers, nn.Sequential):
                for subLayer in layers.children():
                    if isinstance(subLayer, nn.Conv1d):
                        inputSize[0] = self.calcSizeConv(inputSize[0], filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                        inputSize[1] = self.calcSizeConv(inputSize[1], filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                    if isinstance(subLayer, nn.AvgPool1d):
                        inputSize[0] = self.calcSizePool(inputSize[0], filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                        inputSize[1] = self.calcSizePool(inputSize[1], filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
        return inputSize[0] * inputSize[1]

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
    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO441"))
    testPath = getDataRoot().joinpath("testData")
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEOtest441"))
    batchSize = 32
    network = NeuralNetworkTKEO(len(participants), dataset.featureSize)

    # bounds = {"lr": (1e-4, 1e-2), "dr": (0.2, 0.8)}
    # results = network.optimizeParams(bounds=bounds, trainingData=trainingDataset)

    # network.trainOnData(trainingData=trainingDataset, folds=5, epochs=50, batchSize=batchSize, verbose=True, lr=results.get("lr"), dr=results.get("dr"))
    network.trainOnData(trainingData=dataset, folds=5, epochs=300, lr=0.0003, dr=0.8, batchSize=batchSize, verbose=True)
    network.printResults(fullReport=True)
    network.testOnData(testData=testDataset)
    network.printResults(testResult=True)
    network.printLoss()

    savePrompt = input("Do you want to save? (Y or N) ")
    if savePrompt.capitalize() == "Y":
        saveName = input("Which name do you want to give it?: ")
        network.saveModel(getDataRoot().joinpath("models"), name=saveName)
