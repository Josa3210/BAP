import torch
from torch import Tensor, nn

from utils import getDataRoot
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO, FeatureExtractorSTFT
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):

    def __init__(self, nPersons: int, sFeatures: int, method: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkTKEO", method)
        # Params layer1
        self.conf1 = [5, 128, 1]
        self.conf2 = [5, 64, 1]
        self.logger.debug(f"Input features: {sFeatures}")
        # These layers are responsible for extracting features and fixing offsets
        self.fLayers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
        )

        # These layers are responsible for classification after being passed through the fLayers
        self.cInput = self.conf2[0] * self.calcInputSize(sFeatures)
        self.logger.debug(f"cInput: {self.conf2[0]} * {self.calcInputSize(sFeatures)} = {self.cInput}")

        self.cLayers = nn.Sequential(
            nn.Linear(self.cInput, 512),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(128, nPersons),
            nn.Softmax(dim=1)
        )
        self.apply(self.initWeightsZero)

    def calcInputSize(self, nInput):
        inputSize = nInput[0]
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


class NeuralNetworkTKEO2(InterfaceNN):

    def __init__(self, nPersons: int, sFeatures, method: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkTKEO2", method)
        # Params layer1
        self.conf1 = [5, 128, 1]
        self.conf2 = [5, 64, 1]
        self.logger.debug(f"Input features: {sFeatures}")
        # These layers are responsible for extracting features and fixing offsets
        self.fLayers1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )
        self.fLayers2 = nn.Sequential(
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2))

        self.fLayers3 = nn.Sequential(
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )
        self.fLayers4 = nn.Sequential(
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )

        # These layers are responsible for classification after being passed through the fLayers
        self.cInput = self.conf2[0] * self.calcInputSize(sFeatures)
        self.logger.debug(f"cInput: {self.conf2[0]} * {self.calcInputSize(sFeatures)} = {self.cInput}")

        self.cLayers = nn.Sequential(
            nn.Linear(self.cInput, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(256, nPersons),
            nn.Softmax(dim=1)
        )
        self.apply(self.initWeightsZero)

    def calcInputSize(self, nInput):
        endSize = 0
        inputSize = nInput[0]
        for layers in self.children():
            if isinstance(layers, nn.Sequential):
                for subLayer in layers.children():
                    if isinstance(subLayer, nn.Conv1d):
                        inputSize = self.calcSizeConv(inputSize, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                    if isinstance(subLayer, nn.AvgPool1d):
                        inputSize = self.calcSizePool(inputSize, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                endSize += inputSize
        return endSize

    def forward(self, x: Tensor):
        # Add a dimension in front of feature --> "1 channel"
        x = x.unsqueeze(1)
        # Run through feature layers
        x1 = self.fLayers1.forward(x)
        x2 = self.fLayers2.forward(x1)
        x3 = self.fLayers3.forward(x2)
        x4 = self.fLayers4.forward(x3)
        xfl1 = torch.flatten(x1, start_dim=1)
        xfl2 = torch.flatten(x2, start_dim=1)
        xfl3 = torch.flatten(x3, start_dim=1)
        xfl4 = torch.flatten(x4, start_dim=1)
        # Flatten en discard the different channels
        xtot = torch.cat((xfl1, xfl2, xfl3, xfl4), dim=1)
        # Run through classification layers
        out = self.cLayers.forward(xtot)
        return out

    def getTransformedSample(self, sample: torch.Tensor):
        print(sample)
        tSample = sample.unsqueeze(1)
        sample = self.fLayers.forward(tSample)
        print(sample)


class NeuralNetworkSTFT(InterfaceNN):

    def __init__(self, nPersons: int, sFeatures: list[int, int], initMethod: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkSTFT", initMethod)

        # Params layer1
        self.l1 = [5, 10, 1]
        # Params layer 2
        self.l2 = [5, 10, 1]
        # Params layer 3
        self.l3 = [5, 5, 1]
        # Params layer 4
        self.l4 = [5, 5, 1]
        # Params layer 5
        self.l5 = [5, 5, 1]
        # Params layer 5
        self.l6 = [5, 5, 1]
        # Params layer 5
        self.l7 = [5, 5, 1]

        # These layers are responsible for extracting features and fixing offsets
        self.fLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.l1[0], kernel_size=self.l1[1], stride=self.l1[2], padding=round(self.l1[1] / 2)),
            nn.Conv2d(in_channels=self.l1[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l2[0], out_channels=self.l3[0], kernel_size=self.l3[1], stride=self.l3[2], padding=round(self.l3[1] / 2)),
            nn.Conv2d(in_channels=self.l3[0], out_channels=self.l4[0], kernel_size=self.l4[1], stride=self.l4[2], padding=round(self.l4[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l2[0], out_channels=self.l3[0], kernel_size=self.l3[1], stride=self.l3[2], padding=round(self.l3[1] / 2)),
            nn.Conv2d(in_channels=self.l3[0], out_channels=self.l4[0], kernel_size=self.l4[1], stride=self.l4[2], padding=round(self.l4[1] / 2)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),


        )

        # These layers are responsible for classification after being passed through the fLayers
        self.cInput = self.l7[0] * self.calcInputSize(sFeatures)

        self.logger.debug(f"Input features: {sFeatures}")
        self.logger.debug(f"Input classification: {self.cInput}")

        self.cLayers = nn.Sequential(
            nn.Linear(self.cInput, 256),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(256, 128),
            nn.Dropout(self.dropoutRate),
            nn.Linear(128, nPersons),
            nn.Softmax(dim=1)
        )
        self.apply(self.initWeightsZero)

    def calcInputSize(self, nInput: list[int, int]):
        inputSize0, inputSize1 = nInput
        for layers in self.children():
            if isinstance(layers, nn.Sequential):
                for subLayer in layers.children():
                    if isinstance(subLayer, nn.Conv2d):
                        inputSize0 = self.calcSizeConv(inputSize0, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                        inputSize1 = self.calcSizeConv(inputSize1, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                    if isinstance(subLayer, nn.AvgPool2d):
                        inputSize0 = self.calcSizePool(inputSize0, filterSize=subLayer.kernel_size, stride=subLayer.stride, padding=int(subLayer.padding))
                        inputSize1 = self.calcSizePool(inputSize1, filterSize=subLayer.kernel_size, stride=subLayer.stride, padding=int(subLayer.padding))
        return inputSize0 * inputSize1

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
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]
    # participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\STFT"))
    testPath = getDataRoot().joinpath("testData")
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\STFTtest"))
    batchSize = 32
    sFeatures = dataset.featureSize
    network = NeuralNetworkSTFT(len(participants), sFeatures, nn.init.kaiming_uniform_)
    network.learningRate = 0.00045
    network.dropoutRate = 0.20
    network.batchSize = batchSize
    network.folds = 1
    network.epochs = 350

    # bounds = {"epochs": (200, 500)}
    # results = network.optimizeParams(bounds=bounds, trainingData=dataset, batchSize=batchSize)

    # network.trainOnData(trainingData=dataset, folds=1, epochs=results.get("epochs"), batchSize=batchSize, verbose=True, lr=0.00045, dr=0.75)
    network.trainOnData(trainingData=dataset, verbose=True)
    network.printResults(fullReport=True)
    network.testOnData(testData=testDataset)
    network.printResults(testResult=True)
    network.printLoss()

    savePrompt = input("Do you want to save? (Y or N) ")
    if savePrompt.capitalize() == "Y":
        saveName = input("Which name do you want to give it?: ")
        network.saveModel(getDataRoot().joinpath("models"), name=saveName)
