import torch
from torch import Tensor, nn

from NeuralNetwork.InterfaceNN import InterfaceNN, TrainableNN


class NeuralNetworkTKEO(TrainableNN):
    """
    First architecture of NeuralNetworkTKEO. Same architecture as NeuralNetworkSTFT, but withD convolutional layers and different kernel sizes.
    """

    def __init__(self, nPersons: int, sFeatures, method: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkTKEO", method)
        # Params layer1
        self.conf1 = [5, 32, 1]
        self.conf2 = [5, 64, 1]
        self.dropoutRate = 0.2
        self.logger.debug(f"Input features: {sFeatures}")

        # These layers are responsible for extracting features and fixing offsets
        self.convLayers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
        )

        # These layers are responsible for classification after being passed through the convLayers
        self.cInput = self.conf2[0] * self.calcInputSize(sFeatures)
        self.logger.debug(f"cInput: {self.conf2[0]} * {self.calcInputSize(sFeatures)} = {self.cInput}")

        self.denseLayers = nn.Sequential(
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

    @property
    def dropoutRate(self):
        return self._dropoutRate

    @dropoutRate.setter
    def dropoutRate(self, value):
        self._dropoutRate = value

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
        x = self.convLayers.forward(x)
        # Flatten en discard the different channels
        x = torch.flatten(x, start_dim=1)
        # Run through classification layers
        x = self.denseLayers.forward(x)
        return x

    def getTransformedSample(self, sample: torch.Tensor):
        print(sample)
        tSample = sample.unsqueeze(1)
        sample = self.convLayers.forward(tSample)
        print(sample)


class NeuralNetworkTKEO2(TrainableNN):
    """
    A more complex architecture containing blocks of convolutional layers that are passed on to skiplayers.
    Idea is to use the information gathered from the larger layers kernelsizes more directly.
    But to give the other layers an equal share, we reduce the number of inputs to 128 using an MLP.
    """

    def __init__(self, nPersons: int, sFeatures, method: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkTKEO2", method)
        # Params layer1
        self.conf1 = [5, 20, 1]
        self.conf2 = [5, 10, 1]
        self.logger.debug(f"Input features: {sFeatures}")
        # These layers are responsible for extracting features and fixing offsets
        self.convLayers1 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Conv1d(in_channels=1, out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf2[0], kernel_size=self.conf2[1], stride=self.conf2[2], padding=round(self.conf2[1] / 2)),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.convLayers2 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Conv1d(in_channels=self.conf2[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.MaxPool1d(2, 2))

        self.convLayers3 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.convLayers4 = nn.Sequential(
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conf1[0], out_channels=self.conf1[0], kernel_size=self.conf1[1], stride=self.conf1[2], padding=round(self.conf1[1] / 2)),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        inputS1 = self.calcInputSize(sFeatures, 1)
        self.skipLayer1 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(inputS1, 512),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(512, 128),
            nn.Softmax(dim=1)
        )

        inputS2 = self.calcInputSize(sFeatures, 2)
        self.skipLayer2 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(inputS2, 256),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(256, 128),
            nn.Softmax(dim=1)
        )

        inputS3 = self.calcInputSize(sFeatures, 3)
        self.skipLayer3 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(inputS3, 128),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(128, 128),
            nn.Softmax(dim=1)
        )

        inputS4 = self.calcInputSize(sFeatures, 4)
        self.skipLayer4 = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(inputS4, 128),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(128, 128),
            nn.Softmax(dim=1)
        )

        # These layers are responsible for classification after being passed through the convLayers
        self.cInput = self.conf2[0] * 4 * 128
        self.logger.debug(f"cInput: {self.cInput}")

        self.denseLayers = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(self.cInput, 512),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(256, nPersons),
            nn.Softmax(dim=1)
        )
        self.apply(self.initWeightsZero)

    def calcInputSize(self, nInput, end: int):
        iteration = 0
        inputSize = nInput[0]

        for layers in self.children():
            if iteration < end:
                if isinstance(layers, nn.Sequential):
                    for subLayer in layers.children():
                        if isinstance(subLayer, nn.Conv1d):
                            inputSize = self.calcSizeConv(inputSize, filterSize=subLayer.kernel_size[0], stride=subLayer.stride[0], padding=int(subLayer.padding[0]))
                        if isinstance(subLayer, nn.MaxPool1d):
                            inputSize = self.calcSizePool(inputSize, filterSize=subLayer.kernel_size, stride=subLayer.stride, padding=int(subLayer.padding))
                    iteration += 1
            else:
                return inputSize

    def forward(self, x: Tensor):
        # Add a dimension in front of feature --> "1 channel"
        x = x.unsqueeze(1)
        # Run through the different layers
        x1 = self.convLayers1.forward(x)
        x1s = self.skipLayer1(x1)
        xfl1 = torch.flatten(x1s, start_dim=1)

        x2 = self.convLayers2.forward(x1)
        x2s = self.skipLayer2(x2)
        xfl2 = torch.flatten(x2s, start_dim=1)

        x3 = self.convLayers3.forward(x2)
        x3s = self.skipLayer3(x3)
        xfl3 = torch.flatten(x3s, start_dim=1)

        x4 = self.convLayers4.forward(x3)
        x4s = self.skipLayer4(x4)
        xfl4 = torch.flatten(x4s, start_dim=1)

        # Flatten en discard the different channels
        xtot = torch.cat((xfl1, xfl2, xfl3, xfl4), dim=1)
        # Run through classification layers
        out = self.denseLayers.forward(xtot)
        return out


class NeuralNetworkSTFT(TrainableNN):
    """
    Very simple architecture with 4 blocks of 2x Convolutional2D followed by a ReLu. At the end a normal MLP classifier.
    """

    def __init__(self, nPersons: int, sFeatures: list[int, int], initMethod: nn.init = nn.init.xavier_normal_):
        super().__init__("NeuralNetworkSTFT", initMethod)

        # Params layer1
        self.l1 = [5, 10, 1]
        # Params layer 2
        self.l2 = [5, 5, 1]

        # These layers are responsible for extracting features and fixing offsets
        self.convLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.l1[0], kernel_size=self.l1[1], stride=self.l1[2], padding=round(self.l1[1] / 2)),
            nn.Conv2d(in_channels=self.l1[0], out_channels=self.l1[0], kernel_size=self.l1[1], stride=self.l1[2], padding=round(self.l1[1] / 2)),
            nn.BatchNorm2d(self.l1[0]),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l1[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.Conv2d(in_channels=self.l2[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.BatchNorm2d(self.l2[0]),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.l2[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.Conv2d(in_channels=self.l2[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding=round(self.l2[1] / 2)),
            nn.BatchNorm2d(self.l2[0]),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

        # These layers are responsible for classification after being passed through the convLayers
        self.cInput = self.l2[0] * self.calcInputSize(sFeatures)

        self.logger.debug(f"Input features: {sFeatures}")
        self.logger.debug(f"Input classification: {self.cInput}")

        self.denseLayers = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(self.cInput, 256),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropoutRate),
            nn.Linear(128, nPersons),
            nn.Softmax(dim=1)
        )

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
        x = self.convLayers.forward(x)
        # Flatten en discard the different channels
        x = torch.flatten(x, start_dim=1)
        # Run through classification layers
        x = self.denseLayers.forward(x)
        return x


class FootstepNN(InterfaceNN):

    def __init__(self, nPersons: int):
        super().__init__("FootstepNN")
        # Params layer1
        self.lType1 = [5, 10, 1]
        self.lType2 = [5, 5, 1]

        # Labels
        self.stringLabels = ["ann", "celeste", "jan", "patrick", "simon", "sylvia", "tine", "walter"]

        # These layers are responsible for extracting features and fixing offsets
        self.convLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.lType1[0], kernel_size=self.lType1[1], stride=self.lType1[2], padding=round(self.lType1[1] / 2)),
            nn.Conv2d(in_channels=self.lType1[0], out_channels=self.lType1[0], kernel_size=self.lType1[1], stride=self.lType1[2], padding=round(self.lType1[1] / 2)),
            nn.BatchNorm2d(self.lType1[0]),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.lType1[0], out_channels=self.lType2[0], kernel_size=self.lType2[1], stride=self.lType2[2], padding=round(self.lType2[1] / 2)),
            nn.Conv2d(in_channels=self.lType2[0], out_channels=self.lType2[0], kernel_size=self.lType2[1], stride=self.lType2[2], padding=round(self.lType2[1] / 2)),
            nn.BatchNorm2d(self.lType2[0]),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=self.lType2[0], out_channels=self.lType2[0], kernel_size=self.lType2[1], stride=self.lType2[2], padding=round(self.lType2[1] / 2)),
            nn.Conv2d(in_channels=self.lType2[0], out_channels=self.lType2[0], kernel_size=self.lType2[1], stride=self.lType2[2], padding=round(self.lType2[1] / 2)),
            nn.BatchNorm2d(self.lType2[0]),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

        self.denseLayers = nn.Sequential(
            nn.Linear(630, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, nPersons),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor):
        # Add a dimension in front of feature --> "1 channel"
        x = x.unsqueeze(1)
        # Run through feature layers
        x = self.convLayers.forward(x)
        # Flatten en discard the different channels
        x = torch.flatten(x, start_dim=1)
        # Run through classification layers
        x = self.denseLayers.forward(x)
        return x
