from torch import Tensor

from NeuralNetwork.InterfaceNN import InterfaceNN


class NeuralNetworkTKEO(InterfaceNN):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        pass
