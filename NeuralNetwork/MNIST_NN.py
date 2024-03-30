import torch
import torchvision
from torch import nn
from torch.utils.data import ConcatDataset
from torchvision import datasets

from NeuralNetwork.InterfaceNN import InterfaceNN


class MnistNN(InterfaceNN):
    def __init__(self):
        super().__init__()
        self.ConvLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3),
            nn.ReLU()
        )

        self.LinLayers = nn.Sequential(
            nn.Linear(10 * 24 * 24, 265),
            nn.ReLU(),
            nn.Linear(265, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.ConvLayers.forward(x)
        x = torch.flatten(x, 1)
        x = self.LinLayers(x)
        return x


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data: torchvision.datasets.MNIST):
        self.data = data.data
        self.labels = data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index].to(torch.float).unsqueeze(0), self.labels[index]
        return img, label


if __name__ == '__main__':
    network: MnistNN = MnistNN()

    mnistTrainSet = datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=None)
    mnistTestSet = datasets.MNIST(root='./MNISTdata', train=False, download=True, transform=None)

    trainDataset = MNISTDataset(mnistTrainSet)
    testDataset = MNISTDataset(mnistTestSet)

    dataset = ConcatDataset([trainDataset, testDataset])

    network.trainOnData(trainingData=trainDataset,
                        verbose=True)

    network.printResults()
