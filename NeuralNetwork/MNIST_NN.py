import torch
import torchvision
from torch import nn


class MnistNN(nn.Module):
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
        # print(x.size())
        x = self.ConvLayers.forward(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
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
