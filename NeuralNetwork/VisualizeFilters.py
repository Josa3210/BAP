import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO, NeuralNetworkTKEO2, NeuralNetworkSTFT
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


def showTKEO(model: nn.Module):
    for child in model.children():
        if isinstance(child, nn.Sequential):
            for layer in child.children():
                if isinstance(layer, nn.Conv1d):
                    # print(layer.weight)
                    filters = layer.weight
                    fmin, fmax = torch.min(filters.data), torch.max(filters.data)
                    filters = (filters - fmin) / (fmax - fmin)
                    fig, axs = plt.subplots(filters.size()[1], filters.size()[0])
                    fig.suptitle("Kernels")
                    outCh, inCh = filters.size()[0], filters.size()[1]
                    for i in range(inCh):
                        for j in range(outCh):
                            f = filters[j, i, :]
                            data = f[:]
                            data = data.detach().numpy()

                            if inCh > 1:
                                axs[i, j].plot(data)
                                axs[i, j].set_yticks([])
                                axs[i, j].set_xticks([])
                                axs[i, j].set_title(f"Kernel {i},{j}")
                            else:
                                axs[j].plot(data)
                                axs[j].set_yticks([])
                                axs[j].set_xticks([])
                                axs[j].set_title(f"Kernel {j}")
                    plt.show()


def showSTFT(model: nn.Module):
    for child in model.children():
        if isinstance(child, nn.Sequential):
            for layer in child.children():
                if isinstance(layer, nn.Conv2d):
                    # print(layer.weight)
                    filters = layer.weight
                    fmin, fmax = torch.min(filters.data), torch.max(filters.data)
                    filters = (filters - fmin) / (fmax - fmin)
                    fig, axs = plt.subplots(filters.size()[1], filters.size()[0])
                    fig.suptitle("Kernels")
                    outCh, inCh = filters.size()[0], filters.size()[1]
                    for i in range(inCh):
                        for j in range(outCh):
                            f = filters[j, i, :]
                            data = f[:]
                            data = data.detach().numpy()

                            if inCh > 1:
                                axs[i, j].imshow(data, cmap='gray')
                                axs[i, j].set_yticks([])
                                axs[i, j].set_xticks([])
                                axs[i, j].set_title(f"Kernel {i},{j}")
                            else:
                                axs[j].imshow(data, cmap='gray')
                                axs[j].set_yticks([])
                                axs[j].set_xticks([])
                                axs[j].set_title(f"Kernel {j}")
                    plt.show()


if __name__ == '__main__':
    model = NeuralNetworkSTFT(9, [50, 169])
    model.loadModel(utils.getDataRoot().joinpath("model/NeuralNetworkSTFT-Final.pth"))
    model.eval()

    showSTFT(model)
