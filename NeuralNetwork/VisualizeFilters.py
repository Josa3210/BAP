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
    path = utils.getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=utils.getDataRoot().joinpath(r"cache\STFT"))
    labels = dataset.labelStrings
    batchSize = 4
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    print(f"Amount of batches: {len(dataloader)}")
    trainingFeatures, trainingLabels = next(iter(dataloader))
    print(f"Feature batch shape: {trainingFeatures.size()}")
    print(f"Labels batch shape: {trainingLabels.size()}")
    model = NeuralNetworkSTFT(5, dataset.featureSize)
    model.loadModel(utils.getDataRoot().joinpath("models/NeuralNetworkSTFT-2941.pth"))
    model.eval()

    showSTFT(model)

    """
    
    fig, axs = plt.subplots(batchSize, filters)

    for i in range(batchSize):
        for j in range(filters):
            data = out_data[i, j]
            data = data.detach().numpy()
            axs[i, j].plot(data)
            axs[i, j].set_yticks([])
    plt.show()
    """
