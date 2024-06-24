import matplotlib.pyplot as plt
import torch
from torch import nn

from NeuralNetwork.NeuralNetworks import NeuralNetworkSTFT
from Tools import PathFinder


def showTKEO(model: nn.Module):
    """
    A custom function to extract the different weights from the convolutional filters and plot them
    """
    for child in model.children():
        if isinstance(child, nn.Sequential):
            for layer in child.children():
                if isinstance(layer, nn.Conv1d):
                    # Get the filters
                    filters = layer.weight

                    # Normalize the values
                    fmin, fmax = torch.min(filters.data), torch.max(filters.data)
                    filters = (filters - fmin) / (fmax - fmin)

                    # Create a subplot for every input and output channel
                    fig, axs = plt.subplots(filters.size()[1], filters.size()[0])
                    outCh, inCh = filters.size()[0], filters.size()[1]

                    # Plot the filters for every subplot
                    for i in range(inCh):
                        for j in range(outCh):
                            f = filters[j, i, :]
                            data = f[:]
                            data.unsqueeze(dim=1)
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


def showSTFT(model: nn.Module):
    """
    A custom function to extract the different weights from the convolutional filters and plot them
    """
    for child in model.children():
        if isinstance(child, nn.Sequential):
            for layer in child.children():
                if isinstance(layer, nn.Conv2d):
                    # Get the filets
                    filters = layer.weight

                    # Normalize the values
                    fmin, fmax = torch.min(filters.data), torch.max(filters.data)
                    filters = (filters - fmin) / (fmax - fmin)

                    # Create subplots for every input and output channel
                    fig, axs = plt.subplots(filters.size()[1], filters.size()[0])
                    outCh, inCh = filters.size()[0], filters.size()[1]
                    for i in range(inCh):
                        for j in range(outCh):
                            # Extract the right filters
                            f = filters[j, i, :]
                            data = f[:]
                            data = data.detach().numpy()

                            # Plot the different filters on the subplots
                            if inCh > 1:
                                axs[i, j].imshow(data, cmap='gray')
                                axs[i, j].set_yticks([])
                                axs[i, j].set_xticks([])
                                if i == 0:
                                    axs[i, j].set_title(f"{j + 1}", fontsize=25)
                                if j == 0:
                                    axs[i, j].set_ylabel(f"{i + 1}    ", rotation="horizontal", fontsize=25)
                            else:
                                axs[j].imshow(data, cmap='gray')
                                axs[j].set_yticks([])
                                axs[j].set_xticks([])
                                axs[j].set_title(f"{j}", fontsize=25)

                    # Show the plot
                    plt.show()


if __name__ == '__main__':
    # Create a certain type of network
    model = NeuralNetworkSTFT(9, [50, 156])
    # Load in the weights of a previously trained model
    model.loadModel(PathFinder.getDataRoot().joinpath("model/NeuralNetworkSTFT-BestFromBatch-10.pth"))
    # Show the filters
    showSTFT(model)
    # showTKEO(model)
