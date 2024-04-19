import math
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import torch
import utils
from featureExtraction.FeatureExtractor import Filter, FeatureExtractorTKEO, FeatureExtractorSTFT
from footstepDataset.FootstepDataset import FootstepDataset

if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=utils.getDataRoot().joinpath(r"cache\TKEO"))
    labels = dataset.labelArray
    batchSize = 4
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    print(f"Amount of batches: {len(dataloader)}")
    trainingFeatures, trainingLabels = next(iter(dataloader))
    print(f"Feature batch shape: {trainingFeatures.size()}")
    print(f"Labels batch shape: {trainingLabels.size()}")

    # fig, ax = plt.subplots(1, batchSize, subplot_kw=dict(projection='3d'))
    fig, ax = plt.subplots(1, batchSize)
    for i in range(batchSize):
        feature = trainingFeatures[i]
        labelIndex = trainingLabels[i]
        labelStr = labels[labelIndex]
        # t = np.arange(0, 50)
        # f = np.arange(0, 169)
        # f, t = np.meshgrid(f, t)
        # ax[i].plot_surface(t, f, feature)
        ax[i].get_xaxis().set_visible(False)

        ax[i].plot(feature)
        ax[i].title.set_text(labelStr)

    plt.show()
