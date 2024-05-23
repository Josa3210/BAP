import math
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import utils
from featureExtraction.FeatureExtractor import Filter, FeatureExtractorTKEO, FeatureExtractorSTFT
from featureExtraction.Transforms import AddOffset
from footstepDataset.FootstepDataset import FootstepDataset

"""
In this function we checked what results we get for applying different transformations to our dataset.
This was used to:
- Check the overall functionality of FootstepDataset
- Check if the right features were extracted
- Check the effect of adding noise
"""


if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorSTFT()
    filterExtr.noiseProfile = path.joinpath(r"..\noiseProfile\noiseProfile2.wav")
    participants = ["ann", "celeste", "jan", "patrick"]#, "simon", "sylvia", "tine", "walter"]
    transformer = AddOffset(amount=10,maxTimeOffset=2)
    dataset = FootstepDataset(path, fExtractor=filterExtr, labelFilter=participants, cachePath=utils.getDataRoot().joinpath(r"cache\TKEO"), transformer=transformer)
    labels = dataset.labelStrings
    batchSize = 4
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    print(f"Amount of batches: {len(dataloader)}")
    trainingFeatures, trainingLabels = next(iter(dataloader))
    print(f"Feature batch shape: {trainingFeatures.size()}")
    print(f"Labels batch shape: {trainingLabels.size()}")

    fig, ax = plt.subplots(1, batchSize, subplot_kw=dict(projection='3d'))
    #fig, ax = plt.subplots(1, batchSize)
    for i in range(batchSize):
        feature = trainingFeatures[i]
        labelIndex = trainingLabels[i]
        labelStr = labels[labelIndex]
        t = np.arange(0, 50)
        f = np.arange(0, 169)
        f, t = np.meshgrid(f, t)
        ax[i].plot_surface(t, f, feature)
        #ax[i].get_xaxis().set_visible(False)

        #ax[i].plot(feature)
        ax[i].title.set_text(labelStr)

    plt.show()
