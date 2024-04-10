import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils
from featureExtraction.FeatureExtractor import Filter, FeatureExtractorTKEO, FeatureExtractorSTFT
from footstepDataset.FootstepDataset import FootstepDataset

if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("testVDB")
    filterExtr = FeatureExtractorSTFT()
    filterExtr.noiseProfile = r"..\data\testVDB\noiseProfile\noiseProfile1.wav"
    dataset = FootstepDataset(path, 3, transForm=filterExtr)
    print(dataset.__getitem__(2))

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    feature = train_features[0]
    label = train_labels[0]
    plt.plot(feature)
    plt.show()
    print(f"Label: {label}")
