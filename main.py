import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from featureExtraction.FeatureExtractor import Filter
from footstepDataset.FootstepDataset import FootstepDataset

if __name__ == '__main__':
    currentPath = os.getcwd()
    path = currentPath + r"\data\testVDB"
    filterExtr = Filter()
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
