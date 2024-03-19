import os

from featureExtraction.FeatureExtractor import FeatureExtractor
from footstepDataset.FootstepDataset import FootstepDataset

if __name__ == '__main__':
    currentPath = os.getcwd()
    path = currentPath + r"\data"

    dataset = FootstepDataset(path)
    print(dataset.__getitem__(2))
