from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader

import utils
from featureExtraction.FeatureExtractor import FeatureExtractor, Filter, FeatureExtractorTKEO


class FootstepDataset(Dataset):
    def __init__(self, startPath: Path, nrLabels: int, transForm: FeatureExtractor = None, cachePath: Path = None, filter: list[str] = None):
        self.featureExtractor = transForm
        if cachePath is not None:
            transForm.setCachePath(cachePath)
        generator = transForm.extractDirectory(startPath=startPath, filter=filter)

        # Create an array and store the data as (feature, labelNumeric)
        self.dataArray = []
        self.labelArray = []

        # Convert every file into a signal and labels
        while True:
            try:
                # Get all the extracted features and labels in string form
                signal, labelName = next(generator)

                if labelName not in self.labelArray:
                    self.labelArray.append(labelName)

                labelCode = np.zeros(nrLabels, dtype=float)
                target = self.labelArray.index(labelName)

                # Append the acquired data to the array
                self.dataArray.append([signal, target])
            except StopIteration:
                break

    def __getitem__(self, index):
        row = self.dataArray[index]
        return row[0], row[1]

    def __len__(self):
        return len(self.dataArray)


if __name__ == '__main__':
    filterExtr = FeatureExtractorTKEO()
    noiseProfilePath = utils.getDataRoot().joinpath(r"recordings\noiseProfile\noiseProfile1.wav")
    cachePath: Path = utils.getDataRoot().joinpath(r"cache\TKEO")
    filterExtr.noiseProfile = noiseProfilePath
    dataset: FootstepDataset = FootstepDataset(utils.getDataRoot().joinpath("recordings"), 9, filterExtr, cachePath)
    dataloader = DataLoader(dataset)
