from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader

import utils
from featureExtraction.FeatureExtractor import FeatureExtractor, Filter


class FootstepDataset(Dataset):
    def __init__(self, startPath: Path, nrLabels: int, transForm: FeatureExtractor = None):
        self.featureExtractor = transForm
        generator = transForm.extractDirectory(startPath)

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

                labelCode = np.zeros(nrLabels, dtype=int)
                labelCode[self.labelArray.index(labelName)] = 1

                # Append the acquired data to the array
                self.dataArray.append([signal, labelCode])
            except StopIteration:
                break

    def __getitem__(self, index):
        row = self.dataArray[index]
        return row[0], row[1]

    def __len__(self):
        return len(self.dataArray)


if __name__ == '__main__':
    filterExtr = Filter()
    noiseProfilePath = utils.getDataRoot().joinpath(r"testVDB\noiseProfile\noiseProfile1")
    filterExtr.noiseProfile = noiseProfilePath
    dataset: FootstepDataset = FootstepDataset(utils.getDataRoot().joinpath("testVDB"), 3, filterExtr)
    dataloader = DataLoader(dataset)
