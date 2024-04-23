import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader

import utils
from customLogger import CustomLogger
from featureExtraction.FeatureCacher import FeatureCacher
from featureExtraction.FeatureExtractor import FeatureExtractor, Filter, FeatureExtractorTKEO


class FootstepDataset(Dataset):
    def __init__(self, startPath: Path, transform: FeatureExtractor = None, cachePath: Path = None, labelFilter: list[str] = None):
        self.featureExtractor = transform
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if transform is None:
            self.featureExtractor = Filter()

        self.cacher = FeatureCacher()
        if cachePath is not None:
            self.cacher.cachePath = cachePath

        # Create an array and store the data as (feature, labelNumeric)
        self.dataArray = []
        self.labelArray = []

        self.labelFilter = labelFilter

        generator = self.extractDirectory(startPath=startPath)
        # Convert every file into a signal and labels
        while True:
            try:
                # Get all the extracted features and labels in string form
                signal, labelName = next(generator)

                if labelName not in self.labelArray:
                    self.labelArray.append(labelName)

                target = self.labelArray.index(labelName)

                # Append the acquired data to the array
                self.dataArray.append([signal, target])
            except StopIteration:
                break

    def __getitem__(self, index):
        row = self.dataArray[index]
        return row[0], row[1]

        # Extract all the .wav files and convert them into a readable file

    def extractDirectory(self, startPath: Path):
        searchPath = str(startPath) + r"\**\*.wav"
        for fileName in glob.glob(pathname=searchPath, recursive=True):

            # Combine filepath with current file
            filePath = startPath.joinpath(fileName)

            # Extract label
            label = filePath.parts[-1].split("_")[0]

            # Ignore noiseProfiles
            if "noiseProfile" in label or (label is not None and label not in self.labelFilter):
                continue

            # Check if there is a cached version
            cachePath = self.cacher.getCachePath(filePath)
            if os.path.exists(cachePath):
                # Read data from cache file
                torchResult = self.cacher.load(cachePath)

                self.logger.debug(f"Reading from {cachePath}")

            else:
                # Read wav file
                fs, signal = wavfile.read(filePath)

                # Filter the result
                filteredSignal, SNR = self.featureExtractor.filter(signal, fs)
                filteredSignal = np.array(filteredSignal).squeeze()
                # Send data to Matlab and receive the transformed signal
                result = self.featureExtractor.extract(filteredSignal, fs)
                result = np.array(result).squeeze()
                # Convert to tensor and flatten to remove 1 dimension
                # torchResult = torch.flatten(torch.Tensor(result))
                torchResult = torch.Tensor(result)

                # Create a cache file for future extraction
                self.cacher.cache(torchResult, cachePath)

                self.logger.debug(f"Reading from {filePath}")

            yield torchResult, label

    def __len__(self):
        return len(self.dataArray)


if __name__ == '__main__':
    filterExtr = FeatureExtractorTKEO()
    noiseProfilePath = utils.getDataRoot().joinpath(r"recordings\noiseProfile\noiseProfile1.wav")
    testCachePath: Path = utils.getDataRoot().joinpath(r"cache\TKEO")
    filterExtr.noiseProfile = noiseProfilePath
    dataset: FootstepDataset = FootstepDataset(utils.getDataRoot().joinpath("recordings"), filterExtr, testCachePath)
    dataloader = DataLoader(dataset)
