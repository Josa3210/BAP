import glob
import logging
import os
from pathlib import Path

import matlab.engine
import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader

import utils
from CustomLogger import CustomLogger
from featureExtraction import Transforms
from featureExtraction.FeatureCacher import FeatureCacher
from featureExtraction.FeatureExtractor import FeatureExtractor, Filter, FeatureExtractorTKEO
from featureExtraction.Transforms import AddOffset


class FootstepDataset(Dataset):
    def __init__(self, startPath: Path, fExtractor: FeatureExtractor = None, transformer: Transforms = None, cachePath: Path = None, labelFilter: list[str] = None):
        self.featureExtractor = fExtractor
        self.transformer = transformer
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if fExtractor is None:
            self.featureExtractor = Filter()

        self.cacher = FeatureCacher()
        if cachePath is not None:
            self.cacher.cachePath = cachePath

        # Create an array and store the data as (feature, labelNumeric)
        self.featureSize = 0
        dataArray = []
        labelArray = []
        self.labelStrings = []
        self.dataset = []
        self.labelFilter = labelFilter

        generator = self.extractDirectory(startPath=startPath)
        # Convert every file into a signal and labels
        while True:
            try:
                # Get all the extracted features and labels in string form
                signal, labelName = next(generator)

                if labelName not in self.labelStrings:
                    self.labelStrings.append(labelName)

                target = self.labelStrings.index(labelName)
                target = torch.tensor(target)
                labelArray.append(target)

                # Append the acquired data to the array
                dataArray.append(signal)
            except StopIteration:
                dataArray = np.array(dataArray).squeeze().astype('float32')
                maxVal = np.max(dataArray)
                dataArray /= maxVal
                self.dataset = [[x, y] for x, y in zip(dataArray, labelArray)]
                self.featureSize = dataArray.shape[1:]
                break

    def __getitem__(self, index):
        row = self.dataset[index]
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
            if "noiseProfile" in label or (self.labelFilter is not None and label not in self.labelFilter):
                continue

            # Check if there is a cached version of the filtered sound
            cachePath = self.cacher.getCachePath(filePath)
            if os.path.exists(cachePath) and self.transformer is None:
                # Read data from cache file


                self.logger.debug(f"Reading from {cachePath}")

            else:
                # Read wav file
                fs, signal = wavfile.read(filePath)

                # Filter the result
                filteredSignal, SNR = self.featureExtractor.filter(signal, fs)

                # Create a cache file for future extraction
                self.cacher.cache({"t": filteredSignal, "f": fs}, cachePath)
                self.logger.debug(f"Reading from {filePath}")

            if self.transformer is not None:
                transformGen = self.transformer.transform(filteredSignal, fs)
                for j in range(self.transformer.amount):
                    transformedSignal = next(transformGen)[0]

                    # Send data to Matlab and receive the transformed signal
                    result, newFS = self.featureExtractor.extract(transformedSignal, fs)

                    # Convert to tensor and flatten to remove 1 dimension
                    torchResult = torch.Tensor(result)
                    yield torchResult, label
            else:
                # Send data to Matlab and receive the transformed signal
                result, newFS = self.featureExtractor.extract(filteredSignal, fs)

                # Convert to tensor and flatten to remove 1 dimension
                torchResult = torch.Tensor(result)
                yield torchResult, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    engine = matlab.engine.start_matlab()
    filterExtr = FeatureExtractorTKEO(engine=engine)
    transformer = AddOffset(engine=engine, amount=5)

    noiseProfilePath = utils.getDataRoot().joinpath(r"noiseProfile\noiseProfile1.wav")
    recordingsPath = utils.getDataRoot().joinpath("recordings")
    testCachePath: Path = utils.getDataRoot().joinpath(r"cache\TKEO")
    offsetAdder = AddOffset(5, maxTimeOffset=0.5)
    filterExtr.noiseProfile = noiseProfilePath
    datasetTransformed: FootstepDataset = FootstepDataset(recordingsPath, fExtractor=filterExtr, transformer=transformer, cachePath=testCachePath)
    datasetNormal: FootstepDataset = FootstepDataset(recordingsPath, fExtractor=filterExtr, cachePath=testCachePath)

    print(len(datasetTransformed.dataset))
    print(len(datasetNormal.dataset))

    dataloaderTransformed = DataLoader(datasetTransformed, batch_size=32)
    dataloaderNormal = DataLoader(datasetNormal, batch_size=32)

    for i, batch in enumerate(dataloaderTransformed):
        print(batch.size())
