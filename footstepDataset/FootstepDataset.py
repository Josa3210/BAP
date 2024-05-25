import glob
import logging
import math
import os
from pathlib import Path
from typing import List

import matlab.engine
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader


from Tools.CustomLogger import CustomLogger
from FeatureExtraction import Transforms
from FeatureExtraction.FeatureCacher import FeatureCacher
from FeatureExtraction.FeatureExtractor import FeatureExtractor, Filter, FeatureExtractorTKEO
from FeatureExtraction.Transforms import AddOffset


def plotFuncs(signals, fs, title: str = None, block: bool = True):
    time = np.linspace(0, 4, len(signals[0]))
    fig, axs = plt.subplots(len(signals), 1)

    for i in range(len(signals)):
        axs[i].plot(time, signals[i])

    if title is not None:
        plt.suptitle(title)
    plt.show(block=block)


class FootstepDataset(Dataset):
    """
    Custom dataset for footstep data. This dataset reads in all the .wav file from a specified directory (dataSource) and convert it into a dataset.
    The labels are the first part of the title separated by a "_"
    The type of extracted feature is defined by the fExtractor given in the constructor.
    Additionally, it is possible to add a transformation by adding a Transforms object
    and add noise by specifying a non-zero natural number for addNoiseFactor.
    This will add a normal distributed noise with std of the signal times the addNoiseFactor.

    Args:
        dataSource (Path): Path to the data source directory containing audio files.
        fExtractor (FeatureExtractor, optional): Feature extractor. Defaults to None.
        transformer (Transforms, optional): Data transformation. Defaults to None.
        cachePath (Path, optional): Path for caching features. Defaults to None.
        labelFilter (list[str], optional): Filter for specific labels. Defaults to None.
        addNoiseFactor (float, optional): Noise factor. Defaults to 0.

    Attributes:
        featureExtractor (FeatureExtractor): The feature extractor used for signal processing.
        transformer (Transforms): Data transformation (if provided).
        logger (Logger): Logger for logging information.
        cacher (FeatureCacher): Caching utility for storing filtered signals.
        featureSize (tuple): Shape of the extracted features.
        dataset (list): List of data samples, each containing a feature tensor and label.

    Methods:
        __init__(self, dataSource, fExtractor, transformer, cachePath, labelFilter, addNoiseFactor):
            Initializes the dataset by extracting features from audio files.
        __getitem__(self, index):
            Retrieves an item from the dataset.
        __len__(self):
            Returns the number of samples in the dataset.

    Example Usage:
        # Create a dataset instance
        dataset = FootstepDataset(dataSource=Path("path/to/data"), fExtractor=myFeatureExtractor)

        # Access a sample
        feature, label = dataset[0]
    """

    def __init__(self, dataSource: Path = None, fExtractor: FeatureExtractor = None, transformer: Transforms = None, cachePath: Path = None, labelFilter: list[str] = None, addNoiseFactor: float = 0):
        self.featureExtractor = fExtractor
        self.transformer = transformer
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if fExtractor is None:
            self.featureExtractor = Filter()

        self.addNoiseFactor = addNoiseFactor
        self.SNR = []
        self.maxVal = 0

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

        if dataSource is not None:
            generator = self.extractDirectory(startPath=dataSource)

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
                    self.maxVal = np.max(dataArray)
                    dataArray /= self.maxVal
                    self.dataset = [[x, y] for x, y in zip(dataArray, labelArray)]
                    self.featureSize = dataArray.shape[1:]
                    self.featureExtractor.shutdown()
                    break

    def __getitem__(self, index):
        row = self.dataset[index]
        return row[0], row[1]

    def extractDirectory(self, startPath: Path):
        """
            Extract features from audio files in the specified directory.

            Args:
                startPath (Path): Path to the directory containing audio files.

            Yields:
                tuple: Tuple containing the extracted feature tensor and label.
        """
        self.featureExtractor.start()
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

            if os.path.exists(cachePath) and self.addNoiseFactor == 0:
                # Read data from cache file
                filteredSignal, fs = self.cacher.load(cachePath)
                self.logger.debug(f"Reading from {cachePath}")

            else:
                # Read wav file
                fs, signal = wavfile.read(filePath)

                if self.addNoiseFactor != 0:
                    varNoise = np.var(signal) * self.addNoiseFactor
                    signal = signal + np.random.normal(0, np.sqrt(varNoise), len(signal))
                    varSignal = np.var(signal)
                    SNR = 20*math.log10(varSignal/varNoise)
                    self.SNR.append(SNR)
                # Filter the result
                filteredSignal = self.featureExtractor.filter(signal, fs)

                if self.addNoiseFactor == 0:
                    # Create a cache file for future extraction
                    self.cacher.cache({"t": filteredSignal, "f": fs}, cachePath)
                    self.logger.debug(f"Reading from {filePath}")

            if self.transformer is not None:
                transformGen = self.transformer.transform(filteredSignal, fs)
                for j in range(self.transformer.amount):
                    transformedSignal = next(transformGen)
                    # Send data to Matlab and receive the transformed signal
                    result = self.featureExtractor.transform(transformedSignal, fs)

                    # Convert to tensor and flatten to remove 1 dimension
                    torchResult = torch.Tensor(result)
                    yield torchResult, label
            else:
                # Send data to Matlab and receive the transformed signal
                result = self.featureExtractor.transform(filteredSignal, fs)

                # Convert to tensor and flatten to remove 1 dimension
                torchResult = torch.Tensor(result)
                yield torchResult, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    filterExtr = Filter()
    transformer = AddOffset(amount=2)

    noiseProfilePath = PathFinder.getDataRoot().joinpath(r"noiseProfile\noiseProfile1.wav")
    recordingsPath = PathFinder.getDataRoot().joinpath("recordings")
    testCachePath: Path = PathFinder.getDataRoot().joinpath(r"cache\TKEO")
    # offsetAdder = AddOffset(2, maxTimeOffset=0.5)
    filterExtr.noiseProfile = noiseProfilePath
    personFilter = ["ann","Lieve"]
    datasetTransformed: FootstepDataset = FootstepDataset(recordingsPath, labelFilter=personFilter, fExtractor=filterExtr, transformer=transformer, cachePath=testCachePath, addNoiseFactor=0)

    print(len(datasetTransformed.dataset))

    dataloaderTransformed = DataLoader(datasetTransformed, batch_size=4)

    trainingFeatures, trainingLabels = next(iter(dataloaderTransformed))
    plotFuncs(np.array(trainingFeatures), 44100, "Batch")
