import os
from pathlib import Path

import torch

import utils


class FeatureCacher:
    """
    A utility class for caching and loading features using PyTorch.

    Args:
        startPath (Path, optional): The base directory where cached files will be stored.

    Attributes:
        cachePath (Path): Path to the cache directory.

    Methods:
        - cache(result: Any, cachePath: Path) -> None:
            Caches the given result (usually a PyTorch tensor) to the specified cache path.
        - load(filePathCache: Path) -> tuple[torch.Tensor, int]:
            Loads a cached feature from the given file path.
        - clearCache() -> None:
            Clears all cached files in the cache directory.
        - getCachePath(fileName: Path) -> Path | None:
            Returns the cache path for a given file name.

    Example usage:
        cacher = FeatureCacher(startPath="/path/to/cache")
        features = torch.tensor([1.0, 2.0, 3.0])
        cacher.cache(features, cachePath="/path/to/cache/my_feature.cache")
        loaded_features, fs = cacher.load(filePathCache="/path/to/cache/my_feature.cache")
    """

    def __init__(self, startPath: Path = None):
        self.cachePath: Path = None
        if startPath is not None:
            self.cachePath: Path = startPath.joinpath("cache")

    @property
    def cachePath(self):
        return self._cachePath

    @cachePath.setter
    def cachePath(self, value: Path | None):
        if value is None:
            self._cachePath = None
            return

        if not value.exists():
            os.makedirs(value)
        self._cachePath = value

    @staticmethod
    def cache(result, cachePath):
        torch.save(result, cachePath)

    @staticmethod
    def load(filePathCache) -> tuple[torch.Tensor, int]:
        """
        Loads the dictionary from a specific path and extracts the torch.Tensor and sample frequency

        :param filePathCache: path from where to download dictionary
        :return: torch.Tensor with signal values, sample frequency in Hz
        """
        loadedDict = torch.load(filePathCache)
        filteredSignal = loadedDict["t"]
        fs = loadedDict["f"]
        return filteredSignal, fs

    def clearCache(self):
        """
        Clear all .cache files in the cachePath
        """
        if self.cachePath is not None:
            files = self.cachePath.glob("*.cache")
            counter = 0
            for file in files:
                os.remove(self.cachePath.joinpath(file))
                counter += 1
            print(f"Removed {counter} files")

    def getCachePath(self, fileName: Path) -> Path | None:
        """
        Returns the path to the cached file of a specific file.
        :param fileName: Path to the file
        :return: Path to the cached file
        """
        cacheName: str = fileName.parts[-1].split(".")[0] + ".cache"
        cachePath: Path = self.cachePath.joinpath(cacheName)

        return cachePath


if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("testVDB")
    cacher = FeatureCacher(path)
    cacher.clearCache()
