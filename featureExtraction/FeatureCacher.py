import os
from pathlib import Path

import torch

import utils


class FeatureCacher:
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

    def cache(self, result, cachePath):
        torch.save(result, cachePath)

    def load(self, filePathCache) -> torch.Tensor:
        return torch.load(filePathCache)

    def clearCache(self):
        if self.cachePath is not None:
            files = self.cachePath.glob("*.cache")
            counter = 0
            for file in files:
                os.remove(self.cachePath.joinpath(file))
                counter += 1
            print(f"Removed {counter} files")

    def getCachePath(self, fileName: Path) -> Path | None:
        cacheName: str = fileName.parts[-1].split(".")[0] + ".cache"
        cachePath: Path = self.cachePath.joinpath(cacheName)

        return cachePath


if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("testVDB")
    cacher = FeatureCacher(path)
    cacher.clearCache()
