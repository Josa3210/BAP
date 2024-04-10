import os
from pathlib import Path

import torch


class FeatureCacher:
    def __init__(self, startPath: Path = None):
        self.cachePath = None
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
            os.mkdir(value)
        self._cachePath = value

    def cache(self, result, cachePath):
        torch.save(result, cachePath)

    def load(self, filePathCache) -> torch.Tensor:
        return torch.load(filePathCache)

    def getCachePath(self, fileName: Path) -> Path | None:
        cacheName: str = fileName.parts[-1].split(".")[0] + ".cache"
        cachePath: Path = self.cachePath.joinpath(cacheName)

        return cachePath
