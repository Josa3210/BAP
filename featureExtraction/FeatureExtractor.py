import glob
from abc import ABC, abstractmethod
from pathlib import Path

import utils
import numpy as np
from scipy.io import wavfile
import os.path
import matlab.engine
import torch

from featureExtraction.FeatureCacher import FeatureCacher


class FeatureExtractor(ABC):
    @abstractmethod
    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None):
        # Get the directory where this file is locate and add the path to the function to it
        self.funcPath = utils.getFunctionPath().joinpath(funcPath)
        self.filterPath = utils.getFunctionPath().joinpath(filterPath)

        # Check if the path to the featureExtraction.m file exists
        if not os.path.isfile(self.funcPath):
            print(f"{self.funcPath} has not been found! Please add this file or specify location in the constructor (funcPath=)")
            return

        # Check if the path to the filter file exists
        if not os.path.isfile(self.filterPath):
            print(f"{self.filterPath} has not been found! Please add this file or specify location in the constructor (filterPath=)")
            return

        # Matlab engine for running the necessary functions
        self.eng = matlab.engine.start_matlab()

        # Set matlab directory to current directory
        self.eng.cd(str(utils.getFunctionPath()))

        # Params for filtering
        self.noiseProfile = noiseProfile

        pass

    @property
    def noiseProfile(self):
        return self._noiseProfile

    @noiseProfile.setter
    def noiseProfile(self, value: Path):
        if isinstance(value, Path):
            fs, signal = wavfile.read(str(value))
            self.noiseProfile = signal
        else:
            self._noiseProfile = value

    @abstractmethod
    def filter(self, signal, fs):
        if self.noiseProfile is None:
            print(f"No noise profile found")
            return

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal, SNR = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap, nargout=2)
        return filteredSignal, SNR

    @abstractmethod
    def extract(self, signal, fs):
        pass


class FeatureExtractorTKEO(FeatureExtractor):
    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None):
        super().__init__(funcPath, filterPath, noiseProfile)

    def extract(self, signal, fs):
        # Output size: 176319
        result = self.eng.extractTKEOFeatures(signal, fs)
        return result

    def filter(self, signal, fs):
        if self.noiseProfile is None:
            print(f"No noise profile found")
            return

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal, SNR = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap, nargout=2)
        return filteredSignal, SNR


class FeatureExtractorSTFT(FeatureExtractor):
    def __init__(self, funcPath: str = "extractSTFTFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None, nFFT: int = 4096, bound: int = 50):
        super().__init__(funcPath, filterPath, noiseProfile)
        self.nFFT = nFFT
        self.bound = bound
        self.logScale = False

    @property
    def nFFT(self):
        return self._nFFT

    @nFFT.setter
    def nFFT(self, value):
        self._nFFT = value

    @property
    def bound(self):
        return self._bound

    @bound.setter
    def bound(self, bound):
        self._bound = bound

    @property
    def logScale(self):
        return self._logScale

    @logScale.setter
    def logScale(self, logScale: bool):
        self._logScale = logScale

    def filter(self, signal, fs):
        if self.noiseProfile is None:
            print(f"No noise profile found")
            return

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal, SNR = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap, nargout=2)
        return filteredSignal, SNR

    def extract(self, signal, fs):
        return self.extractNormal(signal, fs)

    def extractLogScale(self, signal, fs):
        self.logScale = True
        extracted = self.eng.extractSTFTFeatures(signal, fs, self.nFFT, self.bound, self.logScale, False)
        return extracted

    def extractNormal(self, signal, fs):
        self.logScale = False
        extracted = self.eng.extractSTFTFeatures(signal, fs, self.nFFT, self.bound, self.logScale, False)
        return extracted


class Filter(FeatureExtractor):

    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None):
        super().__init__(funcPath, filterPath, noiseProfile)
        self.nFFT = 256
        self.nFramesAveraged = 0
        self.overlap = 0.5  # Standard set to 0.5

        pass

    @property
    def nFFT(self):
        return self._nFFT

    @nFFT.setter
    def nFFT(self, nFFT):
        self._nFFT = nFFT

    @property
    def nFramesAveraged(self):
        return self._nFramesAveraged

    @nFramesAveraged.setter
    def nFramesAveraged(self, value):
        self._nFramesAveraged = value

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value

    def filter(self, signal, fs):
        if self.noiseProfile is None:
            print(f"No noise profile found")
            return

        filteredSignal, SNR = self.eng.spectralSubtraction(signal, self.noiseProfile, fs, self.nFFT, self.nFramesAveraged, self.overlap, nargout=2)
        return filteredSignal, SNR

    def extract(self, signal, fs):
        return signal

    def filterAdv(self, signal, fs, nFFT, nFramesAveraged, overlap):
        self.nFFT = nFFT
        self.nFramesAveraged = nFramesAveraged
        self.overlap = overlap
        filteredSignal, SNR = self.filter(signal, fs)
        return filteredSignal, SNR
