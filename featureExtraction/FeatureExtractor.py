from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

import utils
from scipy.io import wavfile
import os.path
import matlab.engine

from CustomLogger import CustomLogger


class FeatureExtractor(ABC):
    @abstractmethod
    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None, engine: matlab.engine.MatlabEngine = None):
        # Get the directory where this file is locate and add the trainingPath to the function to it
        self.funcPath = utils.getFunctionPath().joinpath(funcPath)
        self.filterPath = utils.getFunctionPath().joinpath(filterPath)
        self.logger = CustomLogger.getLogger(__name__)

        # Check if the trainingPath to the featureExtraction.m file exists
        if not os.path.isfile(self.funcPath):
            self.logger.error(f"{self.funcPath} has not been found! Please add this file or specify location in the constructor (funcPath=)")
            return

        # Check if the trainingPath to the filter file exists
        if not os.path.isfile(self.filterPath):
            self.logger.error(f"{self.filterPath} has not been found! Please add this file or specify location in the constructor (filterPath=)")
            return

        # Matlab engine for running the necessary functions
        if engine is None:
            self.eng = matlab.engine.start_matlab()
        else:
            self.eng = engine

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
            self.logger.error(f"No noise profile found")
            return
        pass

    @abstractmethod
    def extract(self, signal, fs):
        pass


class FeatureExtractorTKEO(FeatureExtractor):
    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None, engine: matlab.engine.MatlabEngine = None):
        super().__init__(funcPath, filterPath, noiseProfile, engine=engine)

    def extract(self, signal, fs):
        # Output size: 176319
        result, newFs = self.eng.extractTKEOFeatures(signal, fs, nargout=2)
        result = np.array(result).squeeze()
        return result, newFs

    def filter(self, signal, fs):
        if self.noiseProfile is None:
            self.logger.error(f"No noise profile found")
            return

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal, SNR = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap, nargout=2)
        filteredSignal = np.array(filteredSignal).squeeze()
        return filteredSignal, SNR


class FeatureExtractorSTFT(FeatureExtractor):
    def __init__(self, funcPath: str = "extractSTFTFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None, nFFT: int = 4096, bound: int = 50, engine: matlab.engine.MatlabEngine = None):
        super().__init__(funcPath, filterPath, noiseProfile, engine=engine)
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
        super().filter(signal, fs)

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal, SNR = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap, nargout=2)
        filteredSignal = np.array(filteredSignal).squeeze()
        return filteredSignal, SNR

    def extract(self, signal, fs):
        return self.extractNormal(signal, fs)

    def extractLogScale(self, signal, fs):
        self.logScale = True
        extracted, fs = self.eng.extractSTFTFeatures(signal, fs, self.nFFT, self.bound, self.logScale, False, nargout=2)
        return extracted, fs

    def extractNormal(self, signal, fs):
        self.logScale = False
        extracted, fs = self.eng.extractSTFTFeatures(signal, fs, self.nFFT, self.bound, self.logScale, False, nargout=2)
        extracted = np.array(extracted).squeeze()
        return extracted, fs


class Filter(FeatureExtractor):

    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None, engine: matlab.engine.MatlabEngine = None):
        super().__init__(funcPath, filterPath, noiseProfile,engine=engine)
        self.nFFT = 256
        self.nFramesAveraged = 0
        self.overlap = 0.5  # Standard set to 0.5

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
        super().filter(signal, fs)

        filteredSignal, SNR = self.eng.spectralSubtraction(signal, self.noiseProfile, fs, self.nFFT, self.nFramesAveraged, self.overlap, nargout=2)
        filteredSignal = np.array(filteredSignal).squeeze()
        return filteredSignal, SNR

    def extract(self, signal, fs):
        result = np.array(signal).squeeze()
        return result

    def filterAdv(self, signal, fs, nFFT, nFramesAveraged, overlap):
        self.nFFT = nFFT
        self.nFramesAveraged = nFramesAveraged
        self.overlap = overlap
        filteredSignal, SNR = self.filter(signal, fs)
        return filteredSignal, SNR
