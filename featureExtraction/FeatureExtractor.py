from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from Tools import utils
from scipy.io import wavfile
import os.path
import matlab.engine

from Tools.CustomLogger import CustomLogger


class FeatureExtractor(ABC):
    """
    Base class for every FeatureExtractor class. Implements the necessary methods for a FeatureExtractor to function in the FootstepDataset class.
    Main purpose is to convert a .wav file into the specified form (Filtered, TKEO, STFT) depending on the instance of this base class.

    Args:
        funcPath (Path): Path to the main extraction function. This must be a matlab function (extension .m).
        filterPath (Path): Path to the filtering function. This must be a matlab function (extension .m).
        noiseProfile (list[float], optional): Sample of the background noise, used for spectral subtraction in the filtering.

    Attributes:
        funcPath (Path): Path to the main extraction function.
        filterPath (Path): Path to the filtering function.
        logger: Logger instance for debugging.
        eng: Matlab engine for running the matlab scripts.
        noiseProfile: Sample of background noise used for filtering.

    Methods:
        - filter(signal: numpy.ndarray, fs: int) -> None:
            Applies filtering to the input signal using the noise profile.
        - transform(signal: numpy.ndarray, fs: int) -> None:
            Transforms the input signal (implementation specific).
        - shutdown() -> None:
            Exits the MATLAB engine.
        - start() -> None:
            Starts the MATLAB engine and sets the working directory.


    Example usage:
        # Create a custom feature extractor
        class MyFeatureExtractor(FeatureExtractor):
            def transform(self, signal, fs):
                # Implement feature extraction logic here
                pass

        # Instantiate the custom featureExtractor
        extractor = MyFeatureExtractor(funcPath="my_extraction_function.m", filterPath="my_filter_function.m")
        extractor.start()
        # Use the extractor to process audio signals
        # ...
        extractor.shutdown()
    """

    @abstractmethod
    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None):
        # Get the directory where this file is locate and add the trainingPath to the function to it
        self.funcPath = utils.getFunctionPath().joinpath(funcPath)
        self.filterPath = utils.getFunctionPath().joinpath(filterPath)
        self.logger = CustomLogger.getLogger(__name__)
        self.eng = None

        # Check if the trainingPath to the featureExtraction.m file exists
        if not os.path.isfile(self.funcPath):
            self.logger.error(f"{self.funcPath} has not been found! Please add this file or specify location in the constructor (funcPath=)")
            return

        # Check if the trainingPath to the filter file exists
        if not os.path.isfile(self.filterPath):
            self.logger.error(f"{self.filterPath} has not been found! Please add this file or specify location in the constructor (filterPath=)")
            return

        # Params for filtering
        self.noiseProfile = noiseProfile

        pass

    @property
    def noiseProfile(self):
        return self._noiseProfile

    @noiseProfile.setter
    def noiseProfile(self, value: Path):
        # If the given value is a path, then the noiseProfile must be loaded from that file
        if isinstance(value, Path):
            fs, signal = wavfile.read(str(value))
            self.noiseProfile = signal
        # Else the value must be a sequence of numbers that corresponds to the noiseProfile signal itself
        else:
            self._noiseProfile = value

    @abstractmethod
    def filter(self, signal, fs):
        # Always check if there is a noiseProfile
        if self.noiseProfile is None:
            self.logger.error(f"No noise profile found")
            return
        pass

    @abstractmethod
    def transform(self, signal, fs):
        pass

    def shutdown(self):
        # Shutdown the engine to clear memory space
        self.eng.exit()

    def start(self):
        # Open the engine (takes a few seconds)
        self.eng = matlab.engine.start_matlab()
        # Set matlab directory to given directory
        self.eng.cd(str(utils.getFunctionPath()))


class FeatureExtractorTKEO(FeatureExtractor):
    """
    A feature extractor class that computes features using the Teager-Kaiser Energy Operator (TKEO) method.

    Args:
        funcPath (str, optional): Path to the main extraction function (MATLAB script).
        filterPath (str, optional): Path to the filtering function (MATLAB script).
        noiseProfile (list[float], optional): Sample of background noise for spectral subtraction.

    Methods:
        - transform(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, int]:
            Computes TKEO features from the input signal.
        - filter(signal: numpy.ndarray, fs: int) -> numpy.ndarray:
            Applies spectral subtraction filtering to the input signal.

    Example usage:
        # Create a TKEO feature extractor
        tkeo_extractor = FeatureExtractorTKEO(funcPath="extractTKEOFeatures.m", filterPath="spectralSubtraction.m")

        # Process an audio signal
        audio_signal = np.random.randn(44100)  # Example audio signal
        filtered_signal = tkeo_extractor.filter(audio_signal, fs=44100)
        tkeo_features, fs_tkeo = tkeo_extractor.transform(filtered_signal, fs=44100)
    """

    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None):
        super().__init__(funcPath, filterPath, noiseProfile)

    def transform(self, signal, fs):
        result = self.eng.extractTKEOFeatures(signal, fs)
        result = np.array(result).squeeze()
        return result

    def filter(self, signal, fs):
        if self.noiseProfile is None:
            self.logger.error(f"No noise profile found")
            return

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap)
        filteredSignal = np.array(filteredSignal).squeeze()
        return filteredSignal


class FeatureExtractorSTFT(FeatureExtractor):
    """
    A feature extractor class that computes Short-Time Fourier Transform (STFT) features.

    Args:
        funcPath (str, optional): Path to the main extraction function (MATLAB script).
        filterPath (str, optional): Path to the filtering function (MATLAB script).
        noiseProfile (list[float], optional): Sample of background noise for spectral subtraction.
        nFFT (int, optional): Number of points for the FFT (default is 4096).
        bound (int, optional): Frequency bound for STFT features (default is 50).

    Attributes:
        nFFT (int): Number of points for the FFT.
        bound (int): Frequency bound for STFT features.
        logScale (bool): Flag indicating whether to use a logarithmic scale for STFT features.

    Methods:
        - filter(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, float]:
            Applies spectral subtraction filtering to the input signal.
        - transform(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, int]:
            Computes STFT features from the input signal.
        - transformLogScale(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, int]:
            Computes STFT features using a logarithmic scale.
        - transformNormal(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, int]:
            Computes STFT features without a logarithmic scale.

    Example usage:
        # Create an STFT feature extractor
        stft_extractor = FeatureExtractorSTFT(funcPath="extractSTFTFeatures.m", filterPath="spectralSubtraction.m")

        # Process an audio signal
        audio_signal = np.random.randn(44100)  # Example audio signal
        filtered_signal, snr = stft_extractor.filter(audio_signal, fs=44100)
        stft_features, fs_stft = stft_extractor.transform(filtered_signal, fs=44100)
    """

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
        super().filter(signal, fs)

        profile = self.noiseProfile
        nFFT = 256
        nFramesAveraged = 0
        overlap = 0.5  # Standard set to 0.5
        filteredSignal = self.eng.spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap)
        filteredSignal = np.array(filteredSignal).squeeze()
        return filteredSignal

    def transform(self, signal, fs):
        return self.transformNormal(signal, fs)

    def transformLogScale(self, signal, fs):
        self.logScale = True
        transformed= self.eng.extractSTFTFeatures(signal, fs, self.nFFT, self.bound, self.logScale)
        return transformed

    def transformNormal(self, signal, fs):
        self.logScale = False
        transformed = self.eng.extractSTFTFeatures(signal, fs, self.nFFT, self.bound, self.logScale)
        transformed = np.array(transformed).squeeze()
        return transformed


class Filter(FeatureExtractor):
    """
    A class that performs spectral subtraction filtering on audio signals.

    Args:
        funcPath (str, optional): Path to the main extraction function (MATLAB script).
        filterPath (str, optional): Path to the filtering function (MATLAB script).
        noiseProfile (list[float], optional): Sample of background noise for spectral subtraction.
        nFFT (int, optional): Number of points for the FFT (default is 256).
        nFramesAveraged (int, optional): Number of frames averaged (default is 0).
        overlap (float, optional): Overlap ratio for spectral subtraction (default is 0.5).

    Methods:
        - filter(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, float]:
            Applies spectral subtraction filtering to the input signal.
        - transform(signal: numpy.ndarray, fs: int) -> tuple[numpy.ndarray, int]:
            Passes the filtered signal through without modification.
        - filterAdv(signal: numpy.ndarray, fs: int, nFFT: int, nFramesAveraged: int, overlap: float) -> tuple[numpy.ndarray, float]:
            Applies advanced spectral subtraction with custom parameters.

    Example usage:
        # Create a spectral subtraction filter
        spectral_filter = Filter(funcPath="extractTKEOFeatures.m", filterPath="spectralSubtraction.m")

        # Process an audio signal
        audio_signal = np.random.randn(44100)  # Example audio signal
        filtered_signal, snr = spectral_filter.filter(audio_signal, fs=44100)
    """

    def __init__(self, funcPath: str = "extractTKEOFeatures.m", filterPath: str = "spectralSubtraction.m", noiseProfile: list[float] = None):
        super().__init__(funcPath, filterPath, noiseProfile)
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

        filteredSignal = self.eng.spectralSubtraction(signal, self.noiseProfile, fs, self.nFFT, self.nFramesAveraged, self.overlap)
        filteredSignal = np.array(filteredSignal).squeeze()
        return filteredSignal

    def transform(self, signal, fs):
        result = np.array(signal).squeeze()
        return result, fs

    def filterAdv(self, signal, fs, nFFT, nFramesAveraged, overlap):
        """
        Same as the normal filter function, but with more options to specify the different values.
        """
        self.nFFT = nFFT
        self.nFramesAveraged = nFramesAveraged
        self.overlap = overlap
        filteredSignal, SNR = self.filter(signal, fs)
        return filteredSignal
