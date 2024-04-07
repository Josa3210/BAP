from matplotlib import pyplot as plt
from numpy import linspace
from scipy.io import wavfile
import os.path
import matlab.engine
import torch
import sounddevice as sd

from featureExtraction.FeatureCacher import FeatureCacher


class FeatureExtractor:

    def __init__(self, funcPath: str = "matlabFunctions/extractFeatures.m", filterPath: str = "matlabFunctions/spectralSubtraction.m", noiseProfile: list[float] = None):
        # Get the directory where this file is locate and add the path to the function to it
        self.funcPath = os.path.dirname(os.path.realpath(__file__)) + "\\" + funcPath
        self.filterPath = os.path.dirname(os.path.realpath(__file__)) + "\\" + filterPath

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
        self.eng.cd(os.path.dirname(os.path.realpath(__file__)) + "\\matlabFunctions")

        self.cacher = FeatureCacher()

        # Params for filtering
        self.noiseProfile = noiseProfile

    @property
    def noiseProfile(self):
        return self._noiseProfile

    @noiseProfile.setter
    def noiseProfile(self, value):
        if isinstance(value, str):
            fs, signal = wavfile.read(os.path.dirname(os.path.realpath(__file__)) + "\\" + value)
            self.noiseProfile = signal
        else:
            self._noiseProfile = value

    # Extract all the .wav files and convert them into a readable file
    def extractDirectory(self, startPath: str):
        for file in os.listdir(startPath):
            if file.endswith(".wav"):
                # Combine filepath with current file
                filePath = startPath + "\\" + file

                # Extract label
                label = file.split(".wav")[0].split("_")[0]

                # Check if there is a cached version
                filePathCache = filePath.split(".")[0] + ".cache"
                if os.path.exists(filePathCache):
                    # Read data from cache file
                    torchResult = self.cacher.load(filePathCache)

                    print(f"Reading from {filePathCache}")

                else:
                    # Read wav file
                    fs, signal = wavfile.read(filePath)

                    # Filter the result
                    filteredSignal = self.filter(signal, fs)

                    # Send data to Matlab and receive the transformed signal
                    result = self.extract(filteredSignal, fs)

                    # Convert to tensor and flatten to remove 1 dimension
                    torchResult = torch.flatten(torch.Tensor(result))

                    # Create a cache file for future extraction
                    self.cacher.cache(torchResult, filePathCache)

                    print(f"Reading from {filePath}")

                yield torchResult, label

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
        result = self.eng.extractFeatures(signal, fs)
        return result
