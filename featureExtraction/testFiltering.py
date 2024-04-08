import os

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from scipy.io import wavfile
import sounddevice as sd

from featureExtraction.FeatureExtractor import FeatureExtractor, FeatureExtractorTKEO


def filterTest(extractor: FeatureExtractor, path: str):

    for file in os.listdir(path):
        if file.endswith(".wav") and file != "noiseProfile1.wav":
            figure, axis = plt.subplots(2, 1)
            # Combine filepath with current file
            filePath = path + "\\" + file

            fs, soundSignal = wavfile.read(filePath)
            time = linspace(0, len(soundSignal), fs * 5)
            # sd.play(soundSignal, fs)
            # sd.wait()

            filteredSound, SNR = extractor.filter(soundSignal, fs)
            print(f"Input size: {len(soundSignal)}, output size: {len(filteredSound)}")
            # sd.play(filteredSound, fs)
            # sd.wait()

            axis[0].plot(time, soundSignal)
            axis[0].set_title(file)
            axis[1].plot(time[0:len(filteredSound)], filteredSound)
            axis[1].set_title(file + f" SNR: {SNR:.2f}")

            plt.show()


def extractorTest(extractor: FeatureExtractor):
    startPath = "testFiltering1"
    # figure, axis = plt.subplots(1, amSounds)
    counter = 0
    for file in os.listdir(startPath):
        if file.endswith(".wav") and file != "noiseProfile.wav":
            # Combine filepath with current file
            filePath = startPath + "\\" + file

            fs, soundSignal = wavfile.read(filePath)
            time = np.arange(0, len(soundSignal) / fs, 1 / fs)
            # sd.play(soundSignal, fs)
            # sd.wait()

            filteredSound, SNR = extractor.filter(soundSignal, fs)
            filteredSound = np.array(filteredSound)
            filteredSound = np.squeeze(filteredSound)
            print(f"Input size: {len(soundSignal)}, output size: {len(filteredSound)}")
            # sd.play(filteredSound, fs)
            # sd.wait()

            filteredFeatures = np.array(extractor.extract(filteredSound, fs))
            features = np.array(extractor.extract(soundSignal, fs))

            plt.plot(time[0:len(features)], features, color="b", label="normal")
            plt.plot(time[0:len(filteredFeatures)], filteredFeatures, color="orange", label="filtered")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Energy")
            plt.title("Feature extraction of normal and filtered signal")
            plt.show()

            counter += 1


if __name__ == '__main__':
    fExtractor = FeatureExtractorTKEO()
    fExtractor.noiseProfile = r"testData\testFiltering\noiseProfile1.wav"
    startPath = "testData\\testFiltering"
    filterTest(fExtractor, startPath)
