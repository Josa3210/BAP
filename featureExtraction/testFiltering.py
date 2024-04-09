import os

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from scipy.io import wavfile

from featureExtraction.FeatureExtractor import FeatureExtractor, FeatureExtractorTKEO, Filter


def filterTest(extractor1: FeatureExtractor, path: str):
    for file in os.listdir(path):
        if file.endswith(".wav") and file != "noiseProfile1.wav":
            figure, axis = plt.subplots(2, 1)
            # Combine filepath with current file
            filePath = path + "\\" + file

            fs, soundSignal = wavfile.read(filePath)
            time = linspace(0, len(soundSignal), fs * 5)

            filteredSound1, SNR1 = extractor1.filter(soundSignal, fs)

            axis[0].plot(time, soundSignal)
            axis[0].set_title(file)
            axis[0].set_yticks(np.arange(-0.5, 0.5, 0.1))
            axis[0].grid()
            axis[1].plot(time[0:len(filteredSound1)], filteredSound1)
            axis[1].set_title(file + f" SNR: {SNR1:.2f}")
            axis[1].set_yticks(np.arange(-0.2, 0.2, 0.05))
            axis[1].grid()

            plt.show()


def extractorTest(extractor: FeatureExtractor, path: str):
    counter = 0
    for file in os.listdir(path):
        if file.endswith(".wav") and file != "noiseProfile.wav":
            # Combine filepath with current file
            filePath = path + "\\" + file

            fs, soundSignal = wavfile.read(filePath)
            time = np.arange(0, len(soundSignal) / fs, 1 / fs)

            filteredSound, SNR = extractor.filter(soundSignal, fs)
            filteredSound = np.array(filteredSound)
            filteredSound = np.squeeze(filteredSound)

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


def refineFiltering(path: str, noiseProfile: str):
    extractor1: Filter = Filter()
    extractor1.noiseProfile = noiseProfile
    figure, axis = plt.subplots(2, 3)
    nPerson = 0
    for file in os.listdir(path):
        if "Jan" in file and file.endswith(".wav"):
            # Combine filepath with current file
            filePath = path + "\\" + file

            fs, soundSignal = wavfile.read(filePath)
            time = linspace(0, len(soundSignal), fs * 5)

            filteredSound, SNR = extractor1.filterAdv(soundSignal, fs, 512, 6, 0.5)

            time1 = time[0:len(filteredSound)]

            row = 0
            plotAxis(axis, row, nPerson, soundSignal, 0, time, file)
            axis[0, nPerson].set_yticks(np.arange(-0.5, 0.5, 0.1))
            row += 1
            plotAxis(axis, row, nPerson, filteredSound, SNR, time1, file)
            nPerson += 1

    plt.show()


def plotAxis(axis, row: int, columm: int, signal, SNR, time, name: str):
    axis[row, columm].plot(time, signal)
    axis[row, columm].set_title(name + f" SNR: {SNR:.2f}")
    axis[row, columm].set_yticks(np.arange(-0.15, 0.15, 0.05))
    axis[row, columm].get_xaxis().set_visible(False)
    axis[row, columm].grid()


if __name__ == '__main__':
    # fExtractor1 = FeatureExtractorTKEO()
    # fExtractor1.noiseProfile = r"testData\testFilteringVDB\noiseProfile1.wav"

    noiseProfile = r"testData\testFilteringVDB\noiseProfile1.wav"
    testPath = "testData\\testFilteringVDB"
    refineFiltering(testPath, noiseProfile)
