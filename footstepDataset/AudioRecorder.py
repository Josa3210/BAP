import logging
import os.path
from pathlib import Path

import sounddevice as sd
from matplotlib import pyplot as plt
from numpy import linspace
from scipy.io.wavfile import write
import time

from Tools.CustomLogger import CustomLogger


class CustomAudioRecorder:
    """
        A custom audio recorder class for recording and saving audio data as .wav files.

        Args:
            baseLink (str): The base directory where audio files will be saved.
            sampleRate (int): Sample rate in samples per second.
            channels (int): Number of audio channels.
            level (int, optional): Logging level (default is logging.DEBUG).

        Attributes:
            logger: Logger instance for debugging.
            inputDevice: Default input audio device.
            sampleRate (int): Sample rate in samples per second.
            channels (int): Number of audio channels.
            basePath (Path): Base directory for saving audio files.

        Methods:
            - record(duration: float, playBack: bool = False) -> numpy.ndarray:
                Records audio for the specified duration.
            - save(recording: numpy.ndarray, fileName: str = None) -> None:
                Saves the recorded audio to a file.
            - setInputDevice() -> None:
                Sets the input audio device.
            - setOutputDevice() -> None:
                Sets the output audio device.
            - continuousRecord(showImages: bool = False) -> None:
                Continuously records audio with optional real-time visualization.

        Example usage:
            recorder = CustomAudioRecorder(baseLink="/path/to/save/audio", sampleRate=44100, channels=2)
            recording = recorder.record(duration=5.0, playBack=True)
            recorder.save(recording, fileName="my_audio")
    """

    def __init__(self, baseLink: str, sampleRate: int, channels: int, level: int = logging.DEBUG):
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(level)
        self.inputDevice = sd.default.device[0]
        self.sampleRate = sampleRate  # Sample rate (samples per second)
        self.channels = channels  # Channels
        self.basePath = Path(baseLink)

    @property
    def sampleRate(self):
        return self._sampleRate

    @sampleRate.setter
    def sampleRate(self, sampleRate):
        self._sampleRate = sampleRate
        sd.default.samplerate = sampleRate

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels):
        self._channels = channels
        sd.default.channels = channels

    @property
    def basePath(self):
        return self._baseLink

    @basePath.setter
    def basePath(self, link: Path):
        if not link.is_dir():
            os.makedirs(link)

        self._baseLink = link

    def record(self, duration: float, playBack: bool = False):
        """
        Records audio for a specified duration. If `playBack` is set to `True`, it plays back the recorded audio after recording.
        """
        # Record audio
        self.logger.debug("Recording started")
        recording = sd.rec(int(duration * self.sampleRate))
        sd.wait()  # Wait until recording is finished
        self.logger.debug("Recording stopped")

        if playBack:
            self.logger.debug("Playing back...")
            sd.play(recording)
            sd.wait()

        return recording

    def save(self, recording, fileName: str = None):
        """
        Saves the recorded audio. If no filename is given, it asks the user to enter a filename.
        """
        # Check for filename or ask for one
        if fileName is None:
            fileName = input("Give name to this file:\n")
        fileName += ".wav"

        # Save in folder under given filename
        path = self.basePath.joinpath(fileName)

        write(path, self.sampleRate, recording)
        self.logger.debug(f"Recording saved under '{path}'")

    @staticmethod
    def setInputDevice():
        # Ask for the input device
        index = input("Choose your device: (Copy the name) \n")

        # Set device as default
        sd.default.device[0] = index

    @staticmethod
    def setOutputDevice():
        # Ask for the input device
        index = input("Choose your device: (Copy the name) \n")

        # Set device as default
        sd.default.device[1] = index

    def continuousRecord(self, showImages: bool = False):
        """
        This method continuously records audio until the user indicates they want to stop.
        After each recording, it asks the user if the recording should be saved.
        """
        duration = int(input("Give duration: "))
        t = linspace(0, duration, self.sampleRate * duration)

        name = input("Give name: ")
        counter = int(input("Start at: "))
        userInput = input("Ready to record? (Y or N): ")
        while userInput.capitalize() == "Y":
            recording = self.record(duration=duration, playBack=False)
            if showImages:
                plt.plot(t, recording)
                plt.title("Recording")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.show(block=False)
                plt.pause(1.5)
                plt.close()

            saveInput = input("Save? (Y or N): ")
            if saveInput.capitalize() == "Y":
                self.save(recording, fileName=name + "_" + str(counter))
                counter += 1
            userInput = input("Ready to record? (Y or N): ")


if __name__ == '__main__':
    directory = input("In which directory will you save the files? (Don't forget to add '/' at the end): ")
    fs = input("What sampleRate do you use?: ")
    ch = input("How many channels do you use?: ")
    recorder = CustomAudioRecorder(directory, int(fs), int(ch))

    recordNoise = input("Need to record Noise? (Y or N):")
    if recordNoise.capitalize() == "Y":
        noiseProfile = recorder.record(4, playBack=True)
        recorder.save(noiseProfile)

    recorder.continuousRecord(showImages=True)
