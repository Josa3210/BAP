import os.path
from pathlib import Path

import numpy
import sounddevice as sd
from scipy.io.wavfile import write


class Audiorecorder:
    def __init__(self, baseLink: str, sampleRate: int, channels: int):
        """
       Initializes the Audiorecorder class. Sets the baseLink (the path where the audio files will be saved),
       the sample rate (the number of samples per second), and the number of channels.
       """

        self.inputDevice = sd.default.device[0]
        self.sampleRate = sampleRate  # Sample rate (samples per second)
        self.channels = channels  # Channels
        self.baseLink = Path(baseLink)

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
    def baseLink(self):
        return self._baseLink

    @baseLink.setter
    def baseLink(self, link: Path):
        if not link.is_dir():
            os.makedirs(link)

        self._baseLink = link

    def record(self, duration: float, playBack: bool = False):
        """
        Records audio for a specified duration. If `playBack` is set to `True`, it plays back the recorded audio after recording.
        """
        # Record audio
        print("Recording started")
        recording = sd.rec(int(duration * self.sampleRate))
        sd.wait()  # Wait until recording is finished
        print("Recording stopped")

        if playBack:
            print("Playing back...")
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
        path = self.baseLink.joinpath(Path(fileName))
        parentPath = path.parent
        if not parentPath.is_dir():
            os.makedirs(parentPath)

        write(path, self.sampleRate, recording)
        print(f"Recording saved under '{path}'")

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

    def continuousRecord(self):
        """
        This method continuously records audio until the user indicates they want to stop.
        After each recording, it asks the user if the recording should be saved.
        """
        duration = float(input("Give duration: "))

        userInput = input("Ready to record? (Y or N): ")
        while userInput.capitalize() == "Y":
            recording = self.record(duration=duration, playBack=True)
            saveInput = input("Save? (Y or N): ")
            if saveInput.capitalize() == "Y":
                self.save(recording)
            userInput = input("Ready to record? (Y or N): ")


if __name__ == '__main__':
    directory = input("In which directory will you save the files?: ")
    sampleRate = input("What sampleRate do you use?: ")
    channels = input("How many channels do you use?: ")
    recorder = Audiorecorder(directory, int(sampleRate), int(channels))

    print("Start recording...")
    recorder.continuousRecord()
