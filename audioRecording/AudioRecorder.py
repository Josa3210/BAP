import numpy
import sounddevice as sd
from scipy.io.wavfile import write


class Audiorecorder:
    def __init__(self, baseLink: str, sampleRate: int, channels: int):
        self.inputDevice = None
        self.sampleRate = sampleRate  # Sample rate (samples per second)
        self.channels = channels  # Channels
        self.baseLink = baseLink

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
        return self._link

    @baseLink.setter
    def baseLink(self, link: str):
        self._link = link

    def record(self, duration: float, fileName: str = None):
        # Record audio
        print("Recording started")
        recording = sd.rec(int(duration * self.sampleRate))
        sd.wait()  # Wait until recording is finished
        print("Recording stopped")

        # Check for filename or ask for one
        if fileName is None:
            fileName = input("Give name to this file:\n")
        fileName += ".wav"

        # Save in folder under given filename
        write(self.baseLink + fileName, self.sampleRate, recording)
        print(f"Recoding saved under '{self.baseLink + fileName}'")

    def setInputDevice(self):
        print(sd.query_devices(kind="input"))
        index = input("Choose your device: (Copy the name) \n")
        self.inputDevice = sd.query_devices(device=index)
        sd.default.device[0] = index
