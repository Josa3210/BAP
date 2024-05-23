import random
from abc import ABC, abstractmethod

import numpy as np

from CustomLogger import CustomLogger


class Transform(ABC):
    """
        Abstract base class for signal transformation.

        Args:
            amount (int): The number of times the transformation will be applied.

        Attributes:
            amount (int): The number of times the transformation will be applied.

        Methods:
            - transform(signal: numpy.ndarray, fs: int) -> Iterator[numpy.ndarray]:
                Abstract method to transform the input signal.
                Yields transformed signals (e.g., with added offsets).

        Example usage:
            # Create a custom transformation class.
            class MyCustomTransform(Transform):
                def transform(self, signal, fs):
                    # Implement custom transformation logic here
                    pass

            # Instantiate the custom transformation
            my_transform = MyCustomTransform(amount=3)
            for transformed_signal in my_transform.transform(input_signal, fs=44100):
                # Process each transformed signal
                pass
        """

    def __init__(self, amount):
        self.amount = amount
        pass

    @abstractmethod
    def transform(self, signal, fs):
        pass

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        self._amount = value


class AddOffset(Transform):
    """
        A signal transformation that adds random offsets to the input signal.

        Args:
            amount (int): The number of times the offset will be added.
            maxTimeOffset (float, optional): Maximum time offset in seconds (default is 1).

        Attributes:
            maxTimeOffset (float): Maximum time offset in seconds.

        Methods:
            - transform(signal: numpy.ndarray, fs: int) -> Iterator[numpy.ndarray]:
                Adds random offsets to the input signal.
                Yields offsetted signals.

        Example usage:
            # Create an AddOffset transformation
            offset_transform = AddOffset(amount=5, maxTimeOffset=0.5)

            # Process an audio signal
            audio_signal = np.random.randn(44100)  # Example audio signal
            for offsetted_signal in offset_transform.transform(audio_signal, fs=44100):
                # Process each offsetted signal
                pass
        """

    def __init__(self, amount, maxTimeOffset: float = 1):
        super().__init__(amount)
        self.logger = CustomLogger.getLogger(__name__)
        self.maxTimeOffset = maxTimeOffset
        self. i = 0

    def transform(self, signal, fs):
        # Define the maximum possible offset
        maxOffset = round(self.maxTimeOffset * fs)

        # Convert signal to a np.array
        self.i = self.i + 1
        signal = np.array(signal, dtype='f')
        signalLength = len(signal)

        # Create a long sample by adding noise in front and in back.
        noise = np.random.normal(0, np.std(signal) * 0.2, maxOffset)
        longSignal = np.append(np.append(noise, signal), noise)

        for i in range(self.amount):
            # choose random amount of points
            kRandom = round(random.random() * 2 * maxOffset)
            # Take random piece of the sample
            offsetSignal = longSignal[kRandom:kRandom + signalLength]
            yield offsetSignal
