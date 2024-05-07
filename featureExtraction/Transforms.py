import os
from abc import ABC, abstractmethod

import matlab.engine
import numpy as np

import utils
from CustomLogger import CustomLogger


class Transform(ABC):
    def __init__(self, amount, engine: matlab.engine.MatlabEngine = None):
        self.amount = amount
        if engine is None:
            self.engine = matlab.engine.start_matlab()
        else:
            self.engine = engine
        self.engine.cd(str(utils.getFunctionPath()))
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

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, value):
        self._engine = value


class AddOffset(Transform):
    def __init__(self, amount, transformPath: str = utils.getFunctionPath().joinpath("addOffset.m"), maxTimeOffset: float = 1, engine: matlab.engine.matlabengine = None):
        super().__init__(amount, engine)
        self.logger = CustomLogger.getLogger(__name__)
        self.maxTimeOffset = maxTimeOffset

        if not os.path.isfile(transformPath):
            self.logger.error(f"{transformPath} has not been found! Please add this file or specify location in the constructor (filterPath=)")
            return

    def transform(self, signal, fs):
        signal = np.array(signal, dtype='f')
        for i in range(self.amount):
            yield self.engine.addOffset(signal, fs, self.maxTimeOffset)
