import math

from torch import nn


class EarlyStopper:
    '''
    A custom early stopper to use while using the TrainModel script.
    Every time the validation loss reaches a new minimum, this class will save the results and "stateDict" of that epoch.
    After "amount" of times when the difference between "lowesLoss" and the validation loss
    of that epoch is greater than "delta", the EarlyStopper will return the best values.


    Args:
        network (nn.Module): The neural network module being monitored.
        amount (int): The amount of epochs to wait before stopping early.
        delta (float): The change in loss threshold to trigger early stopping.
        counter (int): Counter for the number of epochs waited.
        epochsWaited (int): Number of epochs waited.
        stateDict (dict): The state dictionary of the network when the lowest loss is achieved.
        lowestLoss (float): The lowest loss achieved during training.
        confMatPred (any): Confusion matrix predictions.
        confMatTarget (any): Confusion matrix target.
    Methods:
        init(self, network: nn.Module, amount: int = 5, delta: float = 1e-3): Initializes the EarlyStopper object with given parameters.
        reset(self): Resets the counter, lowest loss, and state dictionary.
        evaluate(self, currentLoss, confMatTarget, confMatPred) -> bool: Evaluates the current loss and updates the state if needed. Returns True if early stopping condition is met.

    Example usage:
        network = NeuralNetwork()
        early_stopper = EarlyStopper(network, amount=3, delta=0.001)
        for epoch in range(10):
            loss = train_step()
            if early_stopper.evaluate(loss, confMatTarget, confMatPred):
                print("Early stopping condition met!")
                break
            else:
                print(f"Epoch {epoch}: Loss - {loss}")
    '''

    def __init__(self, network: nn.Module, amount: int = 5, delta: float = 1e-3):
        self.network = network
        self.amount = amount
        self.counter = amount
        self.epochsWaited = 0
        self.delta = delta
        self.stateDict = None
        self.lowestLoss = math.inf
        self.confMatPred = None
        self.confMatTarget = None

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def confMatPred(self):
        return self._confMatPred

    @confMatPred.setter
    def confMatPred(self, value):
        self._confMatPred = value

    @property
    def confMatTarget(self):
        return self._confMatTarget

    @confMatTarget.setter
    def confMatTarget(self, value):
        self._confMatTarget = value

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        self._amount = value

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        self._delta = value

    def reset(self):
        self.counter = self.amount
        self.lowestLoss = math.inf
        self.stateDict = None

    def evaluate(self, currentLoss, confMatTarget, confMatPred) -> bool:
        self.epochsWaited += 1
        if currentLoss < self.lowestLoss:
            self.lowestLoss = currentLoss
            self.counter = self.amount
            self.stateDict = self.network.state_dict()
            self.epochsWaited = 0
            self.confMatTarget = confMatTarget
            self.confMatPred = confMatPred
        elif currentLoss > self.lowestLoss + self.delta:
            self.counter = self.counter - 1

        if self.counter == 0:
            return True
        else:
            return False
