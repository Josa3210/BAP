import logging
import math
import os
from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn, device, Tensor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from datetime import date
import utils
from customLogger import CustomLogger


class InterfaceNN(nn.Module):
    @abstractmethod
    def __init__(self, name: str):
        super().__init__()
        self.logger = CustomLogger.getLogger(__name__)

        self.device = self.getDevice()

        self.trainingResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        self.testResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        self.lossesPerFold = []

        self.trainingData = None
        self.testData = None

        self.batchSize = 64
        self.learningRate = 1e-5
        self.dropoutRate = 0.5
        self.folds = 5
        self.epochs = 5

        self.savePath = utils.getDataRoot().joinpath("model")
        self._name = name

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @property
    def savePath(self):
        return self._savePath

    @savePath.setter
    def savePath(self, value: Path):
        if not value.exists():
            os.makedirs(value)
        self._savePath = value

    @property
    def trainingData(self):
        return self._trainingData

    @trainingData.setter
    def trainingData(self, data):
        self._trainingData = data

    @property
    def testData(self):
        return self._testData

    @testData.setter
    def testData(self, data):
        self._testData = data

    @property
    def batchSize(self):
        return self._batchSize

    @batchSize.setter
    def batchSize(self, batchSize: int):
        self._batchSize = batchSize

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, lr: float):
        self._learningRate = lr

    @property
    def dropoutRate(self):
        return self._dropoutRate

    @dropoutRate.setter
    def dropoutRate(self, dr: float):
        self._dropoutRate = dr

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs: int):
        self._epochs = epochs

    @property
    def folds(self):
        return self._folds

    @folds.setter
    def folds(self, folds: int):
        self._folds = folds

    @staticmethod
    def getDevice():
        return device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def clearResults(self, clearTestResults: bool = False):
        if clearTestResults:
            self.testResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        else:
            self.trainingResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}

    def printLoss(self):
        xEpochs = list(range(self.epochs))
        for i in range(self.folds):
            plt.plot(xEpochs, self.lossesPerFold[i], label=f"Fold {i + 1}")
        plt.title("Average loss per epoch", fontsize=30)
        plt.xlabel("Epochs")
        plt.xticks([i + 1 for i in xEpochs])
        plt.ylabel("Average loss")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def testOnData(self,
                   testData: Dataset,
                   batchSize: int = None
                   ):
        # Initialize parameters
        if testData is not None:
            self.testData = testData
        if batchSize is not None:
            self.batchSize = batchSize

        if self.testData is None:
            self.logger.info("Define trainingdata using network.setTrainingData()")
            return None

        self.clearResults(clearTestResults=True)

        testLoader = DataLoader(dataset=testData, shuffle=True)
        lossFunction = nn.CrossEntropyLoss()

        currentLoss = 0
        confMatPred, confMatTarget = [], []

        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, batch in enumerate(testLoader):
                # Get inputs
                inputs, targets = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Generate outputs
                outputs = self(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)

                loss = lossFunction(outputs, targets)
                currentLoss += loss.item()

                confMatPred.extend(predicted.data.cpu().numpy())
                confMatTarget.extend(targets.data.cpu().numpy())

            # Calculate confusion matrix and metrics
            self.testResults["Loss"].append(currentLoss)
            self.testResults["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
            self.testResults["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
            self.testResults["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)

    @abstractmethod
    def trainOnData(self,
                    trainingData: Dataset = None,
                    folds: int = None,
                    epochs: int = None,
                    batchSize: int = None,
                    lr: float = None,
                    dr: float = None,
                    verbose: bool = False):

        # Initialize parameters
        if trainingData is not None:
            self.trainingData = trainingData
        if folds is not None:
            self.folds = folds
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
        if lr is not None:
            self.learningRate = lr
        if dr is not None:
            self.dropoutRate = dr

        if not verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Start training network")

        self.clearResults(clearTestResults=False)

        if self.trainingData is None:
            self.logger.error("Define trainingdata using network.setTrainingData()")
            return None

        lossFunction = nn.CrossEntropyLoss()
        self.lossesPerFold = []
        lossPerEpoch = []

        kFold = KFold(n_splits=self.folds, shuffle=True)

        for fold, (train_ids, validation_ids) in enumerate(kFold.split(self.trainingData)):
            # Print

            self.logger.debug(f'\nFOLD {fold}')
            self.logger.debug('=' * 30)

            # Sample elements randomly from a given list of ids, no replacement.
            trainSubSampler = SubsetRandomSampler(train_ids)
            validationSubSampler = SubsetRandomSampler(validation_ids)

            # Define data loaders for training and testing data in this fold
            trainLoader = DataLoader(
                self.trainingData,
                batch_size=self.batchSize, sampler=trainSubSampler)
            validationLoader = DataLoader(
                self.trainingData,
                batch_size=self.batchSize, sampler=validationSubSampler)

            # Get the network to the right device
            self.to(self.device)
            optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)

            lossPerEpoch = []

            # Start training epochs
            for epoch in range(self.epochs):
                # Print epoch
                self.logger.debug(f'\nStarting epoch {epoch + 1}')
                self.logger.debug("-" * 30)

                # Set current loss value
                currentLoss = 0.
                # Iterate over the DataLoader for training data
                for i, batch in enumerate(trainLoader):
                    # Get inputs
                    inputs, targets = batch

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Perform forward pass
                    outputs = self(inputs)

                    # Compute loss
                    loss = lossFunction(outputs, targets)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # Print statistics
                    currentLoss += loss.item()

                    if i % 10 == 1:
                        self.logger.debug(f"{i:4d} / {len(trainLoader)} batches: average loss = {currentLoss / (i + 1)}")

                lossPerEpoch.append(currentLoss / len(trainLoader))

            # Evaluation for this fold
            self.logger.debug('-' * 30)

            # Lists for creating confusion matrix and loss
            currentLoss = 0.
            confMatPred, confMatTarget = [], []

            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, batch in enumerate(validationLoader):
                    # Get inputs
                    inputs, targets = batch

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Generate outputs
                    outputs = self(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)

                    loss = lossFunction(outputs, targets)
                    currentLoss += loss.item()

                    confMatPred.extend(predicted.data.cpu().numpy())
                    confMatTarget.extend(targets.data.cpu().numpy())

                # Calculate confusion matrix and metrics
                self.trainingResults["Loss"].append(currentLoss / len(validationLoader))
                self.trainingResults["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
                self.trainingResults["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
                self.trainingResults["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)

            self.lossesPerFold.append(lossPerEpoch)
        return - sum(self.trainingResults["Loss"]) / len(self.trainingResults["Loss"])

    def saveModel(self, path: Path = None, name: str = None, idNr: int = None):
        fileName = self._name
        if name is not None:
            fileName += "-" + name
        if idNr is not None:
            fileName += "-" + str(idNr)
        if path is None:
            path = self.savePath

        if not path.exists():
            os.makedirs(path)

        torch.save(self.state_dict(), path.joinpath(fileName + ".pth"))

    def printResults(self, testResult: bool = False, fullReport: bool = False):
        if testResult:
            results = self.testResults
            typeResults = "Test"
        else:
            typeResults = "Training"
            results = self.trainingResults

        keys: list[str] = list(results.keys())
        folds = len(results[keys[0]])

        if folds == 0:
            self.logger.warning("No folds found. Exiting")
            return

        self.logger.info(f"\nResults of {typeResults}:")
        self.logger.info("=" * 30)

        if fullReport:
            for i in range(folds):
                self.logger.info(f"For fold {i:d}:")
                self.logger.info("-" * 30)
                self.logger.info(f"Accuracy: {results[keys[1]][i]:.2f}%")
                self.logger.info(f"Precision: {results[keys[2]][i]:.2f}%")
                self.logger.info(f"Recall: {results[keys[3]][i]:.2f}%")
                self.logger.info("-" * 30)

        self.logger.info("Average:")
        self.logger.info("-" * 30)

        avgAccuracy = sum(results[keys[1]]) / folds
        avgPrecision = sum(results[keys[2]]) / folds
        avgRecall = sum(results[keys[3]]) / folds

        self.logger.info(f"Accuracy: {avgAccuracy:.2f}%")
        self.logger.info(f"Precision: {avgPrecision:.2f}%")
        self.logger.info(f"Recall: {avgRecall:.2f}%")
        self.logger.info("=" * 30)

    def optimizeParams(self,
                       bounds: dict[str, tuple[float, float]],
                       trainingData: Dataset = None,
                       init_points: int = 5,
                       n_iter: int = 25,
                       folds: int = None,
                       epochs: int = None,
                       batchSize: int = None):

        results = dict.fromkeys(bounds.keys(), 0)
        params = ""
        for key in results.keys():
            params += " " + key + ","
        self.logger.info(f"Starting optimization of parameters: {params[0:-1]}")

        # Initialize parameters
        if trainingData is not None:
            self.trainingData = trainingData
        if folds is not None:
            self.folds = folds
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize

        # Give the parameter space from which the optimizer can choose
        parameterBounds = bounds

        # Create the optimizer object
        optimizer = BayesianOptimization(
            f=self.trainOnData,
            pbounds=parameterBounds,
            verbose=0
        )

        optimizer.maximize(
            init_points=init_points,  # init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
            n_iter=n_iter  # n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
        )

        self.logger.info("Finished optimizing")
        optimizedKeys = optimizer.max["params"].keys()
        for key in optimizedKeys:
            val = optimizer.max["params"].get(key)
            if "fs" in key:
                val = math.ceil(val)
            self.logger.info(f"Best {key} is: {val}")
            results.update({key: val})

        return results
