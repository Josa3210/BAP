import logging
import math
import os
import time
from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn, device, Tensor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, random_split
import utils
from CustomLogger import CustomLogger


class InterfaceNN(nn.Module):
    @abstractmethod
    def __init__(self, name: str, initMethod: nn.init = nn.init.xavier_normal_):
        super().__init__()
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.device = self.getDevice()

        self.validationResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        self.testResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        self.trainingLossesPerFold = []

        self.trainingData = None
        self.testData = None

        self.initMethod = initMethod
        self.batchSize = 64
        self.learningRate = 1e-5
        self.dropoutRate = 0.5
        self.folds = 5
        self.epochs = 35

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

    def initWeights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            self.initMethod(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def initWeightsZero(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)

    @staticmethod
    def calcSizeConv(inputSize, filterSize: int, stride: int = 1, padding: int = 0):
        return math.floor((inputSize + 2 * padding - filterSize) / stride) + 1

    @staticmethod
    def calcSizePool(inputSize, filterSize: int, stride: int, dilation: int = 1, padding: int = 0):
        return math.floor(((inputSize + 2 * padding) - (dilation * (filterSize - 1)) - 1) / stride) + 1

    def clearResults(self, clearTestResults: bool = False):
        if clearTestResults:
            self.testResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        else:
            self.validationResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}

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
            self.epochs = round(epochs)
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
        self.trainingLossesPerFold = []

        if self.folds > 1:
            kFold = KFold(n_splits=self.folds, shuffle=True)

        for fold in range(self.folds):
            # Reset weights
            self.apply(self.initWeights)

            if self.folds > 1:
                train_ids, validation_ids = next(kFold.split(self.trainingData))
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
            else:
                trainSize = round(len(self.trainingData) * 0.8)
                validationSize = len(self.trainingData) - trainSize
                trainDataset, validationDataset = random_split(self.trainingData, (trainSize, validationSize))
                # Define data loaders for training and testing data in this fold
                trainLoader = DataLoader(
                    self.trainingData,
                    batch_size=self.batchSize)
                validationLoader = DataLoader(
                    self.trainingData,
                    batch_size=self.batchSize)

            # Print
            self.logger.debug(f'\nFOLD {fold}')
            self.logger.debug('=' * 30)

            # Get the network to the right device
            self.to(self.device)
            optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)

            trainingLossPerEpoch = []

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

                    if (i + 1) % 2 == 0:
                        self.logger.debug(f"{i:4d} / {len(trainLoader)} batches: average loss = {currentLoss / (i + 1)}")

                trainingLossPerEpoch.append(currentLoss / len(trainLoader))

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

                averageValidationLoss = currentLoss / len(validationLoader)
                # Calculate confusion matrix and metrics
                self.validationResults["Loss"].append(averageValidationLoss)
                self.validationResults["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
                self.validationResults["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
                self.validationResults["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)

            self.trainingLossesPerFold.append(trainingLossPerEpoch)
        return - sum(self.validationResults["Loss"]) / len(self.validationResults["Loss"])

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
            self.testResults["Loss"].append(currentLoss / len(testLoader))
            self.testResults["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
            self.testResults["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
            self.testResults["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
        return sum(self.testResults["Loss"]) / len(self.testResults["Loss"])

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
        time.sleep(1)

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
            pbounds=parameterBounds
        )
        optimizer.maximize(
            init_points=init_points,  # init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
            n_iter=n_iter  # n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
        )

        time.sleep(1)

        self.logger.info("Finished optimizing")
        optimizedKeys = optimizer.max["params"].keys()
        for key in optimizedKeys:
            val = optimizer.max["params"].get(key)
            if "fs" in key:
                val = math.ceil(val)
            self.logger.info(f"Best {key} is: {val}")
            results.update({key: val})

        return results

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

    def loadModel(self, path: Path):
        self.logger.info(f"Trying to download model from {path}")

        if not path.exists():
            self.logger.warning("Path does not exist")
            return

        self.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info("Successfully downloaded model")
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def printWeights(self):
        self.logger.info("PRINTING WEIGHTS:")
        self.logger.info("=" * 30)
        self.logger.info('\n')
        for layer in self.children():
            for subLayers in layer.children():
                if isinstance(subLayers, nn.Conv1d):
                    self.logger.info(f"Weights of layer {subLayers}")
                    self.logger.info(subLayers.weight)

    def printLoss(self):
        xEpochs = list(range(self.epochs))
        for i in range(self.folds):
            plt.plot(xEpochs, self.trainingLossesPerFold[i], label=f"Fold {i + 1}")
        plt.title("Average loss per epoch", fontsize=30)
        plt.xlabel("Epochs")
        plt.ylabel("Average loss")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def printResults(self, testResult: bool = False, fullReport: bool = False):
        if testResult:
            results = self.testResults
            typeResults = "Test"
        else:
            typeResults = "Training"
            results = self.validationResults

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
                self.logger.info(f"Accuracy:\t{results[keys[1]][i]:.2f}%")
                self.logger.info(f"Precision:\t{results[keys[2]][i]:.2f}%")
                self.logger.info(f"Recall:\t\t{results[keys[3]][i]:.2f}%")
                self.logger.info("-" * 30)

        self.logger.info("Average:")
        self.logger.info("-" * 30)

        avgAccuracy = sum(results[keys[1]]) / folds
        avgPrecision = sum(results[keys[2]]) / folds
        avgRecall = sum(results[keys[3]]) / folds

        self.logger.info(f"Accuracy:\t{avgAccuracy:.2f}%")
        self.logger.info(f"Precision:\t{avgPrecision:.2f}%")
        self.logger.info(f"Recall:\t\t{avgRecall:.2f}%")
        self.logger.info("=" * 30)
