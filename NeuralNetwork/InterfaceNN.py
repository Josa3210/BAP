import logging
import math
import os
import time
from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from bayes_opt import BayesianOptimization
from scipy.io import wavfile
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch import nn, device, Tensor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, random_split

from Tools import PathFinder
from Tools.CustomLogger import CustomLogger
from NeuralNetwork.EarlyStopper import EarlyStopper


class InterfaceNN(nn.Module):
    """
    Abstract base class for neural network interfaces. Adds possibility to save and load models and get the current device

    Args:
        name (str): The name of the interface.

    Attributes:
        device: The device (CPU or CUDA) on which the model runs.
        savePath: The path where model checkpoints will be saved.
        _name (str): The name of the interface.

    Methods:
        forward(x: Tensor) -> None:
            Abstract method for forward pass computation.
        saveModel(path: Path = None, name: str = None, idNr: int = None) -> None:
            Saves the model's state dictionary to a file.
        loadModel(path: Path) -> bool:
            Loads a model state dictionary from a file.
        printWeights() -> None:
            Prints weights of convolutional layers (for debugging).

    Properties:
        savePath: The save path for model checkpoints.
    """

    def __init__(self, name: str):
        super().__init__()
        self.device = self.getDevice()
        self.savePath = PathFinder.getDataRoot().joinpath("model")
        self.name = name
        self.fExtractor = None
        self.maxVal = None

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def fExtractor(self):
        return self._fExtractor

    @fExtractor.setter
    def fExtractor(self, value):
        self._fExtractor = value

    @property
    def maxVal(self):
        return self._maxVal

    @maxVal.setter
    def maxVal(self, value):
        self._maxVal = value

    @property
    def savePath(self):
        return self._savePath

    @savePath.setter
    def savePath(self, value: Path):
        if not value.exists():
            os.makedirs(value)
        self._savePath = value

    @staticmethod
    def getDevice():
        return device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def saveModel(self, maxVal: int, path: Path = None, name: str = None, idNr: int = None):
        # Add different tags to the filename
        fileName = self._name
        if name is not None:
            fileName += "-" + name
        if idNr is not None:
            fileName += "-" + str(idNr)
        if path is None:
            path = self.savePath

        if not path.exists():
            os.makedirs(path)

        # Next to the state_dict of the model, also the maximum value used for normalization of the trainingDataset must be saved
        # to normalize new samples
        saveDict = {
            "stateDict": self.state_dict(),
            "maxVal": maxVal
        }

        # Save the file
        torch.save(saveDict, path.joinpath(fileName + ".pth"))

    def loadModel(self, path: Path) -> bool:
        """
        Load the saved dictionary from path.
        Then set the networks state_dict to the one loaded in from the path.
        Also, set the maxVal to the value from the dataset used wile training
        :param path: path of where the saved file is
        :type path: Path
        :return: if the loading of the model is done successfully
        :rtype: bool
        """
        if not path.exists():
            return False

        loadDict = torch.load(path, map_location=self.device)
        stateDict = loadDict["stateDict"]
        self.maxVal = loadDict["maxVal"]
        self.load_state_dict(stateDict)
        return True

    def transformSample(self, sample: Path | tuple[list[float], int]) -> list[float] | None:
        """
        Convert a raw recording sample into a sample that can be read by the network.
        :param sample: The path to the sample, or the sample itself with also the sampling rate in a list
        """
        if isinstance(sample, Path):
            if sample.exists():
                signal, fs = wavfile.read(sample)
            else:
                self.logger.error(f"Path to sample does not exist: {sample}")
        elif isinstance(sample, tuple):
            signal = sample[0]
            fs = sample[1]
        else:
            self.logger.error(f"Sample is not right type. Expected Path or tuple[list[float], given {type(sample)}")
            return None

        # Filter the sample
        filteredSignal = self.featureExtractor.filter(signal, fs)

        # Transform the sample
        transformedSignal = self.featureExtractor.transform(filteredSignal, fs)

        # Normalize the sample
        transformedSignal /= self.maxVal

        return transformedSignal

    def predictSample(self, sample: Path | tuple[list[float], int]):
        feature = self.transformSample(sample)
        featureT = torch.Tensor(feature).unsqueeze(0)   # Need to add an extra dimension as if it was a batch with size 1
        result = self.forward(featureT)
        return result


class TrainableNN(InterfaceNN):
    """
    Abstract base class for trainable neural networks. This class implements methods trainOnData and testOnData plus other useful static functions.
    These extra functions all help with training the network.

    Args:
        name (str): The name of the interface.
        initMethod (nn.init, optional): The weight initialization method (default: nn.init.xavier_normal_).

    Attributes:
        device: The device (CPU or CUDA) on which the model runs.
        savePath: The path where model checkpoints will be saved.
        _name (str): The name of the interface.
        lossFunction: The loss function used for training.
        dropoutRate: The dropout rate for regularization.
        validationConfMat: The confusion matrix for validation.
        trainingLossesPerFold: List of training losses per fold (for cross-validation).
        validationLossesPerFold: List of validation losses per fold (for cross-validation).
        trainingData: Training data (if applicable).
        testData: Test data (if applicable).
        initMethod: The weight initialization method.

    Methods:
        forward(x: Tensor) -> None:
            Abstract method for forward pass computation.
        saveModel(path: Path = None, name: str = None, idNr: int = None) -> None:
            Saves the model's state dictionary to a file.
        loadModel(path: Path) -> bool:
            Loads a model state dictionary from a file.
        printWeights() -> None:
            Prints weights of convolutional layers (for debugging).
        initWeights(m: nn.Module) -> None:
            Initializes weights using the specified method.
        initWeightsZero(m: nn.Module) -> None:
            Initializes weights to zero.
        calcSizeConv(inputSize: int, filterSize: int, stride: int = 1, padding: int = 0) -> int:
            Calculates output size after a convolutional layer.
        calcSizePool(inputSize: int, filterSize: int, stride: int, dilation: int = 1, padding: int = 0) -> int:
            Calculates output size after a pooling layer.
        trainOnData(trainingData: Dataset = None, folds: int = None, epochs: int = None, batchSize: int = None,
                    lr: float = None, dr: float = None, verbose: bool = False, saveModel: bool = False):
            Trains the model and returns results of the best training iteration.
        testOnData(testData: Dataset, batchSize: int = 32):
            Test the given data on the network.
        optimizeParams(bounds: dict[str, tuple[float, float]], trainingData: Dataset = None,
                    init_points: int = 5, n_iter: int = 25):


    """

    @abstractmethod
    def __init__(self, name: str, initMethod: nn.init = nn.init.xavier_normal_):
        super().__init__(name)
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.lossFunction = nn.CrossEntropyLoss()
        self.dropoutRate = 0

        self.validationConfMat = None
        self.trainingLossesPerFold = []
        self.validationLossesPerFold = []

        self.trainingData = None
        self.testData = None

        self.initMethod = initMethod

    @property
    def dropoutRate(self):
        return self._dropoutRate

    @dropoutRate.setter
    def dropoutRate(self, value):
        self._dropoutRate = value

    @property
    def trainingData(self):
        return self._trainingData

    @trainingData.setter
    def trainingData(self, data):
        self._trainingData = data

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

    def trainOnData(self,
                    trainingData: Dataset = None,
                    folds: int = None,
                    epochs: int = None,
                    batchSize: int = None,
                    lr: float = None,
                    dr: float = None,
                    verbose: bool = False,
                    saveModel: bool = False,
                    earlyStopper: EarlyStopper = None):

        # Initialize parameters
        if trainingData is not None:
            self.trainingData = trainingData
        if dr is not None:
            self.dropoutRate = dr

        # Check if there is data
        if self.trainingData is None:
            self.logger.error("Define trainingdata using network.setTrainingData(trainingDataLoader)")
            return None

        # Set verbose level
        if not verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)

        # Start training
        self.logger.debug("Start training network")

        # Check for folds, if folds = 1 -> choose 80% of data for training and 20% for validation
        kFold = None
        if folds > 1:
            kFold = KFold(n_splits=folds, shuffle=True)

        # Initialize values for capturing performance
        bestResult = math.inf
        self.trainingLossesPerFold = []
        self.validationLossesPerFold = []
        validationResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        bestConfMat = None

        for fold in range(folds):
            # Reset weights
            self.apply(self.initWeights)

            # If there are folds, get the ids of the training and validation samples
            # Put them in a dataloader
            if folds > 1:
                train_ids, validation_ids = next(kFold.split(self.trainingData))

                # Sample elements randomly from a given list of ids, no replacement.
                trainSubSampler = SubsetRandomSampler(train_ids)
                validationSubSampler = SubsetRandomSampler(validation_ids)

                # Define data loaders for training and testing data in this fold
                trainLoader = DataLoader(
                    self.trainingData,
                    batch_size=batchSize,
                    sampler=trainSubSampler)
                validationLoader = DataLoader(
                    self.trainingData,
                    batch_size=batchSize,
                    sampler=validationSubSampler)

            else:
                # Choose random 80% of data and put as trainingdata, other part is validation
                trainSize = round(len(self.trainingData) * 0.8)
                validationSize = len(self.trainingData) - trainSize
                trainDataset, validationDataset = random_split(self.trainingData, (trainSize, validationSize))

                # Define data loaders for training and testing data in this fold
                trainLoader = DataLoader(
                    trainDataset,
                    batch_size=batchSize,
                    shuffle=True)
                validationLoader = DataLoader(
                    validationDataset,
                    batch_size=batchSize,
                    shuffle=True)

            # Print
            self.logger.debug(f'\nFOLD {fold}')
            self.logger.debug('=' * 30)

            # Get the network to the right device
            self.to(self.device)

            # Create optimizer and add learning rate
            optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            if earlyStopper is not None:
                earlyStopper.reset()
            mustStop: bool = False

            # Initialize values for capturing performance
            trainingLossPerEpoch = []
            validationLossPerEpoch = []
            avgValidationLoss = 0
            confMatPred, confMatTarget = [], []

            # Start training epochs
            for epoch in range(epochs):
                # Print epoch
                self.logger.debug(f'\nStarting epoch {epoch + 1}')
                self.logger.debug("-" * 30)

                # Lists for creating confusion matrix and loss
                confMatPred, confMatTarget = [], []

                # Set current loss value
                cumulativeTrainingLoss = 0.
                cumulativeValidationLoss = 0.

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
                    loss = self.lossFunction(outputs, targets)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # Print statistics
                    cumulativeTrainingLoss += loss.item()

                # Calculate average training loss of this epoch
                avgTrainingLoss = cumulativeTrainingLoss / len(trainLoader)
                self.logger.debug(f"Average training loss: {avgTrainingLoss}")

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

                        # Calculate loss and add to total loss
                        loss = self.lossFunction(outputs, targets)
                        cumulativeValidationLoss += loss.item()

                        confMatPred.extend(predicted.data.cpu().numpy())
                        confMatTarget.extend(targets.data.cpu().numpy())

                avgValidationLoss = cumulativeValidationLoss / len(validationLoader)
                self.logger.debug(f"Average validation loss: {avgValidationLoss}")

                trainingLossPerEpoch.append(avgTrainingLoss)
                validationLossPerEpoch.append(avgValidationLoss)

                if earlyStopper is not None:
                    mustStop = earlyStopper.evaluate(currentLoss=avgValidationLoss, confMatTarget=confMatTarget, confMatPred=confMatPred)

                if mustStop:
                    avgValidationLoss = earlyStopper.lowestLoss
                    self.load_state_dict(earlyStopper.stateDict)
                    confMatTarget = earlyStopper.confMatTarget
                    confMatPred = earlyStopper.confMatPred
                    self.logger.info(f"Stopped early at epoch {epoch} with loss of {avgValidationLoss}")
                    break

            # Evaluation for this fold
            self.logger.debug('-' * 30)

            # Save best model
            if avgValidationLoss < bestResult:
                bestConfMat: confusion_matrix = confusion_matrix(y_true=confMatTarget, y_pred=confMatPred)
                bestResult = avgValidationLoss
                if saveModel: self.saveModel(self.maxVal, self.savePath, "BestResult")

            # Add current results to dictionary
            validationResults["Loss"].append(avgValidationLoss)
            validationResults["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
            validationResults["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
            validationResults["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
            self.trainingLossesPerFold.append(trainingLossPerEpoch)
            self.validationLossesPerFold.append(validationLossPerEpoch)

        # Set itself to the best model from the fold
        if saveModel:
            self.loadModel(self.savePath.joinpath(self._name + "-" + "BestResult.pth"))
        return validationResults, bestConfMat

    def getName(self):
        return self._name

    def testOnData(self,
                   testData: Dataset,
                   batchSize: int = 32):

        # Initialize parameters
        if testData is not None:
            self.testData = testData

        # Check if there is data
        if self.testData is None:
            self.logger.info("Define trainingdata using network.setTestData(testDataLoader)")
            return None

        self.to(self.device)

        # Create dataloader
        testLoader = DataLoader(dataset=testData, batch_size=batchSize, shuffle=True)

        # Initialize variables
        currentLoss = 0
        confMatPred, confMatTarget = [], []
        testResults = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}

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

                loss = self.lossFunction(outputs, targets)
                currentLoss += loss.item()

                confMatPred.extend(predicted.data.cpu().numpy())
                confMatTarget.extend(targets.data.cpu().numpy())

            avgTestLoss = currentLoss / len(testLoader)

            # Calculate confusion matrix and metrics
            testResults["Loss"].append(avgTestLoss)
            testResults["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
            testResults["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
            testResults["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
            confMat = confusion_matrix(y_true=confMatTarget, y_pred=confMatPred)
        return testResults, confMat

    def optimizeParams(self,
                       bounds: dict[str, tuple[float, float]],
                       trainingData: Dataset = None,
                       init_points: int = 5,
                       n_iter: int = 25):

        results = dict.fromkeys(bounds.keys(), 0)
        params = ""
        for key in results.keys():
            params += " " + key + ","
        self.logger.info(f"Starting optimization of parameters: {params[0:-1]}")
        time.sleep(1)

        # Initialize parameters
        if trainingData is not None:
            self.trainingData = trainingData

        # Give the parameter space from which the optimizer can choose
        parameterBounds = bounds

        # Create the optimizer object
        optimizer = BayesianOptimization(
            f=self.funcToOptimize,
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

    def funcToOptimize(self, lr, dr: int = None, epochs: int = 200):
        if dr is None:
            dr = self.dropoutRate

        result, confMat = self.trainOnData(folds=1, epochs=epochs, lr=lr, dr=dr, batchSize=32, saveModel=False)
        return - (sum(result["Loss"]) / len(result["Loss"]))

    def printLoss(self):
        folds = len(self.trainingLossesPerFold)
        if folds > 1:
            fig, axs = plt.subplots(1, folds)
            for i in range(folds):
                axs[i].plot(self.trainingLossesPerFold[i], label=f"Train")
                axs[i].plot(self.validationLossesPerFold[i], label=f"Val")
                axs[i].title("Average loss per epoch", fontsize=30)
                axs[i].xlabel("Epochs")
                axs[i].legend()

        else:
            plt.plot(self.trainingLossesPerFold, label=f"Train")
            plt.plot(self.validationLossesPerFold, label=f"Val")
            plt.title("Average loss per epoch", fontsize=30)
            plt.xlabel("Epochs")
            plt.legend()

        plt.show()

    def printResults(self, results, testResult: bool = False, fullReport: bool = False):
        if testResult:
            typeResults = "Test"
        else:
            typeResults = "Training"

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
