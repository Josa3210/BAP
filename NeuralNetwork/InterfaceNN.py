import torch
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn, device, Tensor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, dataset


class InterfaceNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = self.getDevice()
        self.results = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}
        self.trainingData = None
        self.batchSize = 64
        self.learningRate = 1e-5
        self.bestLR = 1e-5
        self.folds = 5
        self.epochs = 5

    def forward(self, x: Tensor):
        pass

    @property
    def trainingData(self):
        return self._trainingData

    @trainingData.setter
    def trainingData(self, data):
        self._trainingData = data

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

    def clearResults(self):
        self.results = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}

    def trainOnData(self,
                    trainingData: Dataset = None,
                    folds: int = None,
                    epochs: int = None,
                    batchSize: int = None,
                    lr: int = None,
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

        self.clearResults()

        if self.trainingData is None:
            print("Define trainingdata using network.setTrainingData()")
            return None

        lossFunction = nn.CrossEntropyLoss()

        kFold = KFold(n_splits=self.folds, shuffle=True)

        for fold, (train_ids, validation_ids) in enumerate(kFold.split(self.trainingData)):
            # Print
            if verbose:
                print(f'\nFOLD {fold}')
                print('=' * 30)

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

            # Start training epochs
            for epoch in range(self.epochs):
                if verbose:
                    # Print epoch
                    print(f'\nStarting epoch {epoch + 1}')
                    print("-" * 30)

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

                    if verbose:
                        if i % 500 == 1:
                            print(f"{i:4d} / {len(trainLoader)} batches: average loss = {currentLoss / i}")

            if verbose:
                # Evaluation for this fold
                print('-' * 30)

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
                self.results["Loss"].append(currentLoss)
                self.results["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred) * 100)
                self.results["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)
                self.results["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro", zero_division=0) * 100)

        return sum(self.results["Loss"]) / len(self.results["Loss"])

    def printResults(self, fullReport: bool = False):
        print("Results of training:")
        print("=" * 30)

        keys: list[str] = list(self.results.keys())
        folds = len(self.results[keys[0]])
        if fullReport:

            for i in range(folds):
                print(f"For fold {i:d}:")
                print("-" * 30)
                print(f"Accuracy: {self.results[keys[1]][i]:.2f}%")
                print(f"Precision: {self.results[keys[2]][i]:.2f}%")
                print(f"Recall: {self.results[keys[3]][i]:.2f}%")
                print("-" * 30)

        print("Average:")
        print("-" * 30)

        avgAccuracy = sum(self.results[keys[1]]) / folds * 100
        avgPrecision = sum(self.results[keys[2]]) / folds * 100
        avgRecall = sum(self.results[keys[3]]) / folds * 100

        print(f"Accuracy: {avgAccuracy:.2f}%")
        print(f"Precision: {avgPrecision:.2f}%")
        print(f"Recall: {avgRecall:.2f}%")
        print("=" * 30)

    def optimizeLR(self,
                   bounds: tuple[float, float],
                   trainingData: Dataset = None,
                   init_points: int = 5,
                   n_iter: int = 10,
                   folds: int = None,
                   epochs: int = None,
                   batchSize: int = None):

        # Initialize parameters
        if trainingData is not None:
            self.trainingData = trainingData
        if folds is not None:
            self.folds = folds
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize

        print("Start optimization")
        # Give the parameter space from which the optimizer can choose
        parameterBounds = {"lr": (bounds[0], bounds[1])}

        # Create the optimizer object
        optimizer = BayesianOptimization(
            f=self.trainOnData,
            pbounds=parameterBounds
        )

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )

        self.bestLR = optimizer.max["params"]["lr"]
        print(f"Best learning rate is: {self.bestLR:.8f}")
