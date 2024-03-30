import torch
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn, device, Tensor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, dataset


class InterfaceNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = self.getDevice()
        self.results = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}

    def forward(self, x: Tensor):
        pass

    @staticmethod
    def getDevice():
        return device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def clearResults(self):
        self.results = {"Loss": [], "Accuracy": [], "Precision": [], "Recall": []}

    def trainOnData(self,
                    trainingData: Dataset,
                    verbose: bool = False,
                    folds: int = 5,
                    epochs: int = 5,
                    batchSize: int = 64,
                    lr: float = 1e-5,
                    optimize: bool = False):

        self.clearResults()

        if optimize:
            verbose = False

        lossFunction = nn.CrossEntropyLoss()

        kFold = KFold(n_splits=folds, shuffle=True)

        for fold, (train_ids, validation_ids) in enumerate(kFold.split(trainingData)):
            # Print
            if verbose:
                print(f'\nFOLD {fold}')
                print('=' * 30)

            # Sample elements randomly from a given list of ids, no replacement.
            trainSubSampler = SubsetRandomSampler(train_ids)
            validationSubSampler = SubsetRandomSampler(validation_ids)

            # Define data loaders for training and testing data in this fold
            trainLoader = DataLoader(
                trainingData,
                batch_size=batchSize, sampler=trainSubSampler)
            validationLoader = DataLoader(
                trainingData,
                batch_size=batchSize, sampler=validationSubSampler)

            # Get the network to the right device
            self.to(self.device)
            optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=lr)

            # Start training epochs
            for epoch in range(epochs):
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

                    if not optimize:
                        # Save data in list
                        confMatPred.extend(predicted.data.cpu().numpy())
                        confMatTarget.extend(targets.data.cpu().numpy())

                self.results["Loss"].append(currentLoss)

                if not optimize:
                    # Calculate confusion matrix and metrics
                    self.results["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred))
                    self.results["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro"))
                    self.results["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro"))

        return sum(self.results["Loss"]) / len(self.results["Loss"])

    def printResults(self):
        print("Results of training:")
        print("=" * 30)

        keys: list[str] = list(self.results.keys())
        folds = len(self.results[keys[0]])
        for i in range(folds):
            print(f"For fold {i:d}:")
            print("-" * 30)
            print(f"Accuracy: {self.results[keys[1]][i]:.2f}%")
            print(f"Precision: {self.results[keys[2]][i]:.2f}%")
            print(f"Recall: {self.results[keys[3]][i]:.2f}%")
            print("-" * 30)
        print("\nAverages:")
        print("-" * 30)

        avgAccuracy = sum(self.results[keys[1]]) / folds
        avgPrecision = sum(self.results[keys[2]]) / folds
        avgRecall = sum(self.results[keys[3]]) / folds

        print(f"Accuracy: {avgAccuracy:.2f}%")
        print(f"Precision: {avgPrecision:.2f}%")
        print(f"Recall: {avgRecall:.2f}%")
        print("=" * 30)
