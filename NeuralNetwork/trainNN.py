import os

import PIL.Image
import torch
import torchvision.datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import SubsetRandomSampler, ConcatDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from sklearn import metrics
from NeuralNetwork.MNIST_NN import MNISTDataset, MnistNN
from footstepDataset.FootstepDataset import FootstepDataset


def trainNN():
    # Parameters
    folds = 3
    epochs = 5
    batchSize = 32
    lr = 1e-4
    lossFunction = nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mnistTrainSet = datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=None)
    mnistTestSet = datasets.MNIST(root='./MNISTdata', train=False, download=True, transform=None)

    trainDataset = MNISTDataset(mnistTrainSet)
    testDataset = MNISTDataset(mnistTestSet)

    dataset = ConcatDataset([trainDataset, testDataset])

    kFold = KFold(n_splits=folds, shuffle=True)
    results = {"ConfMat": [], "Accuracy": [], "Precision": [], "Recall": []}

    for fold, (train_ids, test_ids) in enumerate(kFold.split(dataset)):
        # Print
        print(f'\nFOLD {fold}')
        print('=' * 30)

        # Sample elements randomly from a given list of ids, no replacement.
        trainSubSampler = SubsetRandomSampler(train_ids)
        testSubSampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainLoader = DataLoader(
            dataset,
            batch_size=batchSize, sampler=trainSubSampler)
        testLoader = DataLoader(
            dataset,
            batch_size=batchSize, sampler=testSubSampler)

        # Get the network
        network: nn.Module = MnistNN()
        network.to(device)
        optimizer: torch.optim.Optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        # Start training epochs
        for epoch in range(epochs):

            # Print epoch
            print(f'\nStarting epoch {epoch + 1}')
            print("-" * 30)

            # Set current loss value
            currentLoss = 0.
            # Iterate over the DataLoader for training data
            for i, batch in enumerate(trainLoader):
                # Get inputs
                inputs, targets = batch

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = lossFunction(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                currentLoss += loss.item()

                if i % 500 == 1:
                    print(f"{i:4d} / {len(trainLoader)} batches: average loss = {currentLoss / i}")

        # Evaluation for this fold
        print('-' * 30)

        # Lists for creating confusion matrix
        confMatPred, confMatTarget = [], []

        with torch.no_grad():
            entries = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            # Iterate over the test data and generate predictions
            for i, batch in enumerate(testLoader):
                # Get inputs
                inputs, targets = batch

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)

                # Save data in list
                confMatPred.extend(predicted.data.cpu().numpy())
                confMatTarget.extend(targets.data.cpu().numpy())

            # Calculate confusion matrix and metrics
            results["ConfMat"].append(metrics.confusion_matrix(confMatTarget, confMatPred))
            results["Accuracy"].append(metrics.accuracy_score(confMatTarget, confMatPred))
            results["Precision"].append(metrics.precision_score(confMatTarget, confMatPred, average="macro"))
            results["Recall"].append(metrics.recall_score(confMatTarget, confMatPred, average="macro"))

    return results


def printResults(results: dict[str, list[float]]):
    print("Results of training:")
    print("=" * 30)

    keys: list[str] = list(results.keys())
    folds = len(results[keys[0]])
    for i in range(folds):
        print(f"For fold {i:d}:")
        print("-" * 30)
        print(f"Accuracy: {results[keys[1]][i]:.2f}%")
        print(f"Precision: {results[keys[2]][i]:.2f}%")
        print(f"Recall: {results[keys[3]][i]:.2f}%")
        print("-" * 30)
    print("\nAverages:")
    print("-" * 30)

    avgAccuracy = sum(results[keys[1]]) / folds
    avgPrecision = sum(results[keys[2]]) / folds
    avgRecall = sum(results[keys[3]]) / folds

    print(f"Accuracy: {avgAccuracy:.2f}%")
    print(f"Precision: {avgPrecision:.2f}%")
    print(f"Recall: {avgRecall:.2f}%")
    print("=" * 30)
    return avgAccuracy, avgPrecision, avgRecall


if __name__ == '__main__':
    results: dict = trainNN()
    avgAccuracy, avgPrecision, avgRecall = printResults(results)
