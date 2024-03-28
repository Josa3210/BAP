import os

import PIL.Image
import torch
import torchvision.datasets
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import SubsetRandomSampler, ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms

from NeuralNetwork.MNIST_NN import MNISTDataset, MnistNN
from footstepDataset.FootstepDataset import FootstepDataset


def trainNN():
    # Parameters
    folds = 5
    epochs = 5
    lr = 1e-4
    lossFunction = nn.CrossEntropyLoss()

    mnistTrainSet = datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=None)
    dataset = MNISTDataset(mnistTrainSet)

    kFold = KFold(n_splits=folds, shuffle=True)
    results = dict()

    for fold, (train_ids, test_ids) in enumerate(kFold.split(dataset)):
        # Print
        print(f'\nFOLD {fold}')
        print('=' * 30)

        # Sample elements randomly from a given list of ids, no replacement.
        trainSubSampler = SubsetRandomSampler(train_ids)
        testSubSampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64, sampler=trainSubSampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64, sampler=testSubSampler)

        # Get the network
        network: nn.Module = MnistNN()
        optimizer: torch.optim.Optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        # Start training epochs
        for epoch in range(epochs):

            # Print epoch
            print(f'\nStarting epoch {epoch + 1}')
            print("-" * 30)

            # Set current loss value
            currentLoss = 0.

            # Iterate over the DataLoader for training data
            for i, batch in enumerate(trainloader):
                # Get inputs
                inputs, targets = batch

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

                if i % 500 == 0:
                    print(f"{i:4d} / {len(trainloader)} batches processed")
        # Evaluation for this fold
        print('-' * 30)
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, batch in enumerate(testloader):
                # Get inputs
                inputs, targets = batch

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('-' * 30)
            results[fold] = (100.0 * (correct / total))

        # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS')
    print('=' * 30)
    sum = 0.
    for key, value in results.items():
        print(f'Fold {key}: {value:.2f} %')
        sum += value
    print('=' * 30)
    print(f'Average: {sum / len(results.items()):.2f} %')
    print('=' * 30)


if __name__ == '__main__':
    trainNN()
