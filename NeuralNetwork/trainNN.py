import os

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import SubsetRandomSampler

from footstepDataset.FootstepDataset import FootstepDataset


def trainNN():
    # Parameters
    folds = 5
    epochs = 1
    lr = 1e-4
    lossFunction = nn.CrossEntropyLoss()

    # Get dataset
    currentPath = r"D:\_Opslag\GitKraken\BAP"
    path = currentPath + r"\data"

    dataset = FootstepDataset(path, "Ann")
    kfold = KFold(n_splits=folds, shuffle=True)
    results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'\nFOLD {fold}')
        print('--------------------------------')

        print(f"Train ids: {train_ids}")
        print(f"Test ids: {test_ids}")

        # Sample elements randomly from a given list of ids, no replacement.
        trainSubSampler = SubsetRandomSampler(train_ids)
        testSubSampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10, sampler=trainSubSampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10, sampler=testSubSampler)

        print('--------------------------------\n')

        # Get the network
        network: nn.Module = None
        optimizer: torch.optim.Optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        # Start training epochs
        for epoch in range(epochs):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            currentLoss = 0.

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs, targets = data

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

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)

        # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS')
    print('--------------------------------')
    sum = 0.
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')


if __name__ == '__main__':
    trainNN()
