import os

import PIL.Image
import torch
import torchvision.datasets
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import SubsetRandomSampler, ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms

from footstepDataset.FootstepDataset import FootstepDataset


class MnistNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3),
            nn.ReLU()
        )

        self.LinLayers = nn.Sequential(
            nn.Linear(10 * 24 * 24, 265),
            nn.ReLU(),
            nn.Linear(265, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        # print(x.size())
        x = self.ConvLayers.forward(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.LinLayers(x)
        return x


class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data: torchvision.datasets.MNIST):
        self.data = data.data
        self.labels = data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index].to(torch.float).unsqueeze(0), self.labels[index]
        return img, label


def trainNN():
    # Parameters
    folds = 5
    epochs = 3
    lr = 1e-4
    lossFunction = nn.CrossEntropyLoss()

    # Get dataset
    currentPath = r"D:\_Opslag\GitKraken\BAP"
    path = currentPath + r"\data"
    dataset = FootstepDataset(path, "Ann")

    kFold = KFold(n_splits=folds, shuffle=True)
    confMatrix = {"Accuracy": [], "ClassError": [], "Recall": [], "Precision": []}

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
        network: nn.Module = None
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
        with torch.no_grad():
            entries = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            # Iterate over the test data and generate predictions
            for i, batch in enumerate(testloader):
                # Get inputs
                inputs, targets = batch

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                for target, predict in zip(targets, predicted):
                    entries += 1
                    match (target, predict):
                        case (0, 1):
                            FP += 1
                        case (1, 0):
                            FN += 1
                        case (0, 0):
                            TN += 1
                        case (1, 1):
                            TP += 1
            # Print accuracy
            accuracy = ((TP + TN) / entries) * 100
            classificationError = ((FP + FN) / entries) * 100
            recall = (TP / (TP + FN)) * 100
            precision = (TP / (TP + FP)) * 100

            confMatrix["Accuracy"].append(accuracy)
            confMatrix["ClassError"].append(classificationError)
            confMatrix["Recall"].append(recall)
            confMatrix["Precision"].append(precision)

            print(f"Accuracy for fold {fold:d}: {accuracy:.2f}%")
            print('=' * 30)

        # Calculate average of confusion matrix
        avgAccuracy = sum(confMatrix["Accuracy"]) / len(confMatrix["Accuracy"])
        avgClassErr = sum(confMatrix["ClassError"]) / len(confMatrix["ClassError"])
        avgRecall = sum(confMatrix["Recall"]) / len(confMatrix["Recall"])
        avgPrecision = sum(confMatrix["Precision"]) / len(confMatrix["Precision"])

        # Print confusion matrix
        print("\n Average performance statistics")
        print("=" * 30)
        print(f"Accuracy: {avgAccuracy:.2f}%")
        print(f"Classification Error: {avgClassErr:.2f}%")
        print(f"Recall: {avgRecall:.2f}%")
        print(f"Precision: {avgPrecision:.2f}%")
        print("=" * 30)

        print(confMatrix)


if __name__ == '__main__':
    trainNN()
