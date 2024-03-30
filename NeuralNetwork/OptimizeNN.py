import torch
from bayes_opt import BayesianOptimization, JSONLogger, Events
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torchvision import datasets

from NeuralNetwork.MNIST_NN import MNISTDataset, MnistNN

def simpleTrain(lr: float = 1e-4):
    # Parameters
    folds = 3
    epochs = 5
    batchSize = 32
    lr = lr
    lossFunction = nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mnistTrainSet = datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=None)
    mnistTestSet = datasets.MNIST(root='./MNISTdata', train=False, download=True, transform=None)

    trainDataset = MNISTDataset(mnistTrainSet)
    testDataset = MNISTDataset(mnistTestSet)

    dataset = ConcatDataset([trainDataset, testDataset])

    kFold = KFold(n_splits=folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kFold.split(dataset)):
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

        currentLoss = 0.
        with torch.no_grad():
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

                loss = lossFunction(outputs, targets)
                currentLoss += loss.item()
    return -currentLoss


if __name__ == '__main__':

    # Give the parameter space from which the optimizer can choose
    parameterBounds = {"lr": (1e-6, 1e-3)}

    # Create the optimizer object
    optimizer = BayesianOptimization(
        f=simpleTrain,
        pbounds=parameterBounds
    )

    print("Start optimization")
    optimizer.maximize(
        init_points=3,
        n_iter=5
    )

    print(f"Best params are: {optimizer.max}")
