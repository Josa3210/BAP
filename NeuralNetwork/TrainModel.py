import logging
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO, NeuralNetworkTKEO2, NeuralNetworkSTFT
from CustomLogger import CustomLogger
from Timer import Timer
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO, FeatureExtractorSTFT
from footstepDataset.FootstepDataset import FootstepDataset
from utils import getDataRoot

"""
This function will train n times the same neural network and gather information about the results.

Training:
- Will create/load the right dataset.
- Training is done nTraining times.
- The parameters are defined in advance (epochs, folds, batchSize, ...)

Results:
- Validation loss of each training
- Min and Max validation loss
- Average validation loss
- Std of validation loss
- Best model:
    - state_dict
    - validation loss
    - confusion matrix
"""


def printDict(dict, logger: logging.Logger):
    for key in dict.keys():
        str = f"{key:<10}: "
        for value in dict.get(key):
            if key == "Loss":
                str += f"{value:.4f<5} "
            else:
                str += f"{value:.2f<5}% "
        logger.info(str)


if __name__ == '__main__':
    # Get objects for debug/documentation
    logger = CustomLogger.getLogger(__name__)
    timer = Timer()

    # Define the path to the different data files
    trainingPath = getDataRoot().joinpath("recordings")
    testPath = getDataRoot().joinpath("testData")
    noisePath = getDataRoot().joinpath(r"noiseProfile\noiseProfile2.wav")

    # Define the type of data and add noise profile for filtering
    filterExtr = FeatureExtractorSTFT()
    filterExtr.noiseProfile = noisePath

    # Choose the participants from which the data will be used
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]

    # Create training dataset
    trainingDataset = FootstepDataset(trainingPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\STFT"))

    # Create type of neural network
    network = NeuralNetworkSTFT(len(participants), trainingDataset.featureSize, nn.init.kaiming_uniform_)

    # Set training parameters
    nTrainings = 10
    batchSize = 32
    learningRate = 0.001
    network.dropoutRate = 0.2
    folds = 5
    epochs = 100

    # Initialise variables
    trainingResults = []
    trainingAccuracy = []
    valLossPerFold = []
    trainLossPerFold = []

    bestResult = 0
    bestConfMat = None
    id = 1

    # Start training
    logger.info(f"Start training for {nTrainings} trainings\n")
    timer.start()

    for i in range(nTrainings):
        validationResults, confMat = network.trainOnData(trainingData=trainingDataset, verbose=False, folds=folds, lr=learningRate, epochs=epochs, batchSize=batchSize)

        trainingResults.extend(validationResults["Loss"])
        trainingAccuracy.extend(validationResults["Accuracy"])
        valLossPerFold.append(network.validationLossesPerFold)
        trainLossPerFold.append(network.trainingLossesPerFold)

        logger.info("=" * 30)
        logger.info(f"Training {i + 1} results:")
        printDict(validationResults, logger)
        logger.info("=" * 30)
    timer.stop()
    logger.info(f"Finished training in {timer.get_elapsed_time() // 60} minutes {timer.get_elapsed_time() % 60:.2f} seconds ")

    valLossPerFold = np.array(valLossPerFold)
    trainLossPerFold = np.array(trainLossPerFold)
    fig, axs = plt.subplots(1, nTrainings, sharex=True, sharey=True)
    for j in range(nTrainings):
        trainMeanLoss = trainLossPerFold[j].mean(axis=0)
        valMeanLoss = valLossPerFold[j].mean(axis=0)
        axs[j].plot(trainMeanLoss, label=f"Training {j + 1}")
        axs[j].plot(valMeanLoss, "--", label=f"Validation {j + 1}")
    fig.suptitle("Average loss per epoch", fontsize=25)
    fig.supxlabel("Epochs")
    fig.supylabel("Average loss")

    logger.info(f"\nRESULT OF {nTrainings} TRAININGS")
    logger.info("=" * 30)
    logger.info("Validation loss")
    logger.info(f"Average:\t{np.mean(trainingResults):.5f}")
    logger.info("Validation accuracy:")
    logger.info(f"Average:\t{np.mean(trainingAccuracy):.2f}%")
    logger.info(f"Maximum:\t{np.max(trainingAccuracy):.2f}%")
    logger.info(f"Minimum:\t{np.min(trainingAccuracy):.2f}%")
    logger.info(f"Variance:\t{np.std(trainingAccuracy):.2f}%")
    logger.info("-" * 30)

    plt.show()
