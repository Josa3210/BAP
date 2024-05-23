import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn

import utils
from CustomLogger import CustomLogger
from NeuralNetwork.EarlyStopper import EarlyStopper
from NeuralNetwork.NeuralNetworks import NeuralNetworkSTFT, NeuralNetworkTKEO2, NeuralNetworkTKEO
from Timer import Timer
from featureExtraction.FeatureExtractor import FeatureExtractorSTFT, FeatureExtractorTKEO
from featureExtraction.Transforms import AddOffset
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
                str += f"{value:<5.4f} "
            else:
                str += f"{value:<5.2f}% "
        logger.info(str)


if __name__ == '__main__':
    # Get objects for debug/documentation
    logger = CustomLogger.getLogger(__name__)
    timer = Timer()

    # Define the path to the different data files
    trainingPath = getDataRoot().joinpath("trainingData")
    noisePath = getDataRoot().joinpath(r"noiseProfile\noiseProfile2.wav")

    # Define the type of data and add noise profile for filtering
    filterExtr = FeatureExtractorSTFT()
    filterExtr.noiseProfile = noisePath

    # Add a transformation to the data for more datapoints
    transformer = AddOffset(amount=10, maxTimeOffset=1)

    # Choose the participants from which the data will be used
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]

    # Create training dataset
    trainingDataset = FootstepDataset(trainingPath, fExtractor=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO"), transformer=transformer)

    # Create type of neural network
    network = NeuralNetworkSTFT(len(participants), trainingDataset.featureSize, nn.init.kaiming_uniform_)

    earlyStopper = EarlyStopper(network=network, amount=10, delta=0.01)

    # Set training parameters
    nTrainings = 1
    batchSize = 32
    learningRate = 0.01
    network.dropoutRate = 0.2
    folds = 5
    epochs = 200

    # Initialise variables
    trainingResults = []
    trainingAccuracy = []
    valLossPerFold = []
    trainLossPerFold = []

    bestResult = 0
    bestConfMat = None
    id = 8

    logger.info(network.dropoutRate)
    # Start training
    logger.info(f"Start training for {nTrainings} trainings")
    logger.info(f"Amount of data: {len(trainingDataset.dataset)}")
    timer.start()

    for i in range(nTrainings):
        validationResults, confMat = network.trainOnData(trainingData=trainingDataset, verbose=False, folds=folds, lr=learningRate, epochs=epochs, batchSize=batchSize, saveModel=True)

        trainingResults.extend(validationResults["Loss"])
        trainingAccuracy.extend(validationResults["Accuracy"])
        valLossPerFold.append(network.validationLossesPerFold)
        trainLossPerFold.append(network.trainingLossesPerFold)

        logger.info("=" * 30)
        logger.info(f"Training {i + 1} results:")
        printDict(validationResults, logger)
        logger.info("=" * 30)

        if max(validationResults["Accuracy"]) > bestResult:
            network.saveModel(maxVal= network.maxVal, name="BestFromBatch", idNr=id)
            bestResult = max(validationResults["Accuracy"])
            bestConfMat = confMat

    timer.stop()
    logger.info(f"Finished training in {timer.get_elapsed_time() // 60} minutes {round(timer.get_elapsed_time() % 60)} seconds")
    """
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
    """
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

    loadedDict = torch.load(network.savePath.joinpath(f"{network.name}-BestFromBatch-{id}.pth"))
    logger.info(loadedDict["maxVal"])

    confMatAx = plt.subplot()
    confMatAx.set_xlabel('Predicted labels', fontsize=18)
    confMatAx.set_ylabel('True labels', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=bestConfMat).plot(colorbar=False, ax=confMatAx)
    plt.savefig(str(utils.getDataRoot().joinpath(f"Figures/ConfMat_train{network.name}.png")))
    disp.plot()

    plt.show()
