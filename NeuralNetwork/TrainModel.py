import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO, NeuralNetworkTKEO2
from CustomLogger import CustomLogger
from Timer import Timer
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
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


def printDict(dict: dict[str, float], logger: CustomLogger):
    for key in dict.keys():
        str = key + "\t\t:\t"
        for value in dict.get(key):
            str += f"{value:.2f} "
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
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = noisePath

    # Choose the participants from which the data will be used
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]

    # Create training dataset
    trainingDataset = FootstepDataset(trainingPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO441"))

    # Create type of neural network
    network = NeuralNetworkTKEO2(len(participants), trainingDataset.featureSize, nn.init.kaiming_uniform_)

    # Set training parameters
    nTrainings = 10
    batchSize = 32
    learningRate = 0.00045
    network.dropoutRate = 0.2
    folds = 1
    epochs = 350

    # Initialise variables
    trainingResults = []
    trainingAccuracy = []
    lossPerFold = []

    # Start training
    logger.info(f"Start training for {nTrainings} trainings\n")
    timer.start()

    for i in range(nTrainings):
        validationResults, confMat = network.trainOnData(trainingData=trainingDataset, verbose=False, folds=folds, lr=learningRate, epochs=epochs, batchSize=batchSize)

        trainingResults.append(*validationResults["Loss"])
        trainingAccuracy.append(*validationResults["Accuracy"])
        lossPerFold.append(network.trainingLossesPerFold)

        logger.info("=" * 30)
        logger.info(f"Training {i + 1} done")
        logger.info(f"Results:")
        printDict(validationResults, logger)
        logger.info("=" * 30)
    timer.stop()
    logger.info(f"Finished training in {timer.get_elapsed_time() // 60} minutes {timer.get_elapsed_time() % 60:.2f} seconds ")

    lossPerFold = np.array(lossPerFold)
    fig, axs = plt.subplots(1, 2)
    for j in range(nTrainings):
        meanLoss = lossPerFold[j].mean(axis=0)
        axs[0].plot(meanLoss, label=f"Training {j + 1}")
    axs[0].set_title("Average loss per epoch", fontsize=30)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Average loss")
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs[1].plot(trainingResults, label="Training loss")
    axs[1].set_title("Training loss", fontsize=30)
    axs[1].set_xlabel("Training")
    axs[1].set_ylabel("Loss")

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
