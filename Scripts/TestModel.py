import logging
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn

import utils
from CustomLogger import CustomLogger
from NeuralNetwork.NeuralNetworks import NeuralNetworkSTFT, NeuralNetworkTKEO2
from Timer import Timer
from featureExtraction.FeatureExtractor import FeatureExtractorSTFT, FeatureExtractorTKEO
from featureExtraction.Transforms import AddOffset
from footstepDataset.FootstepDataset import FootstepDataset
from utils import getDataRoot


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
    testPath = getDataRoot().joinpath("testData")
    noisePath = getDataRoot().joinpath(r"noiseProfile\noiseProfile2.wav")

    # Define the type of data and add noise profile for filtering
    featureExtractor = FeatureExtractorSTFT()
    featureExtractor.noiseProfile = noisePath

    # Define the type of transformation on the data
    transformer = AddOffset(amount=10, maxTimeOffset=1)

    # Choose the participants from which the data will be used
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "Lieve"]

    testDataset = FootstepDataset(testPath, fExtractor=featureExtractor, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache/TKEO"), transformer=transformer)
    # Create type of neural network
    network = NeuralNetworkSTFT(len(participants), testDataset.featureSize, nn.init.kaiming_uniform_)
    network.loadModel(getDataRoot().joinpath(f"model/{network.name}-BestFromBatch-8.pth"))
    network.fExtractor = featureExtractor

    # Set training parameters
    batchSize = 32
    noiseFactors = (np.arange(21))
    accuracies = []
    losses = []
    SNR = []

    # Add different magnitudes of noise to the sample before filtering
    logger.info(f"Noise factors: {noiseFactors}")
    logger.info(f"Amount of datapoints: {len(testDataset.dataset)}")
    for factor in noiseFactors:
        # Create test dataset
        testDataset = FootstepDataset(testPath, fExtractor=featureExtractor, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache"), transformer=transformer, addNoiseFactor=factor)

        # Test the dataset on the network and get the results back
        testResult, confMat = network.testOnData(testData=testDataset, batchSize=batchSize)

        # Because in the beginning the noise is 0, we get a value of infinity. This is unreasonable, we we chose the value 10
        if factor != 0:
            avgSNR = np.mean(testDataset.SNR)
        else:
            avgSNR = 10

        # Extract the results
        testLoss: List[float] = testResult["Loss"]
        testAcc: List[float] = testResult["Accuracy"]
        testPrec: List[float] = testResult["Precision"]
        testRec: List[float] = testResult["Recall"]

        # Process the results
        losses.append(testLoss[0])
        accuracies.append(testAcc[0])
        SNR.append(avgSNR)

        # Document the results for every test
        logger.info(f"\nRESULT OF TEST WITH nFactor {factor}")
        logger.info("=" * 30)
        logger.info(f"Validation loss: {testLoss[0]:.5f}")
        logger.info(f"Validation accuracy: {testAcc[0]:.2f}%")
        logger.info(f"Validation precision: {testPrec[0]:.2f}%")
        logger.info(f"Validation recall: {testRec[0]:.2f}%")
        logger.info(f"Average SNR: {avgSNR}")
        logger.info("-" * 30)

        # Setup a plot for the confusin matrix to be plotted on and save that plot
        confMatAx = plt.subplot()
        confMatAx.set_xlabel('Predicted labels', fontsize=18)
        confMatAx.set_ylabel('True labels', fontsize=18)
        disp = ConfusionMatrixDisplay(confusion_matrix=confMat).plot(colorbar=False, ax=confMatAx)
        plt.savefig(str(utils.getDataRoot().joinpath(f"Figures/ConfMat_test_{network.name}_SNR{round(avgSNR * 100)}.png")))
        plt.close()

    # After all the testing, we print out the different losses and accuracies over the averageSNR of that test loop
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1,2)

    # Plot the losses over the SNR
    ax1.plot(SNR, losses)
    ax1.set_title("Losses over noise",fontsize=20)
    ax1.set_xlabel("SNR (dB)", fontsize=20)
    ax1.set_ylabel("Loss",fontsize=20)
    ax1.set_xscale("log")
    ax1.invert_xaxis()

    # Plot the accuracies over the SNR
    ax2.plot(SNR, accuracies)
    ax2.set_title("Accuracy over noise",fontsize=20)
    ax2.set_xlabel("SNR (dB)",fontsize=20)
    ax2.set_ylabel("Accuracy (%)",fontsize=20)
    ax2.set_xscale("log")
    ax2.invert_xaxis()

    # Show the plot
    plt.show()
