import logging
from typing import List

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn

from CustomLogger import CustomLogger
from NeuralNetwork.NeuralNetworks import NeuralNetworkSTFT
from Timer import Timer
from featureExtraction.FeatureExtractor import FeatureExtractorSTFT
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
    trainingPath = getDataRoot().joinpath("recordings")
    testPath = getDataRoot().joinpath("testData")
    noisePath = getDataRoot().joinpath(r"noiseProfile\noiseProfile2.wav")

    # Define the type of data and add noise profile for filtering
    filterExtr = FeatureExtractorSTFT()
    filterExtr.noiseProfile = noisePath

    # Choose the participants from which the data will be used
    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]

    # Create training dataset
    testDataset = FootstepDataset(testPath, fExtractor=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\STFTtest"))

    # Create type of neural network
    network = NeuralNetworkSTFT(len(participants), testDataset.featureSize, nn.init.kaiming_uniform_)
    network.loadModel(getDataRoot().joinpath("model/NeuralNetworkSTFT-BestFromBatch-1.pth"))

    # Set training parameters
    nTrainings = 10
    batchSize = 32

    testResult, confMat = network.testOnData(testData=testDataset, batchSize=batchSize)

    testLoss: List[float] = testResult["Loss"]
    testAcc: List[float] = testResult["Accuracy"]

    logger.info(f"\nRESULT OF TEST")
    logger.info("=" * 30)
    logger.info(f"Validation loss: {testLoss.pop():.5f}")
    logger.info(f"Validation accuracy: {testAcc.pop()}%")
    logger.info("-" * 30)

    disp = ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=testDataset.labelStrings).plot()
    plt.show()