import numpy as np
from matplotlib import pyplot as plt

from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO
from customLogger import CustomLogger
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset
from utils import getDataRoot

if __name__ == '__main__':
    logger = CustomLogger.getLogger(__name__)

    trainingPath = getDataRoot().joinpath("recordings")
    testPath = getDataRoot().joinpath("testData")
    noisePath = trainingPath.joinpath(r"noiseProfile\noiseProfile2.wav")

    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = noisePath

    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    trainingDataset = FootstepDataset(trainingPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO441"))
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEOtest441"))

    network = NeuralNetworkTKEO(len(participants), trainingDataset.featureSize)

    batchSize = 32
    nTrainings = 10

    trainingResults = []
    trainingAccuracy = []
    testResults = []
    testAccuracy = []
    lossPerFold = []

    logger.info(f"Start training for {nTrainings} trainings\n")
    for i in range(nTrainings):
        logger.info("=" * 30)
        logger.info(f"Start training {i + 1}")
        logger.info("=" * 30)
        trainingResults.append(-network.trainOnData(trainingData=trainingDataset, folds=5, epochs=30, lr=0.0003, dr=0.8, batchSize=batchSize, verbose=False))
        trainingAccuracy.append(sum(network.trainingResults["Accuracy"]) / len(network.trainingResults["Accuracy"]))
        network.printResults(fullReport=False)
        testResults.append(network.testOnData(testData=testDataset))
        network.printResults(testResult=True)
        lossPerFold.append(network.lossesPerFold)
        testAccuracy.append(sum(network.testResults["Accuracy"]) / len(network.testResults["Accuracy"]))
    logger.info(f"Finished training")

    lossPerFold = np.array(lossPerFold)
    for j in range(nTrainings):
        meanLoss = lossPerFold[j].mean(axis=0)
        plt.plot(meanLoss, label=f"Training {j + 1}")
    plt.title("Average loss per epoch", fontsize=30)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    logger.info(f"\nRESULT OF {nTrainings} TRAININGS")
    logger.info("=" * 30)
    logger.info(f"Average training results: {sum(trainingResults) / len(trainingResults):.5f}")
    logger.info(f"Average training accuracy: {sum(trainingAccuracy) / len(trainingAccuracy):.2f}%")
    logger.info("-" * 30)
    logger.info(f"Average test results: {sum(testResults) / len(testResults):.5f}")
    logger.info(f"Average test accuracy: {sum(testAccuracy) / len(testAccuracy):.2f}%")
    logger.info("=" * 30)
