import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO
from CustomLogger import CustomLogger
from Timer import Timer
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset
from utils import getDataRoot

if __name__ == '__main__':
    logger = CustomLogger.getLogger(__name__)
    timer = Timer()

    trainingPath = getDataRoot().joinpath("recordings")
    testPath = getDataRoot().joinpath("testData")
    noisePath = trainingPath.joinpath(r"noiseProfile\noiseProfile2.wav")

    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = noisePath

    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    trainingDataset = FootstepDataset(trainingPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO441"))
    testDataset = FootstepDataset(testPath, transform=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEOtest441"))

    network = NeuralNetworkTKEO(len(participants), trainingDataset.featureSize, nn.init.kaiming_uniform_)

    batchSize = 32
    nTrainings = 10

    trainingResults = []
    trainingAccuracy = []
    testResults = []
    testAccuracy = []
    lossPerFold = []

    logger.info(f"Start training for {nTrainings} trainings\n")
    timer.start()
    for i in range(nTrainings):
        logger.info("=" * 30)
        logger.info(f"Start training {i + 1}")
        logger.info("=" * 30)
        trainingResults.append(-network.trainOnData(trainingData=trainingDataset, folds=5, epochs=450, lr=0.0003, dr=0.8, batchSize=batchSize, verbose=False))
        trainingAccuracy.append(sum(network.validationResults["Accuracy"]) / len(network.validationResults["Accuracy"]))
        network.printResults(fullReport=False)
        testResults.append(network.testOnData(testData=testDataset))
        network.printResults(testResult=True)
        lossPerFold.append(network.trainingLossesPerFold)
        testAccuracy.append(sum(network.testResults["Accuracy"]) / len(network.testResults["Accuracy"]))
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
    axs[1].plot(testResults, label="Test loss")
    axs[1].set_title("Test and training loss", fontsize=30)
    axs[1].set_xlabel("training")
    axs[1].set_ylabel("loss")
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    logger.info(f"\nRESULT OF {nTrainings} TRAININGS")
    logger.info("=" * 30)
    logger.info(f"Average training results:\t{np.mean(trainingResults):.5f}")
    logger.info(f"Average training accuracy:\t{np.mean(trainingAccuracy):.2f}%")
    logger.info("-" * 30)
    logger.info(f"Average test results:\t{np.mean(testResults):.5f}")
    logger.info(f"Average test accuracy:\t{np.mean(testAccuracy):.2f}%")
    logger.info("-" * 30)
    logger.info(f"Average accuracy:\t{np.mean([testAccuracy + trainingAccuracy]):.2f}%")
    logger.info(f"Stdev accuracy:\t\t{np.std([testAccuracy + trainingAccuracy]):.2f}%")
    logger.info(f"Max accuracy: {max(testAccuracy):.2f}%")
    logger.info(f"Min accuracy: {min(testAccuracy):.2f}%")
    logger.info("=" * 30)

    plt.show()
