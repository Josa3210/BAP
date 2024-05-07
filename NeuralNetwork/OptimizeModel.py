from torch import nn

from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO2
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset
from utils import getDataRoot

if __name__ == '__main__':
    path = getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = getDataRoot().joinpath(r"noiseProfile\noiseProfile2.wav")

    participants = ["sylvia", "tine", "patrick", "celeste", "simon", "walter", "ann", "jan", "lieve"]
    dataset = FootstepDataset(path, fExtractor=filterExtr, labelFilter=participants, cachePath=getDataRoot().joinpath(r"cache\TKEO441"))

    batchSize = 32
    sFeatures = dataset.featureSize
    network = NeuralNetworkTKEO2(len(participants), sFeatures, nn.init.kaiming_uniform_)

    network.dropoutRate = 0.20
    folds = 1
    epochs = 350

    bounds = {"lr": (0.0001, 0.05)}
    results = network.optimizeParams(init_points=10, n_iter=10, bounds=bounds, trainingData=dataset)
