from torch import Tensor, nn
from torch.utils.data import DataLoader

import utils
from NeuralNetwork.InterfaceNN import InterfaceNN
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset


class NeuralNetworkTKEO(InterfaceNN):
    def __init__(self, nPersons: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(176319, 1024),
            nn.Dropout(self.dropoutRate),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(self.dropoutRate),
            nn.ReLU(),
            nn.Linear(256, nPersons),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor):
        return self.layers.forward(x)


if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    filterExtr.setCachePath(utils.getDataRoot().joinpath(r"cache\TKEO"))
    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, len(participants), transForm=filterExtr, filter=participants)
    labels = dataset.labelArray
    batchSize = 4
    network = NeuralNetworkTKEO(len(participants))

    bounds = {"lr": (1e-5, 1e-3), "dr": (0.2, 0.7)}

    network.optimizeParams(
        bounds=bounds,
        trainingData=dataset,
        n_iter=10,  # n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
        init_points=5)  # init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    network.trainOnData(dataset, 5, 5, batchSize, verbose=True)
    network.printResults(True)
