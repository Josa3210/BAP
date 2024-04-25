import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils
from NeuralNetwork.NeuralNetworks import NeuralNetworkTKEO
from featureExtraction.FeatureExtractor import FeatureExtractorTKEO
from footstepDataset.FootstepDataset import FootstepDataset

if __name__ == '__main__':
    path = utils.getDataRoot().joinpath("recordings")
    filterExtr = FeatureExtractorTKEO()
    filterExtr.noiseProfile = path.joinpath(r"noiseProfile\noiseProfile2.wav")
    participants = ["sylvia", "tine", "patrick", "celeste", "simon"]
    dataset = FootstepDataset(path, transform=filterExtr, labelFilter=participants, cachePath=utils.getDataRoot().joinpath(r"cache\TKEO4410"))
    labels = dataset.labelStrings
    batchSize = 4
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    print(f"Amount of batches: {len(dataloader)}")
    trainingFeatures, trainingLabels = next(iter(dataloader))
    print(f"Feature batch shape: {trainingFeatures.size()}")
    print(f"Labels batch shape: {trainingLabels.size()}")
    model = NeuralNetworkTKEO(5, dataset.featureSize)
    model.loadModel(utils.getDataRoot().joinpath("models/NeuralNetworkTKEO-6078_resample4410.pth"))
    model.eval()

    out_data = model.fLayers.forward(trainingFeatures.unsqueeze(1))
    filters = model.l5[0]
    fig, axs = plt.subplots(batchSize, filters)

    for i in range(batchSize):
        for j in range(filters):
            data = out_data[i, j]
            data = data.detach().numpy()
            axs[i, j].plot(data)
            axs[i, j].set_yticks([])
    plt.show()
