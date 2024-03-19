import os

from matplotlib import pyplot as plt

from featureExtraction.FeatureExtractor import FeatureExtractor

if __name__ == '__main__':
    currentPath = os.getcwd()
    extractor = FeatureExtractor(currentPath + r"\data", "out")
    result = extractor.extract()

    while True:
        try:
            plt.plot(next(result))
            plt.show()
        except StopIteration:
            break
