import os

from featureExtraction.FeatureExtractor import FeatureExtractor

if __name__ == '__main__':
    print("Hello World")
    currentPath = os.getcwd()
    print(currentPath + r"\data")
    extractor = FeatureExtractor(currentPath + r"\data", "out")
    result = extractor.extract()

