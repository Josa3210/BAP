import os.path
import matlab.engine
import torch


class FeatureExtractor:

    def __init__(self, funcPath: str = "matlabFunctions/extractFeatures.m"):
        # Get the directory where this file is locate and add the path to the function to it
        self.funcPath = os.path.dirname(os.path.realpath(__file__)) + "\\" + funcPath

        # Check if the path to the featureExtraction.m file exists
        if not os.path.isfile(self.funcPath):
            print(f"{self.funcPath} has not been found! Please add this file or specify location in the constructor (funcPath=)")
            return

        # Matlab engine for running the necessary functions
        self.eng = matlab.engine.start_matlab()

        # Set matlab directory to current directory
        self.eng.cd(os.path.dirname(os.path.realpath(__file__)))

    # Extract all the .wav files and convert them into a readable file
    def extract(self, startPath: str):
        for file in os.listdir(startPath):
            if file.endswith(".wav"):
                # Combine filepath with current file
                filePath = startPath + "\\" + file
                label = file.split(".wav")[0].split("_")[0]

                # Send data to Matlab and receive the transformed signal
                result = self.eng.extractFeatures(filePath)

                # Convert to tensor and flatten to remove 1 dimension
                torchResult = torch.flatten(torch.Tensor(result))

                yield torchResult, label
