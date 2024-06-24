import logging
import tkinter as tk
from pathlib import Path

import numpy as np
import torch

from Tools import PathFinder
from Tools import CustomLogger
from NeuralNetwork.InterfaceNN import InterfaceNN
from NeuralNetwork.NeuralNetworks import NeuralNetworkSTFT
from FootstepDataset.AudioRecorder import CustomAudioRecorder
from FeatureExtraction.FeatureExtractor import FeatureExtractorSTFT


class Application:
    """
    The Application class is used for predicting persons based on the sound of their footsteps.
    After the application is initialised and a noiseProfile has been recorded, the user can take recordings of
    footsteps, and the application will try to classify them and return which person it was.

    Attributes:
        modelPath (Path): The path to the model file.
        sampleRate (int): The sample rate in Hertz for audio processing.
        loggingLevel (int): The logging level for application debugging.
        recorder (CustomAudioRecorder): An audio recorder instance for recording audio.
        featureExtractor (FeatureExtractorSTFT): A feature extractor instance for extracting features from audio.
        logger (CustomLogger): A custom logger instance for logging application events.
        model (InterfaceNN): An interface for the neural network model.

    Methods:
        __init__(self, modelPath: Path, sampleRate: int = 44100, loggingLevel: int = logging.INFO)
            The initializer method for the Application class.
            Sets up the model, audio recorder, feature extractor, and GUI.

        setLoggingLevel(self, loggingLevel: int)
            Set the logging level for application logging.

        setNoiseProfile(self) -> bool
            Records the noise profile of the environment.
            Returns True if noise profile is successfully recorded, False otherwise.

        run(self) -> Tuple[str, float]
            Runs the audio processing pipeline for speech recognition.
            Returns the predicted label and confidence level for the audio.
    """

    def __init__(self, modelPath: Path, sampleRate: int = 44100, loggingLevel: int = logging.INFO):
        # Initialise parameters
        self.logger = CustomLogger.getLogger(__name__)
        self.logger.setLevel(loggingLevel)

        # Setup of the model
        self.logger.debug("Setting up application")
        self.logger.debug("-" * 30)
        self.logger.debug(f"Loading model from {modelPath}")
        self.model: InterfaceNN = NeuralNetworkSTFT(9, [50, 169])
        if not self.model.loadModel(modelPath):
            self.logger.error(f"No model found at path {modelPath}. Quitting construction.")
            return
        self.logger.debug(f"max value: {self.model.normalizationValue}")

        # Setup of audio recorder
        self.logger.debug("Installing audio recorder")
        audioSavePath: Path = PathFinder.getDataRoot().joinpath("application\saveFiles")
        self.sampleRate = sampleRate
        self.recorder: CustomAudioRecorder = CustomAudioRecorder(str(audioSavePath), self.sampleRate, 1, level=loggingLevel)

        # Setup of the feature extractor
        self.logger.debug("Installing feature extractor")
        self.featureExtractor: FeatureExtractorSTFT = FeatureExtractorSTFT()
        self.featureExtractor.start()

        # Record noiseProfile of the environment
        self.logger.debug("Recording noise profile")
        if not self.setNoiseProfile():
            self.logger.warning("No noise profile has been recorded at this time.")

        # Create GUI (In progress)
        window = tk.Tk()
        frm_main = tk.Frame()
        btn_run = tk.Button(
            master=frm_main,
            text="Predict",
            width=25,
            height=5,
        )
        btn_run.pack()
        frm_main.pack()
        window.mainloop()

    def setLoggingLevel(self, loggingLevel: int):
        self.logger.setLevel(loggingLevel)

    def setNoiseProfile(self) -> bool:
        # Ask if the user is ready to start recording
        startRecording = input("Ready for recording noise? (Y or N): ")
        if not startRecording.capitalize() == "Y":
            return False

        # Record 3 seconds of the environment noise
        noiseProfile = self.recorder.record(3)

        # Save the noiseProfile for later use
        noiseprofileStr = input("How is the noise profile called?")
        self.recorder.save(noiseProfile, noiseprofileStr)

        # Put the noiseProfile in the feature extractor for filtering the sounds
        self.featureExtractor.noiseProfile = noiseProfile
        return True

    def run(self):
        # Start recording the person
        recording = self.recorder.record(4, playBack=True)

        # Save the recording for later use
        filename: str = input("Give filename: ")
        self.recorder.save(recording, filename)

        # Start preprocessing the recorded audio
        self.logger.debug("Start feature extraction")
        self.logger.debug("Filtering signal from recording")
        filteredRecording = self.featureExtractor.filter(recording, self.sampleRate)
        self.logger.debug("Extracting features from filtered signal")
        extractedFeatures, _ = self.featureExtractor.transform(filteredRecording, self.sampleRate)
        extractedFeatures = extractedFeatures / self.model.normalizationValue
        extractedFeatures = torch.Tensor(extractedFeatures).unsqueeze(0)

        # Let the network classify the preprocessed signal
        self.logger.debug("Start predicting")
        predictionLogits = self.model(extractedFeatures).cpu().detach().numpy()

        # Extract the results from the prediction
        predictedVal = predictionLogits.max()   # This is the confidence of the prediction
        predictedInd = predictionLogits.argmax()
        predictedLabel = self.model.stringLabels[predictedInd]  # This is the name of the person

        # Print out the different results
        self.logger.debug(f"Predicted logits: {np.array(predictionLogits)[0]}")
        return predictedLabel, predictedVal


# Here we set everything up to use the application and do multiple predictions in sequence.
# After every prediction, the results are shared and the program waits to take another recording
if __name__ == '__main__':
    modelPath: Path = PathFinder.getDataRoot().joinpath("model").joinpath("NeuralNetworkSTFT-BestFromBatch-10.pth")
    application: Application = Application(modelPath=modelPath, loggingLevel=logging.DEBUG)
    inputStr: str = ""
    print("\n" + "=" * 100)
    print("Start predictions")
    while not inputStr.lower() == "exit":
        inputStr = input("Ready for recording? ")
        if inputStr.capitalize() == "Y":
            person, certainty = application.run()
            print("=" * 100)
            print(f"Predicted {person} with certainty of {certainty}\n")
            print("=" * 100)
