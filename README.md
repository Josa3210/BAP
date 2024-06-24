<head>
    <link rel="stylesheet" href="./style.css">
</head>
<body>

<div class=container>
<div class=titleBlock>

# BAP: Gait detection and identification using acoustic signatures

**A bachelor thesis with the goal of classifying persons based on the acoustic sound of their footsteps**


*Author*        : Joeri Winckelmans \
*Last updated*  : 25/5/2024 


</div>
<div class=ToC>

## Table of contents
- [Description](#description)
- [Structure](#structure)
- [Getting Started](#getting-started)
    - [FootstepDataset](#footstepdataset)
    - [TrainModel](#trainmodel)
    - [TestModel](#testmodel)
    - [Application WIP](#application-wip)
- [Requirements](#requirements)


</div>
<div class=Desc>

## Description

In this project you will find all the code to achieve the following:
- Create a dataset called *FootstepDataset* that can be used in the pytorch environment to train neural networks.
    - Reads in the files from a directory with .wav files.
    - Uses Matlab to filter the data:
        - Envelope filter
        - Spectral subtraction
    - Uses Matlab to extract features:
        - Taeger-Kaiser Energy Operator (TKEO)
        - Short-Time Fourier Transform (STFT)
- Create models to identify different persons based on the sound of their footsteps.
    - One model for the different extracted features:
        - NeuralNetworkTKEO (v1 and v2)
        - NeuralNetworkSTFT
- Create an application for live predictions of recorded sound.

</div>

<div class=Structure>

## Structure
This project has 6 main directories that contain the following files/directories:
- **[Application](./Application)**
    
    [Application](./Application/Application.py) class that combines the AudioRecorder, FeatureExtractor and a trained model from NeuralNetwork to make predictions on live recorded audio.

- **[FeatureExtraction](./FeatureExtraction)**

    [FeatureCacher](./FeatureExtraction/FeatureCacher.py) that caches the converted signal and loads them back in.\
    [FeatureExtractor](./FeatureExtraction/FeatureExtractor.py) class to filter recordings and transform them into input for the neural networks. \
    [MatlabFunctions](./FeatureExtraction/MatlabFunctions/) that holds the different functions used by the FeatureExtractor. \
    [TestFiltering](./FeatureExtraction/TestFiltering.py) to test the different functions of FeatureExtractor.\
    [Transforms](./FeatureExtraction/Transforms.py) class that can be passed on as parameter to the FootstepDataset to add a transformation to the filtered signals before being handled by the FeatureExtractor class.

- **[FootstepDataset](./FootstepDataset)**

    [AudioRecorder](./FootstepDataset/AudioRecorder.py) class to record sound and save it in .wav files.\
    [FootstepDataset](./FeatureExtraction/FootstepDataset.py) class to read in a series of .wav files and converts it to a Dataset from the Pytorch package.
    

- **[NeuralNetwork](./NeuralNetwork)**

    [EarlyStopper](./NeuralNetwork/EarlyStopper.py) class that will stop the training loop of a model once a certain criteria is met.\
    [InterfaceNN](./NeuralNetwork/InterfaceNN.py) interface class from where all other neural networks are derived. Contains also TrainableNN class that holds functions for optimizing, training and testing.\
    [NeuralNetworks](./NeuralNetwork/NeuralNetworks.py) class that contains the different architectures created for person identification.

- **[Scripts](./Scripts)** 

    [TestModel](./Scripts/TestModel.py) script which tests the given model with different magnitudes of noise. \
    [TrainModel](./Scripts/TrainModel.py) script that trains the model for x trainings with k folds. Afterwards the best model is saved and can be tested.\
    [VisualizeFilters](./Scripts/VisualizeFilters.py) script which is able to visualize the weights of the loaded model. 

- **[Tools](./Tools/)**

    [CustomLogger](./Tools/CustomLogger.py) a logger for debugging and displaying results. Has custom format and colors for different type of messages (warning, info, error, ...)\
    [SplitTestData](./Tools/SplitTestData.py) script to randomly but evenly split the recordings of all the different test subjects.\
    [Timer](./Tools/Timer.py) a simple class that records the time and returns the elapsed time between two points in the code.\
    [PathFinder](./Tools/PathFinder.py) contains functions that returns a pathLib Path to data, MatlabFunctions and project root directories.

</div>

## Getting Started
Here we will go over the most important scripts: how to use them and what to expect from them.

### FootstepDataset

### TrainModel

### TestModel

### Application (WIP)

## Requirements

</div>
</body>