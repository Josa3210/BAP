<head>
    <link rel="stylesheet" href="./style.css">
</head>
<body>


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
- **[Application](./Application):**
    
    [Application](./Application/Application.py) class that combines the AudioRecorder, FeatureExtractor and a trained model from NeuralNetwork to make predictions on live recorded audio.

- **[FeatureExtraction](./FeatureExtraction):**
    - [FeatureCacher](./FeatureExtraction/FeatureCacher.py) that caches the converted signal and loads them back in.\
    - [FeatureExtractor](./FeatureExtraction/FeatureExtractor.py) class to filter recordings and transform them into input for the neural networks. \
    - [MatlabFunctions](./FeatureExtraction/MatlabFunctions/) that holds the different functions used by the FeatureExtractor. \
    - [TestFiltering](./FeatureExtraction/TestFiltering.py) to test the different functions of FeatureExtractor.\
    - [Transforms](./FeatureExtraction/Transforms.py) class that can be passed on as parameter to the FootstepDataset to add a transformation to the filtered signals before being handled by the FeatureExtractor class.

- **[FootstepDataset](./FootstepDataset):**

    - [AudioRecorder](./FootstepDataset/AudioRecorder.py) class to record sound and save it in .wav files.\
    - [FootstepDataset](./FeatureExtraction/FootstepDataset.py) class to read in a series of .wav files and converts it to a Dataset from the Pytorch package.
    

- **[NeuralNetwork](./NeuralNetwork):**

    - [EarlyStopper](./NeuralNetwork/EarlyStopper.py) class that will stop the training loop of a model once a certain criteria is met.\
    - [InterfaceNN](./NeuralNetwork/InterfaceNN.py) interface class from where all other neural networks are derived. Contains also TrainableNN class that holds functions for optimizing, training and testing.\
    - [NeuralNetworks](./NeuralNetwork/NeuralNetworks.py) class that contains the different architectures created for person identification.

- **[Scripts](./Scripts):** 

    - [TestModel](./Scripts/TestModel.py) script which tests the given model with different magnitudes of noise. \
    - [TrainModel](./Scripts/TrainModel.py) script that trains the model for x trainings with k folds. Afterwards the best model is saved and can be tested.\
    - [VisualizeFilters](./Scripts/VisualizeFilters.py) script which is able to visualize the weights of the loaded model. 

- **[Tools](./Tools/):**

    - [CustomLogger](./Tools/CustomLogger.py) a logger for debugging and displaying results. Has custom format and colors for different type of messages (warning, info, error, ...)\
    - [SplitTestData](./Tools/SplitTestData.py) script to randomly but evenly split the recordings of all the different test subjects.\
    - [Timer](./Tools/Timer.py) a simple class that records the time and returns the elapsed time between two points in the code.\
    - [PathFinder](./Tools/PathFinder.py) contains functions that returns a pathLib Path to data, MatlabFunctions and project root directories.

</div>

<div class=GS>

## Getting Started
Here we will go over the most important scripts: how to use them and what to expect from them.

### FootstepDataset
This class wil (recursively) convert a directory filled with .wav files into a PyTorch Dataset object. This object can be used with the PyTorch library (including DataLoaders, Modules, ...).
The FootstepDataset objects needs/can have the following arguments:
- dataSource (Path): Path to the data source directory containing audio files.
- fExtractor (FeatureExtractor, optional): Feature extractor. Defaults to None.
- transformer (Transforms, optional): Data transformation. Defaults to None.
- cachePath (Path, optional): Path for caching features. Defaults to None.
- labelFilter (list[str], optional): Filter for specific labels. Defaults to None.
- addNoiseFactor (float, optional): Noise factor. Defaults to 0.

In order for the object to properly label the different files, the names of the audio files should be in the form of "label_nr.wav". This way, the samples will be labeled with the word before the first underscore. Additionally, by specifying labels in the *labelfilter*, the object will only convert the files which will have these labels.

Another major component of the FootstepDataset is the FeatureExtractor. This is an initial transformation of the audio files. These use the FeatureExtractor interface and can have a _filter_ and _transform_ function. This makes the object adaptable by reusing the same _filter_ function for cleaning up the sound whilst allowing different kind of transformation for different objects using the _transform_ function. When no FeatureExtractor object is passed on, the standard _Filter_ object will be used which only filters the sound.

On top of initially transforming the sound, the object also can perform transformations after filtering using the _Transformer_ interface. These will transform the filtered sound before the FeatureExtractor's _transform_ function. The _Transform_ object will do this n amount of times, which will increase the size of the returned dataset by a factor of n.

Also a _cachePath_ can be provided. Here the object will read if there is already a filtered version of the sound. If not, it will create one and save it in the _cachePAth_ for future use. This drastically reduces the time for loading in a dataset.

Lastly, a *addNoiseFactor* can be added to the object. This will add Gaussian noise with a standard devciation of *addNoiseFactor* to the signal **before filtering and transforming**. This can be used to simulate a noisy environment and test the robustness of a Module/FeatureExtractor against noise.

In the end a FootstepDataset object will be returned. These samples will be normalized against the largest value in the whole dataset. This value can be found under the _maxVal_ variable in the dataset.


### TrainModel
This script is used for training a TrainableNN (see [NeuralNetworks](./NeuralNetwork/NeuralNetworks.py)). Before starting the training, these objects need to be assigned:
- trainingDataset (FootstepDataset): the dataset on which will be trained.
- network (TrainableNN): the network that will be trained.
- earlyStopper (EarlyStopper): if no earlystopper must be used, set it to None.

Following, the parameters of the training must be chosen:
- nTrainings: the amount of trainings that will happen. 
- batchSize: the size of the batches that will be passed on to the DataLoader.
- learningRate: the learning rate can be manually set, or obtained using hyperparameter tuning
- folds: how many fold per training will be used.
- epochs: how many epochs will happen in one fold (assuming no earlystopper is assigned).
Additionally, an _id_ can be provided for better saving the results of the training. 

After providing these parameters, running the program will start training the network.
During the training loop, every fold the following parameters are calculated and saved:
- averageLoss (training and validation)
- average accuracy
- confusion matrix

From these, the best are kept for future comparisson.
After every training (x folds), the model with the best accuracy will be saved under "*networkName*-BestResult.pth". At the end of all the trainings (nTrainings), the best model will be saved under "*networkName*-BestFromBatch-*id*.pth.
This saved file contains a dictionary with the stateDict of the best model and the normilazation factor of the dataset. These are the two values needed for effectively using the trained model in applications.

Also an overview of the different trainings will be printed, the time taken for training and the confusion matrix of the best training will be shown.

### TestModel
This script will test a network against a test dataset and decrease the SNR after every testloop to eventually map the robustness against noise.

Before running this script, the following parameters must be assigned:
- testDataset (FootstepDataset): the dataset containing test samples.
- network (InterfaceNN): the network that will be tested.
- batchsize: the size of the batches that will be passed on to the DataLoader.
- noiseFactors (list[int]): these values will be passed on to the dataset _addNoiseFactor_ variable which mimics adding Gaussian noise with a specific std.

After every test, the following results are calculated and saved:
- The running loss
- The accuracy, precision and recall
- The average SNR of the samples
- The confusion matrix

After all the tests are done, a plot will be created mapping the loss and accuracy against the average SNR to provide a full view of the robustness of the network. The confusion matrices are saved on the device under *Figures/ConfMat_test_{network.name}_SNR{round(avgSNR * 100)}.png*

### Application (WIP)
This application is designed for real time prediction of people passing by. It will record live a 4 second sample and try to identify the person passing by. All the recordings will be saved under "\application\savefiles".

**Setup:**  
The application will need a live recording, so a microphone must be connected to the laptop. Additionally, the application must be provided with the following arguments:
- model (InterfaceNN): a network architecture that will be used for identification (standard: NeuralNetworkSTFT). 
- modelPath: a path to the trained model which will be used for identification.
- sampleRate: the samplerate used for the recording of the sounds (standard 44.1kHz).

After running the script, it will ask to record a nosieProfile. This is the recordign of the environmental/background noise which will be used for filtering the live recordings using spectral subtraction. 

**Use:**  
To use the application, simply follow the CLI and enter "Y" when a person is passing by the microhpone. The following steps will be taken by the application:
- Record the sound.
- Replay the sound.
- Ask a filename in order to save the recorded file.
- Transform the recording and try to predict the person passing by.

This will happen in an infinite loop. To exit the application, type "exit".

## Requirements
The following packages are required:
- PyTorch (and prerequisites)
- MatPlotLib
- bayesian-optimization
- ignite
- matlab, matlabengine
- psutil
- scipy
- scikit-learn, scikit-optimize
- sounddevice


</div>
</body>