import glob
import shutil
import random
from Tools import PathFinder

"""
This function splits the recorded files into 2 parts: test files and training files
This is done by randomly choosing 20% of the files as test files and the rest as training files
Finally, it will copy all the files to their appropriate directory
Afterwards the function should not be called upon again
"""
if __name__ == '__main__':
    # Get path where the different recordings are
    startPath = PathFinder.getDataRoot().joinpath("recordings")
    searchPath = str(startPath) + r"\**"

    # Create the different destination paths
    testPath = PathFinder.getDataRoot().joinpath("testData")
    trainingPath = PathFinder.getDataRoot().joinpath("trainingData")

    # Check all the directories containing gait recordings
    dirNames = [fn for fn in glob.glob(pathname=searchPath, recursive=False) if not fn.__contains__("noiseProfile")]
    for dir in dirNames:
        searchStr = dir + r"\*.wav"  # String to find all .wav files
        files = glob.glob(pathname=searchStr)  # Search all .wav files

        # Get a random 20% of the files of each person as test files
        testS = round(len(files) * 0.2)
        testFiles = random.sample(files, testS)

        # Training files are the ones not in testFiles
        trainingFiles = [f for f in files if f not in testFiles]

        # Check if all the files are used for each person
        print(len(trainingFiles))
        print(len(testFiles))
        print(len(files))

        # Copy the test files to the testData folder
        for file in testFiles:
            name = file.split("\\")[-1]
            shutil.copy(file, str(testPath.joinpath(name)))

        # Copy the training files to the trainingData folder
        for file in trainingFiles:
            name = file.split("\\")[-1]
            shutil.copy(file, str(trainingPath.joinpath(name)))
