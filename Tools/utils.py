from pathlib import Path

"""
Simple functions that returns the path of certain 
"""
def getProjectRoot() -> Path:
    return Path(__file__).parent.parent


def getDataRoot() -> Path:
    return getProjectRoot().joinpath("data")


def getFunctionPath() -> Path:
    return getProjectRoot().joinpath("FeatureExtraction").joinpath("MatlabFunctions")


if __name__ == '__main__':
    print(getFunctionPath())
    print(getProjectRoot())
    print(getDataRoot())
