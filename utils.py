from pathlib import Path


def getProjectRoot() -> Path:
    return Path(__file__).parent


def getDataRoot() -> Path:
    return Path(__file__).parent.joinpath("data")


def getFunctionPath() -> Path:
    return Path(__file__).parent.joinpath("featureExtraction").joinpath("matlabFunctions")


if __name__ == '__main__':
    print(getFunctionPath())
    print(getProjectRoot())
    print(getDataRoot())
