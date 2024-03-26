from torch.utils.data import Dataset
from featureExtraction.FeatureExtractor import FeatureExtractor


class FootstepDataset(Dataset):
    def __init__(self, startPath: str, trueLabel: str):
        featureExtractor = FeatureExtractor()
        gen = featureExtractor.extract(startPath)

        # Create an array and store the data as (feature, labelNumeric)
        self.dataArray = []

        # Convert every file into a signal and labels
        while True:
            try:
                # Get all the extracted features and labels in string form
                signal, label = next(gen)

                # If the labels are equal, the label is 1
                labelNum = 1 if label == trueLabel else 0

                # Append the acquired data to the array
                self.dataArray.append([signal, labelNum])
            except StopIteration:
                break

    def __getitem__(self, index):
        row = self.dataArray[index]
        return row[0], row[1]

    def __len__(self):
        return len(self.dataArray)
