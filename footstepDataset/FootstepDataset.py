import pandas as pd
from torch.utils.data import Dataset

from featureExtraction.FeatureExtractor import FeatureExtractor


class FootstepDataset(Dataset):
    def __init__(self, startPath: str):
        featureExtractor = FeatureExtractor()
        gen = featureExtractor.extract(startPath)

        # Create an array and later convert it to dataframe for easier recollection of data.
        # This is the fastest way according to this stackoverflow post: https://stackoverflow.com/questions/10715965/create-a-pandas-dataframe-by-appending-one-row-at-a-time/17496530#17496530
        dataArray = []

        while True:
            try:
                signal, label = next(gen)
                dataArray.append([signal, label])
            except StopIteration:
                break
        self.dataframe = pd.DataFrame(dataArray, columns=["signal", "label"])

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return row["signal"], row["label"]

    def __len__(self):
        return len(self.dataframe)
