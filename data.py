"""
Steven Abreu, 2022

Dataset implementation for DVS Cytometer data.
"""
from functools import reduce
from torch.utils.data import Dataset
import numpy as np
import tonic


ORIG_SENSOR_SIZE = (640, 480, 2)
SENSOR_SIZE = (32, 24, 2)
BASE_PATH = './data/compressed'
OUTPUT_A = 0
OUTPUT_B = 1


class CytometerDataset(Dataset):
    """
    Dataset implementation for DVS Cytometer data.

    Currently assumes that each files stores and array of event arrays (where each event
    array contains events belonging to a single sample). TODO: make this compatible with 
    the expelliarmus library and its data loading functions.

    Parameters:
        time_window (int): time window length for one frame (same unit as event timestamps).
        max_samples (int): how many samples to take from each file (if None, take all).
    """
    def __init__(self, file_idxs=range(1,5), time_window=1, max_samples:int=None):
        super().__init__()

        paths = [f"{BASE_PATH}/{ab}{i}.npy" for i in file_idxs for ab in "AB"]
        self.time_window = time_window

        # each event contains events from a 1e4us=10ms
        self.events = np.concatenate([np.load(p, allow_pickle=True)[:max_samples] for p in paths])

        # create labels for all events (based on filename)
        labels = [OUTPUT_A if "A" in p else OUTPUT_B for p in paths]
        lengths = [max_samples or len(np.load(p, allow_pickle=True)) for p in paths]
        self.labels = reduce(lambda x,y: x+y, [[labels[i]] * lengths[i] for i in range(len(paths))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # transform to frame
        transform = tonic.transforms.ToFrame(sensor_size=SENSOR_SIZE, time_window=self.time_window)
        return transform(self.events[idx]), self.labels[idx], idx
