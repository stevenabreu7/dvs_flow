"""
Steven Abreu, 2022.

Run this script to convert the .raw data from the event-based camera into
numpy arrays. Reads A1.raw, ..., A4.raw, B1.raw, ..., B4.raw files from the
./data/raw/ directory and stores the numpy arrays into the ./data/numpy/
directory with the same filenames, but .npy file extension.
"""
import os
from expelliarmus import Wizard
import numpy as np


os.makedirs('./data/numpy', exist_ok=True)

for ch in "AB":
    for idx in range(1, 5):
        wizard = Wizard(encoding="evt2", fpath=f"./data/raw/{ch}{idx}.raw")
        arr = wizard.read()
        print(f"{ch}{idx}:", arr.shape)
        np.save(f"./data/numpy/{ch}{idx}.npy", arr)
