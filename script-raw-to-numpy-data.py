from expelliarmus import Wizard
import numpy as np
import os


os.makedirs('./data/numpy', exist_ok=True)

for ch in "AB":
  for idx in range(1, 5):
    wizard = Wizard(encoding="evt2", fpath=f"./data/raw/{ch}{idx}.raw")
    arr = wizard.read()
    print(f"{ch}{idx}:", arr.shape)
    np.save(f"./data/numpy/{ch}{idx}.npy", arr)
