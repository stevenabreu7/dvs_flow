# DVSFlow: flow cytometer with event-based camera 

## Setup

Create `data` folder and copy the dataset into the folder `data/raw`, with names `A1.raw`, ..., `A4.raw`, `B1.raw`, ..., `B4.raw`. 

```
mkdir data data/raw
```

### Install required libraries

```bash
pip install numpy expelliarmus
```

*TODO: make a pipfile with snnTorch and all other libraries*

## Data processing

### Process data into numpy arrays

```bash
python script_raw_to_numpy_data.py
```

Should take around ~1min. 

### Compress numpy data

Downsampling and low-pass-filtering. With the default parameters, this should result in a ~10x compression, but the script will take quite some time to run. There is multiprocessing support so you can run the script using multiple cores, simply change `N_PROCESSORS`. Keep in mind that each process will require 4GB to 10GB of memory (with 32GB of memory, I was able to run the script using `N_PROCESSORS = 3`).

```bash
python script-compress-numpy-data.py
```

You can then delete all files from the `data/numpy` if you are low on disk space (~40 GB).
