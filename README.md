# DVS Flow: flow cytometry with event-based camera and neuromorphic hardware

Code for the paper "Flow cytometry with event-based vision and spiking neuromorphic hardware" at CVPR 2023 Workshop on Event-based Vision ([workshop link](https://tub-rip.github.io/eventvision2023), [conference link](https://cvpr2023.thecvf.com)).

For reference, the code for the paper "Training a spiking neural network on an event-based label-free flow cytometry dataset" at NICE 2023 ([link](https://arxiv.org/abs/2303.10632)) is provided in the [NICE23](./NICE23/) folder.

## Setup

Download and unpack the data from [DOI:10.5281/zenodo.7804830](http://doi.org/10.5281/zenodo.7804830). Place it into the `data` folder.

Use metavision to read the `.raw` files, then convert them to numpy files. For less disk space usage, use [expelliarmus](https://github.com/open-neuromorphic/expelliarmus) to save the data into `.raw` files again (as of early 2023, expelliarmus did not read the original raw files correctly - therefore make sure to read them with metavision first). See also [script_raw_to_numpy_data.py](script_raw_to_numpy_data.py).

Install the required libraries with conda with the provided `.yml` file.
```bash
conda env create -f environment.yml
```

## Data processing

Process the data using [script_compress_data.py](./script_compress_data.py) (downsampling and low-pass-filtering). With the default parameters, this should result in a ~10x compression, but the script will take quite some time to run. There is multiprocessing support so you can run the script using multiple cores, simply change `N_PROCESSORS`. Keep in mind that each process will require 4GB to 10GB of memory (with 32GB of memory, I was able to run the script using `N_PROCESSORS = 3` in ~1h40min).

## Data exploration

Some scripts and notebooks to plot the data are provided in the [plots](./plots/) folder.

## Training models on frames

The following files train models on frame representations of the data: 
- [train_frames_bin_lin.ipynb](./train_frames_bin_lin.ipynb): linear model
- [train_frames_bin_fnn.py](./train_frames_bin_fnn.py): MLP 
- [train_frames_bin_cnn.py](./train_frames_bin_cnn.py): CNN
See the files for performance results.

## Training spiking neural networks

The training script is [train_slayer.py](./train_slayer.py). It uses the dataset as defined in [bcdataset.py](./bcdataset.py) and the network as defined in [bdnetwork.py](bdnetwork.py). 

## Running on Loihi

The notebook [loihi_preparation.ipynb](./loihi_preparation.ipynb) loads the trained SNN and converts it to a format that is understandable by Loihi. The notebook [loihi_inference.ipynb](./loihi_inference.ipynb) is then used to run the SNN on Loihi. 

## Results

The results of this study can be found in the [results](./results) folder. 
