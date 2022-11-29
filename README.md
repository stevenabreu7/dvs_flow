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
python script-expelliarmus-data.py
```

Should take around ~1min. 
