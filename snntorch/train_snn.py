"""
Steven Abreu, 2022

Train spiking neural network with one hidden layer to do binary classification on event-based data.

Arguments:
    name: name of the run (used as folder name for cache, model checkpoints, other logs).

Optional arguments:
    -d, --debug (bool): run in debug mode (saves membrane and spike values for all layers).
    --maxsamples (int): max number of samples to load from each file (default: None -> load all).
    --checkpoint (str): path to checkpoint file to load network parameters from.

Notes:
- data loading time comparison: simple 1.6s / cached 1.4s / batch-cached 0.2s

TODO:
- replace torch.swapaxes with torch.transpose
- when loading a model checkpoint, change epoch and batch accordingly
- add validation data
"""
import argparse
import os
import shutil
import time
from functools import reduce
import numpy as np
import tonic
from tonic import DiskCachedDataset
import torch
from torch.utils.data import DataLoader
from snntorch import functional as SF
from snntorch import surrogate
from data import CytometerDataset, SENSOR_SIZE
from network import CytometerNetwork


##############################
# parse arguments, run checks
##############################

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('files')
parser.add_argument('--checkpoint')
parser.add_argument('--maxsamples')
parser.add_argument('--epochs', type=int)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-g', '--gpu', action='store_true')
args = parser.parse_args()
print('starting run:', args.name)

# assert maxsamples is None, int, or float (will raise exception if not)
if args.maxsamples is not None:
    if args.maxsamples.isdecimal():
        args.maxsamples = int(args.maxsamples)
    else:
        args.maxsamples = float(args.maxsamples)

# get train file idxs
tr_fidxs = list(map(int, list(args.files)))
print('files:', tr_fidxs)

# assert base path is not taken already
BASE_PATH = f'models/{args.name}'
if os.path.exists(BASE_PATH) and len(os.listdir(BASE_PATH)) > 0:
    input('model name exists already. press any key to continue')
os.makedirs(BASE_PATH, exist_ok=True)

# copy files to model folder (for logging)
shutil.copy('train_snn.py', f'{BASE_PATH}/train_snn.py')
shutil.copy('network.py', f'{BASE_PATH}/network.py')
shutil.copy('data.py', f'{BASE_PATH}/data.py')

# assert checkpoint exists, if given
if args.checkpoint is not None:
    print(args.checkpoint)
    assert os.path.exists(args.checkpoint), 'checkpoint file not found'

##############################
# data loading
##############################

tr_fstr = reduce(lambda x,y: x+y, map(str, tr_fidxs))
trainset = CytometerDataset(file_idxs=tr_fidxs, max_samples=args.maxsamples, time_window=1)
cached_trainset = DiskCachedDataset(trainset, cache_path=f'./cache/{args.name}_train{tr_fstr}')
print(f'loading train files {tr_fstr} with {len(trainset)} samples')

# batched dataloaders
BATCH_SIZE = 64 if args.gpu else 512
padding = tonic.collation.PadTensors()
trainloader = DataLoader(cached_trainset, batch_size=BATCH_SIZE, collate_fn=padding, shuffle=True)

##############################
# setup network
##############################

# determine torch device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
print('running on', device)

# set up network with hyperparameters
net_params = {
    'n_inputs': reduce(lambda x, y: x*y, SENSOR_SIZE),
    'n_hidden': 100,
    'n_outputs': 20,
    'threshold': 0.5,
    'beta': 0.9,
    'spk_grad': surrogate.fast_sigmoid(slope=75),
    'debug': args.debug
}
net = CytometerNetwork(**net_params).to(device)

# load state dict, if given
if args.checkpoint:
    print('loading state dict:', args.checkpoint)
    net.load_state_dict(torch.load(args.checkpoint))

##############################
# setup training
##############################

# optimizer and loss function
IS_POPULATION = True
N_CLASSES = 2
optimizer = torch.optim.Adam(
    net.parameters(), lr=2e-2, betas=(0.9, 0.999)
)
lossf = SF.mse_count_loss(
    correct_rate=0.8, incorrect_rate=0.2, population_code=IS_POPULATION, num_classes=N_CLASSES
)

# training loop
N_EPOCHS = args.epochs
loss_hist = []
acc_hist = []
t_0 = time.time()
print('loading data...')
for epoch in range(N_EPOCHS):
    for bidx, (data, targets) in enumerate(iter(trainloader)):
        # continue
        t_start = time.time()
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        every_n_batches = 100 if BATCH_SIZE < 100 else 10
        if args.debug and (bidx == 0 or (bidx+1) % every_n_batches == 0):
            # if debug, save all data for one of five batches (and the first five batches)
            spk1, mem1, spk_out, mem2 = net(data)
            np.save(f'{BASE_PATH}/spk1_e{epoch+1}_b{bidx+1}.npy', spk1.cpu().detach().numpy())
            np.save(f'{BASE_PATH}/mem1_e{epoch+1}_b{bidx+1}.npy', mem1.cpu().detach().numpy())
            np.save(f'{BASE_PATH}/spk2_e{epoch+1}_b{bidx+1}.npy', spk_out.cpu().detach().numpy())
            np.save(f'{BASE_PATH}/mem2_e{epoch+1}_b{bidx+1}.npy', mem2.cpu().detach().numpy())
            if epoch == 0:
                np.save(f'{BASE_PATH}/targets_{bidx+1}.npy', targets.cpu().detach().numpy())
            # np.save(f'{BASE_PATH}/realidxs_e{epoch+1}_b{bidx+1}.npy', real_idxs.detach().numpy())
            del mem1, spk1, mem2
        elif args.debug:
            _, _, spk_out, _ = net(data)
        else:
            spk_out, _ = net(data)
        loss = lossf(torch.swapaxes(spk_out, 0, 1), targets)

        # gradient calculation + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy + log
        acc = SF.accuracy_rate(torch.swapaxes(spk_out, 0, 1), targets,
                               population_code=IS_POPULATION, num_classes=N_CLASSES)
        t_elapsed = time.time() - t_start
        t_total = time.time()-t_0
        print(f"Epoch {epoch+1}, batch {bidx+1:2}/{len(trainloader)} ({t_elapsed:4.1f}s):", end=" ")
        print(f"loss {loss.item():6.2f}, accuracy {acc * 100:6.2f}% [{t_total:4.1f}s]")

        # Store loss history for future plotting
        loss_hist.append(loss.item())
        acc_hist.append(acc)

        # store model and state dict
        if args.checkpoint is None:
            torch.save(net.state_dict(), f'{BASE_PATH}/SD_e{epoch+1}_b{bidx+1}.pt')
            np.save(f'{BASE_PATH}/loss.npy', np.array(loss_hist))
            np.save(f'{BASE_PATH}/acc.npy', np.array(acc_hist))
