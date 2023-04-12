"""
Steven Abreu, 2022

Test a trained SNN (from `train_snn.py`).
"""
import argparse
import os
import time
from functools import reduce
import numpy as np
import tonic
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
parser.add_argument('checkpoint')
parser.add_argument('fileidx', type=int)
parser.add_argument('--maxsamples', type=int)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-g', '--gpu', action='store_true')
args = parser.parse_args()
print('starting testing for model:', args.checkpoint)
BASE_PATH = '/'.join(args.checkpoint.split('/')[:-1])
print('base_path:', BASE_PATH)

# assert checkpoint exists, if given
if args.checkpoint is not None:
    print(args.checkpoint)
    assert os.path.exists(args.checkpoint), 'checkpoint file not found'

CHCKPT_ID = args.checkpoint.replace('SD_', '').replace('.pt', '')

##############################
# data loading
##############################

te_fidxs = [args.fileidx]
te_fstr = reduce(lambda x,y: x+y, map(str, te_fidxs))
testset = CytometerDataset(file_idxs=te_fidxs, max_samples=args.maxsamples, time_window=1)
print(f'loading test files {te_fstr} with {len(testset)} samples')

# batched dataloaders
BATCH_SIZE = 64 if args.gpu else 512
padding = tonic.collation.PadTensors()
testloader = DataLoader(testset, batch_size=BATCH_SIZE, collate_fn=padding, shuffle=True)

##############################
# setup network
##############################

# determine torch device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
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

# load state dict
print('loading state dict:', args.checkpoint)
net.load_state_dict(torch.load(args.checkpoint, map_location=device))

##############################
# setup training
##############################

# optimizer and loss function
IS_POPULATION = True
N_CLASSES = 2
lossf = SF.mse_count_loss(
    correct_rate=0.8, incorrect_rate=0.2, population_code=IS_POPULATION, num_classes=N_CLASSES
)

# training loop
loss_hist = []
acc_hist = []
t_0 = time.time()
print('loading data...')
for bidx, (data_cpu, targets_cpu) in enumerate(iter(testloader)):
    # continue
    t_start = time.time()
    data = data_cpu.to(device)
    targets = targets_cpu.to(device)

    # forward pass
    net.eval()
    if args.debug and (bidx == 0 or (bidx+1) % 5 == 0):
        # if debug, save all data for one of five batches (and the first five batches)
        spk1, mem1, spk_out, mem2 = net(data)
        np.save(f'{BASE_PATH}/tst_spk1_b{bidx+1}.npy', spk1.detach().numpy())
        np.save(f'{BASE_PATH}/tst_mem1_b{bidx+1}.npy', mem1.detach().numpy())
        np.save(f'{BASE_PATH}/tst_spk2_b{bidx+1}.npy', spk_out.detach().numpy())
        np.save(f'{BASE_PATH}/tst_mem2_b{bidx+1}.npy', mem2.detach().numpy())
        np.save(f'{BASE_PATH}/tst_targets_{bidx+1}.npy', targets.detach().numpy())
        # np.save(f'{BASE_PATH}/realidxs_e{epoch+1}_b{bidx+1}.npy', real_idxs.detach().numpy())
        del mem1, spk1, mem2
    else:
        spk_out, _ = net(data)
    # print(data.size(), targets.size(), spk_out.size())
    # print(data.get_device(), targets.get_device(), spk_out.get_device())
    # print(data.device, targets.device, spk_out.device)
    # print(next(net.parameters()).device)
    loss = lossf(torch.swapaxes(spk_out, 0, 1).to(device), targets.to(device))

    # accuracy + log
    acc = SF.accuracy_rate(torch.swapaxes(spk_out, 0, 1), targets,
                            population_code=IS_POPULATION, num_classes=N_CLASSES)
    t_elapsed = time.time() - t_start
    t_total = time.time()-t_0
    print(f"Batch {bidx+1:2}/{len(testloader)} ({t_elapsed:4.1f}s):", end=" ")
    print(f"loss {loss.item():6.2f}, accuracy {acc * 100:6.2f}% [{t_total:4.1f}s]")

    # Store loss history for future plotting
    loss_hist.append(loss.item())
    acc_hist.append(acc)

    # store performance metrics
    np.save(f'{CHCKPT_ID}_test_loss.npy', np.array(loss_hist))
    np.save(f'{CHCKPT_ID}_test_acc.npy', np.array(acc_hist))
