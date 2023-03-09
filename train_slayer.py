"""Train a network on the DVS dataset using SLAYER.

Args:
    fold (int): fold to use for testing
    network (str): network description
    timestep_us (int): timestep in microseconds, default = 1

feb24, 1ms timestep, with delay, 2 layers
    python train_slayer.py 1 512d-512d IDX
mar1,  1ms timestep, no delay,   2 layers
    python train_slayer.py 1 512-512 IDX
mar8,  1ms timestep, with delay, 1 layer
    python train_slayer.py 1 512d IDX

once the current run is finished: 
python train_slayer.py 10 512d 1
python train_slayer.py 10 512d 2
python train_slayer.py 10 512d 3
python train_slayer.py 10 512d 4
------
python train_slayer.py 100 512d 1 & python train_slayer.py 100 512d 2
python train_slayer.py 100 512d 3 & python train_slayer.py 100 512d 4
"""
import argparse
import os
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader

from bcnetwork import BinCytometryNetwork
from bcdataset import get_datasets_fold


# check if CUDA is available
if torch.cuda.is_available():
    print(f'CUDA is available with {torch.cuda.device_count()} devices')
    device = torch.device('cuda')
else:
    print('CUDA is not available, falling back to CPU')
    device = torch.device('cpu')

DATA_FOLDER = '../data/bin_1ms_comp_ds'
CHARS = 'AB'
FIDXS = range(1, 5)
BATCH_SIZE = 64
EPOCHS = 50

# parse arguments
parser = argparse.ArgumentParser(description='Train a network on the DVS dataset')
parser.add_argument('timestep_us', type=int, help='timestep in microseconds')
parser.add_argument('network', type=str, help='network description')
parser.add_argument('fold', type=int, help='fold to use for testing')
args = parser.parse_args()
# arg: fold to use for testing
test_file_idx = args.fold
assert test_file_idx in range(1, 5), f'fold must be in range 1-4, got {test_file_idx}'
print(f'using fold {test_file_idx} for testing')
# arg: timestep in microseconds
timestep_us = args.timestep_us
print(f'using timestep: {timestep_us}us')
# arg: network layers
layers = args.network
net = BinCytometryNetwork(layers=layers).to(device)
print(net)

train_folder = f'./logs/mar9_{layers}_dt{timestep_us}us/fold{test_file_idx}/'
os.makedirs(os.path.join(train_folder, 'checkpoints'), exist_ok=True)

print('loading dataset')
ds_tr, ds_te = get_datasets_fold(DATA_FOLDER, test_file_idx, run_checks=True, seed=42, 
                                 timestep_us=timestep_us)
trainloader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

error = slayer.loss.SpikeRate(
    true_rate=0.2, false_rate=0.03, reduction='sum'
).to(device)

stats = slayer.utils.LearningStats()
classifier = slayer.classifier.Rate.predict
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

assistant = slayer.utils.Assistant(
    net, error, optimizer, stats, classifier=classifier, count_log=True
)

for epoch in range(EPOCHS):
    for i, (inp, lab) in enumerate(trainloader):
        out, count = assistant.train(inp, lab)
        header = 'Event rate : ' + ', '.join([f'{c.item():.4f}' for c in count])
        stats.print(epoch, iter=i, header=[header], dataloader=trainloader)

    for i, (inp, lab) in enumerate(testloader):
        out, count = assistant.test(inp, lab)
        header = 'Event rate : ' + ', '.join([f'{c.item():.4f}' for c in count])
        stats.print(epoch, iter=i, header=[header], dataloader=testloader)

    torch.save(net.state_dict(), os.path.join(train_folder, 'checkpoints', f'net_{epoch}.pt'))
    stats.update()
    stats.save(train_folder)
    stats.plot(path=train_folder)
    net.grad_flow(train_folder)
