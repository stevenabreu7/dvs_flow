import argparse
import os
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader

from cytometrybin import BinCytometryNetwork, get_datasets_fold


# parse argument: fold to use for testing
parser = argparse.ArgumentParser(description='Train a network on the DVS dataset')
parser.add_argument('fold', type=int, help='fold to use for testing')
TEST_FILE_IDX = parser.parse_args().fold
assert TEST_FILE_IDX in range(1, 5), f'fold must be in range 1-4, got {TEST_FILE_IDX}'
print(f'using fold {TEST_FILE_IDX} for testing')


DATA_FOLDER = '../data/bin_1ms_comp_ds'
CHARS = 'AB'
FIDXS = range(1, 5)
BATCH_SIZE = 64
EPOCHS = 50

if torch.cuda.is_available():
    print(f'CUDA is available with {torch.cuda.device_count()} devices')
    device = torch.device('cuda')
else:
    print('CUDA is not available, falling back to CPU')
    device = torch.device('cpu')

print('loading dataset')
ds_tr, ds_te = get_datasets_fold(DATA_FOLDER, TEST_FILE_IDX, run_checks=True, seed=42)
trainloader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

net = BinCytometryNetwork().to(device)

error = slayer.loss.SpikeRate(
    true_rate=0.2, false_rate=0.03, reduction='sum'
).to(device)

stats = slayer.utils.LearningStats()
classifier = slayer.classifier.Rate.predict
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

assistant = slayer.utils.Assistant(
    net, error, optimizer, stats, classifier=classifier, count_log=True
)

TRAIN_FOLDER = f'./trained_fold{TEST_FILE_IDX}_feb24'
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(os.path.join(TRAIN_FOLDER, 'checkpoints'), exist_ok=True)

for epoch in range(EPOCHS):
    for i, (inp, lab) in enumerate(trainloader):
        out, count = assistant.train(inp, lab)
        header = 'Event rate : ' + ', '.join([f'{c.item():.4f}' for c in count])
        stats.print(epoch, iter=i, header=[header], dataloader=trainloader)

    for i, (inp, lab) in enumerate(testloader):
        out, count = assistant.test(inp, lab)
        header = 'Event rate : ' + ', '.join([f'{c.item():.4f}' for c in count])
        stats.print(epoch, iter=i, header=[header], dataloader=testloader)

    torch.save(net.state_dict(), os.path.join(TRAIN_FOLDER, 'checkpoints', f'net_{epoch}.pt'))
    stats.update()
    stats.save(TRAIN_FOLDER + '/')
    stats.plot(path=TRAIN_FOLDER + '/')
    net.grad_flow(TRAIN_FOLDER + '/')
