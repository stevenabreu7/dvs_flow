import os
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader

from cytometrybin import get_datasets, BinCytometryNetwork
# from dataset import save_dataset_to_raw_files


DATA_FOLDER = '../data/bin_1ms_comp_ds'
CHARS = 'AB'
FIDXS = range(1, 5)
# save_dataset_to_raw_files('../data/bin_1ms_comp', CHARS, FIDXS, DATA_FOLDER)


if torch.cuda.is_available():
    print(f'CUDA is available with {torch.cuda.device_count()} devices')
    device = torch.device('cuda')
else:
    print('CUDA is not available, falling back to CPU')
    device = torch.device('cpu')


print('loading dataset')
BATCH_SIZE = 64
ds_tr, ds_te = get_datasets(DATA_FOLDER, temporal_split=True, seed=42, run_checks=False)
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

TRAIN_FOLDER = './trained_feb24'
os.makedirs(TRAIN_FOLDER, exist_ok=True)

EPOCHS = 100
for epoch in range(EPOCHS):
    for i, (inp, lab) in enumerate(trainloader):

        out, count = assistant.train(inp, lab)
        header = 'Event rate : ' + ', '.join([f'{c.item():.4f}' for c in count])
        stats.print(epoch, iter=i, header=[header], dataloader=trainloader)
        # inp = inp.to(device)
        # lab = lab.to(device)
        # net.train()
        # out, count = net(inp)
        # loss = error(out, lab)
        # stats.training.num_samples += inp.shape[0]
        # stats.training.loss_sum += loss.cpu().data.item() * out.shape[0]
        # stats.training.correct_samples += torch.sum(classifier(out) == lab).cpu().data.item()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    
    for i, (inp, lab) in enumerate(testloader):

        out, count = assistant.test(inp, lab)
        header = 'Event rate : ' + ', '.join([f'{c.item():.4f}' for c in count])
        stats.print(epoch, iter=i, header=[header], dataloader=testloader)

    torch.save(net.state_dict(), os.path.join(TRAIN_FOLDER, f'net_{epoch}.pt'))
    stats.update()
    stats.save(TRAIN_FOLDER + '/')
    stats.plot(path=TRAIN_FOLDER + '/')
    net.grad_flow(TRAIN_FOLDER + '/')
