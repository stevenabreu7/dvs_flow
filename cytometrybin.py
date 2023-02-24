import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import lava.lib.dl.slayer as slayer
import os
import random
import torch
from expelliarmus import Wizard
from torch.utils.data import Dataset


def save_dataset_to_raw_files(data_folder, chars, fidxs, dst_folder, expelliarmus=False):
    """Save dataset to raw files."""
    os.makedirs(dst_folder, exist_ok=True)
    if expelliarmus:
        wizard = Wizard(encoding='evt2')
    for ch in chars:
        for fi in fidxs:
            print(ch, fi, end='...\r')
            path = f'{data_folder}/{ch}{fi}.npy'
            d = np.load(path, allow_pickle=True)
            # filter out bins with <1k events
            d = d[[e.shape[0] >= 1_000 for e in d]]
            for evarridx, evarr in enumerate(d):
                if evarr['t'].max() - evarr['t'].min() > 1_000:
                    print(f'time too long: {evarr["t"].max() - evarr["t"].min()}, '
                          f'{evarr["t"].max()}, {evarr["t"].min()}')
                if evarr is None:
                    print(f'evarr empty')
                if expelliarmus:
                    new_path = os.path.join(dst_folder, f'{ch}{fi}_{evarridx}.raw')
                    wizard.save(fpath=new_path, arr=evarr)
                else:
                    np.save(os.path.join(dst_folder, f'{ch}{fi}_{evarridx}.npy'), evarr)
    print('done!')


def convert_ev_arr_to_tensor(ev_arr):
    """Convert event array (xypt compressed numpy) to tensor of shape (2*32*24, 1000)."""
    tnsr = np.zeros((2, 32, 24, 1000))
    t_start = ev_arr['t'].min() // 1_000 * 1_000
    assert ev_arr['t'].max() - t_start <= 1_000, 'event array too long'
    tnsr[ev_arr['p'], ev_arr['x'], ev_arr['y'], ev_arr['t'] - t_start] = 1
    tnsr = torch.from_numpy(tnsr.reshape(-1, 1000)).float()
    return tnsr


class BinCytometryDataset(Dataset):
    """Dataset for binary cytometry data. Each sample is a 2D tensor of shape (2, 32, 24, 1000).


    Parameters
    ----------
    data_folder : str
        path of dataset root
    train : bool, optional
        train/test flag, by default True
    temporal_split : bool, optional
        split dataset into train/test respecting temporal order, by default False
        if False, split dataset into train/test randomly
    """

    TRAIN_SPLIT = 0.8
    CHARS = "AB"
    FIDXS = list(range(1, 5))

    def __init__(self, data_folder, train=True, temporal_split=False, seed=42, ext='npy'):
        self.data_folder = data_folder
        self.EXT = ext
        self.n_total = len(glob.glob(f'{data_folder}/*.{self.EXT}'))

        if self.EXT == 'raw':
            self.wizard = Wizard(encoding='evt2')

        # temporary list of all possible combinations of char and fidx
        chfi_list = [f'{ch}{fi}' for ch in self.CHARS for fi in self.FIDXS]

        # map from id (e.g. A1) to index list of dataset
        idx_map = {
            chfi: list(range(len(glob.glob(f'{data_folder}/{chfi}_*.{self.EXT}'))))
            for chfi in chfi_list
        }
        if not temporal_split:
            # shuffle indices
            for chfi in chfi_list:
                random.seed(seed)
                random.shuffle(idx_map[chfi])

        # map from id (e.g. A1) to number of samples in train/test
        chfi_to_n_train = {
            chfi: int(self.TRAIN_SPLIT * len(glob.glob(f'{data_folder}/{chfi}_*.{self.EXT}')))
            for chfi in chfi_list
        }
        chfi_to_n_test = {
            chfi: len(glob.glob(f'{data_folder}/{chfi}_*.{self.EXT}')) - chfi_to_n_train[chfi]
            for chfi in chfi_list
        }

        all_filenames = {
            chfi: [f'{data_folder}/{chfi}_{idx}.{self.EXT}' for idx in idx_map[chfi]]
            for chfi in chfi_list
        }

        if train:
            self.n_samples = sum(chfi_to_n_train.values())
            self.sample_filenames = [
                all_filenames[chfi][idx] 
                for chfi in chfi_list
                for idx in range(chfi_to_n_train[chfi])
            ]
        else:
            self.n_samples = sum(chfi_to_n_test.values())
            self.sample_filenames = [
                all_filenames[chfi][idx] 
                for chfi in chfi_list
                for idx in range(
                    chfi_to_n_train[chfi], 
                    chfi_to_n_train[chfi] + chfi_to_n_test[chfi]
                )
            ]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx, return_trial=False):
        """Returns a sample from the dataset. 

        data: tensor of shape (2, 32, 24, 1000)
        label: int with value 0 or 1
        """
        sample_filename = self.sample_filenames[idx]
        label = int(os.path.basename(sample_filename)[0] == 'B')
        trial = int(os.path.basename(sample_filename)[1])
        if self.EXT == 'raw':
            ev_arr = self.wizard.read(sample_filename)
        else:
            ev_arr = np.load(sample_filename)
        assert ev_arr is not None, f'event array is None: {sample_filename}'
        data = convert_ev_arr_to_tensor(ev_arr)
        assert data is not None, f'event array is longer than 1000 time steps? {sample_filename}'
        if return_trial:
            return data, label, trial
        else:
            return data, label

def get_datasets(data_folder, seed=42, temporal_split=False, run_checks=True):
    """Get train and test datasets."""
    print('getting datasets...', end='\r')
    train_dataset = BinCytometryDataset(
        data_folder=data_folder, 
        train=True, 
        temporal_split=temporal_split,
        seed=seed
    )
    test_dataset = BinCytometryDataset(
        data_folder=data_folder, 
        train=False, 
        temporal_split=temporal_split,
        seed=seed
    )

    print('running checks...', end='\r')
    if run_checks:
        # check that train/test split is correct
        for e in train_dataset.sample_filenames:
            assert e not in test_dataset.sample_filenames, f'train/test split overlap: {e}'
        assert train_dataset.n_total == test_dataset.n_total
        ntotal = train_dataset.n_total
        assert len(train_dataset) + len(test_dataset) == ntotal
        assert len(train_dataset.sample_filenames) + len(test_dataset.sample_filenames) == ntotal
        assert abs(len(train_dataset) - int(ntotal * train_dataset.TRAIN_SPLIT)) < 10
        assert abs(len(test_dataset) - int(ntotal * (1 - train_dataset.TRAIN_SPLIT))) < 10

    # return datasets
    print('successfully loaded train and test datasets')
    return train_dataset, test_dataset


class BinCytometryNetwork(torch.nn.Module):
    def __init__(self):
        super(BinCytometryNetwork, self).__init__()

        neuron_params = {
            'threshold'     : 1.25,
            'current_decay' : 0.25,
            'voltage_decay' : 0.03,
            'tau_grad'      : 0.03,
            'scale_grad'    : 3,
            'requires_grad' : False,
        }
        # neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05)}
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Dense(
                neuron_params_drop, 2*32*24, 512,
                weight_norm=True, delay=True
            ),
            slayer.block.cuba.Dense(
                neuron_params_drop, 512, 512,
                weight_norm=True, delay=True
            ),
            slayer.block.cuba.Dense(
                neuron_params, 512, 2,
                weight_norm=True
            ),
        ])

    def forward(self, spike):
        """Forward pass of the network.

        Args
        spike: input spike tensor of shape (B, P, W, H, T)
        
        Returns
        spike: output spike tensor of shape (B, C, T)
        count: output count tensor of shape (1, n_layers)
        """
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
        count = torch.FloatTensor(count).flatten().to(spike.device)
        return spike, count

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        """export network to hdf5 format."""
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
