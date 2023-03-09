import glob
import numpy as np
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


def convert_ev_arr_to_tensor(ev_arr, timestep_us=1):
    """Convert event array (xypt compressed numpy) to tensor of shape (2*32*24, 1000)."""
    # timestep is in us
    n_timesteps = np.ceil(1_000 / timestep_us).astype(int)
    tnsr = np.zeros((2, 32, 24, n_timesteps))
    t_start = ev_arr['t'].min() // 1_000 * 1_000
    assert ev_arr['t'].max() - t_start <= 1_000, 'event array too long'
    tnsr[ev_arr['p'], ev_arr['x'], ev_arr['y'], (ev_arr['t'] - t_start) // timestep_us] = 1
    tnsr = torch.from_numpy(tnsr.reshape(-1, n_timesteps)).float()
    return tnsr


class BinCytometryDataset(Dataset):
    TRAIN_SPLIT = 0.8
    CHARS = "AB"
    FIDXS = list(range(1, 5))

    def __init__(self, data_folder, train=True, test_fi=None, temp_split=False, seed=42, ext='npy',
                 timestep_us=1):
        """Dataset for binary cytometry data. Each sample is a 2D tensor of shape (2, 32, 24, 1000).
        This dataset has two modes:
        1) if test_fi is None, all files are used for training and testing. If temp_split is True,
            the dataset is split into train/test respecting temporal order. If not, randomly.
        2) if test_fi is not None, all files except test_fi are used for training and test_fi is 
            used for testing. temp_split has no effect here.

        Parameters
        ----------
        data_folder : str
            path of dataset root
        train : bool, optional
            train/test flag, by default True
        test_fi: int, optional
            index (1-4) of file to use for testing data (use all others for training), default: None
            if None, load data from all files and then split into train/test
        temp_split : bool, optional
            split dataset into train/test respecting temporal order, by default False
            if False, split dataset into train/test randomly
            !!! only works if test_fi is None !!! # CODEX WROTE THIS
        seed: int, optional
            random seed, by default 42 (used for splitting dataset into train/test)
        ext: str, optional
            file extension in data_folder ('raw' for expelliarmus, 'npy' for numpy), default: 'npy'
        """
        self.data_folder = data_folder
        self.EXT = ext
        self.n_total = len(glob.glob(f'{data_folder}/*.{self.EXT}'))
        self.timestep_us = timestep_us
        if 1000 // self.timestep_us != 1000 / self.timestep_us:
            print('[WARNING] 1000/timestep_us not an integer - last time bin contains fewer events')

        if self.EXT == 'raw':
            print('[WARNING] using expelliarmus is unstable [WARNING]')
            self.wizard = Wizard(encoding='evt2')
            raise NotImplementedError('using expelliarmus is unstable')

        # temporary list of all possible combinations of char and fidx
        chfi_list = [f'{ch}{fi}' for ch in self.CHARS for fi in self.FIDXS]

        ############################
        # if test_fi is given
        if test_fi is not None:
            chfi_to_n_total = {
                chfi: len(glob.glob(f'{data_folder}/{chfi}_*.{self.EXT}'))
                for chfi in chfi_list
            }
            if train:
                self.sample_filenames = [
                    f'{data_folder}/{ch}{fi}_{idx}.{self.EXT}'
                    for ch in self.CHARS for fi in self.FIDXS
                    for idx in range(chfi_to_n_total[f'{ch}{fi}'])
                    if fi != test_fi
                ]
            else:
                # test_filenames = glob.glob(f'{data_folder}/?{test_fi}_*.{self.EXT}')
                self.sample_filenames = [
                    f'{data_folder}/{ch}{fi}_{idx}.{self.EXT}'
                    for ch in self.CHARS for fi in self.FIDXS
                    for idx in range(chfi_to_n_total[f'{ch}{fi}'])
                    if fi == test_fi
                ]
            return

        ############################
        # if test_fi is *not* given
        print('no test_fi given...')

        # map from id (e.g. A1) to index list of dataset
        idx_map = {
            chfi: list(range(len(glob.glob(f'{data_folder}/{chfi}_*.{self.EXT}'))))
            for chfi in chfi_list
        }
        if not temp_split:
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
            self.sample_filenames = [
                all_filenames[chfi][idx] 
                for chfi in chfi_list
                for idx in range(chfi_to_n_train[chfi])
            ]
        else:
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
        return len(self.sample_filenames)

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
        data = convert_ev_arr_to_tensor(ev_arr, self.timestep_us)
        assert data is not None, f'event array is longer than 1000 time steps? {sample_filename}'
        if return_trial:
            return data, label, trial
        else:
            return data, label


def get_datasets(data_folder, seed=42, temp_split=False, run_checks=True):
    """Get train and test datasets."""
    print('getting datasets...', end='\r')
    train_dataset = BinCytometryDataset(
        data_folder=data_folder, 
        train=True, 
        temp_split=temp_split,
        seed=seed
    )
    test_dataset = BinCytometryDataset(
        data_folder=data_folder, 
        train=False, 
        temp_split=temp_split,
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


def get_datasets_fold(data_folder, test_file_idx, run_checks=True, seed=42, timestep_us=1):
    print('getting datasets...', end='\r')
    ds_tr = BinCytometryDataset(
        data_folder=data_folder, 
        train=True, 
        test_fi=test_file_idx, 
        seed=seed,
        timestep_us=timestep_us
    )
    ds_te = BinCytometryDataset(
        data_folder=data_folder, 
        train=False, 
        test_fi=test_file_idx, 
        seed=seed,
        timestep_us=timestep_us
    )

    # check that train/test split is correct
    if run_checks:
        print('running checks...', end='\r')
        for e in ds_tr.sample_filenames:
            assert e not in ds_te.sample_filenames, f'train/test split overlap: {e}'
        test_filenames = glob.glob(f'{data_folder}/?{test_file_idx}_*.npy')
        train_filenames = [e for idx in range(1, 5) if idx != test_file_idx for e in glob.glob(f'{data_folder}/?{idx}_*.npy')]
        assert len(train_filenames) ==  len(ds_tr.sample_filenames)
        assert len(test_filenames) == len(ds_te.sample_filenames)
        assert len(train_filenames) == len(set(train_filenames) & set(ds_tr.sample_filenames))
        assert len(test_filenames) == len(set(test_filenames) & set(ds_te.sample_filenames))
        assert ds_tr.n_total == ds_te.n_total
        ntotal = ds_tr.n_total
        assert len(ds_tr) + len(ds_te) == ntotal
        assert len(ds_tr.sample_filenames) + len(ds_te.sample_filenames) == ntotal
        # assert right number of timestep
        assert ds_tr[0][0].shape[-1] == np.ceil(1000 / timestep_us).astype(int)

    # return datasets
    print('successfully loaded train and test datasets')
    return ds_tr, ds_te
