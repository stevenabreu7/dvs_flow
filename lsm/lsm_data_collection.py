import argparse
import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.source import RingBuffer as InputRingBuffer
from lava.proc.io.sink import RingBuffer as OutputRingBuffer


def forward_lsm(inp_data):
    """Forward pass of LSM.
    
    Input: inp_data (n_neurons, n_steps)
    Output: spike rates of LSM (n_neurons,)
    """
    n_neurons = inp_data.shape[0]
    n_steps = inp_data.shape[1]
    win_sparsity = 0.95
    w_sparsity = 0.99
    win_max = 0.5
    w_max = 0.05

    win = np.random.rand(n_neurons, n_neurons) * win_max
    win[np.random.rand(n_neurons, n_neurons) < win_sparsity] = 0.
    w = np.random.rand(n_neurons, n_neurons) * w_max
    w[np.random.rand(n_neurons, n_neurons) < w_sparsity] = 0.

    source = InputRingBuffer(data=inp_data)
    lsm = LIF(shape=(n_neurons,), vth=5., dv=0.1, du=0.1, bias_mant=0.)
    win_dense = Dense(weights=win)
    w_dense = Dense(weights=w)
    sink = OutputRingBuffer(shape=(n_neurons,), buffer=n_steps)

    source.s_out.connect(win_dense.s_in)
    win_dense.a_out.connect(lsm.a_in)
    lsm.s_out.connect(w_dense.s_in)
    w_dense.a_out.connect(lsm.a_in)
    lsm.s_out.connect(sink.a_in)

    run_condition = RunSteps(num_steps=n_steps)
    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    lsm.run(condition=run_condition, run_cfg=run_cfg)
    lsm_spikes = sink.data.get()
    return lsm_spikes # .sum(axis=1)

import time
import os

N_processes = 8
N_per_process = 15

def forward_batch(samples, idxs):
    t_start = time.time()
    for n, (sample, idx) in enumerate(zip(samples, idxs)):
        path = f'data_lsm/lsm_spikes_5k_{idx}.npy'
        if not os.path.exists(path):
            inp_data = sample.reshape(100, -1).T
            s_out = forward_lsm(inp_data)
            np.save(path, s_out)
            dur = time.time() - t_start
            print(f'{n:>2} {idx:>4} {dur:6.1f}s {s_out.mean():6.2f} +- {s_out.std():6.2f}')

from multiprocessing import Process
import glob
import sys

if __name__ == '__main__':
    idxs = list(
        set(range(5000)) - 
        set([int(e.split('_')[-1].strip('.npy')) for e in glob.glob('data_lsm/lsm_spikes_5k_*.npy')])
    )
    data = np.load('data_lsm/data_5k.npy')

    processes = []
    for idx in range(N_processes):
        cur_idxs = idxs[idx*N_per_process : (idx+1)*N_per_process]
        processes.append(Process(target=forward_batch, args=[data[cur_idxs], cur_idxs]))

    for process in processes:
        process.start()
    time.sleep(60)
    sys.exit(0)
