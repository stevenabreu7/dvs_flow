import numpy as np
import os
from expelliarmus import Wizard
from metavision_core.event_io import RawReader


MAX_EVENTS = 1_000_000_000


def read_file_raw(fpath):
    rawreader = RawReader(fpath, max_events=MAX_EVENTS)
    while not rawreader.done:
        rawreader.seek_event(MAX_EVENTS // 100)
    n_events = rawreader.current_event_index()
    rawreader.reset()
    events = rawreader.load_n_events(n_events)
    assert events.shape[0] < MAX_EVENTS, 'too many events in file'
    return events


fnames = [f'{ch}{idx}.raw' for ch in 'AB' for idx in range(1, 5)]

metavision_dir = '../../data/raw/'
expelliarmus_dir = '../data/raw_exp/'

# for fname in fnames:
for fname in ['B3.raw', 'B4.raw']:
    print(f'[{fname}] loading file', end=' '*10)
    wiz = Wizard(encoding='evt2')
    exp = wiz.read(expelliarmus_dir + fname)
    met = read_file_raw(metavision_dir + fname)

    if np.all([np.array_equal(met[c], exp[c]) for c in 'xypt']):
        print(f'\r[{fname}] success' + ' '*10)
    else:
        print(f'\r[{fname}] FAILED' + ' '*10)
