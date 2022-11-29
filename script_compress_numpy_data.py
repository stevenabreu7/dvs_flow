"""
Steven Abreu, 2022.

Run this script to compress the numpy arrays from ./data/numpy through spatial
downsampling and temporal low-pass-filtering, storing the resulting arrays in
./data/compressed.
"""
from dataclasses import dataclass
import multiprocessing
import os
from expelliarmus import Wizard
import numpy as np


# to disable multiprocessing, set N_PROCESSORS to 1
N_PROCESSORS = 3
SENSOR_SIZE = (640, 480, 2,)


@dataclass
class Downsample:
    """Copied from tonic.transforms.Downsample. Removed events.copy() for
    (possibly) better memory efficiency."""

    spatial_factor: float = 1

    def __call__(self, events):
        if "x" in events.dtype.names:
            events["x"] = events["x"] * self.spatial_factor
        if "y" in events.dtype.names:
            events["y"] = events["y"] * self.spatial_factor
        return events


@dataclass
class LowPassLIF:
    """Low pass filter through simple LIF neuron model."""

    weight: float = 1.0
    vrest: float = 0.0
    vthr: float = 3.0
    leak: float = 0.9
    trefr: int = 2

    sensor_size: tuple = (32, 24, 2)

    def __call__(self, events):
        print("start event sorting")
        map_time_to_evidxs = {}
        for idx, evt in enumerate(events):
            if N_PROCESSORS == 1:
                if idx % int(1e5) == 0:
                    print(f"\r{idx/len(events):.2%}", end=" "*10)
            if evt['t'] in map_time_to_evidxs:
                map_time_to_evidxs[evt['t']].append(idx)
            else:
                map_time_to_evidxs[evt['t']] = [idx]
        print("finished event sorting")

        print("start LIF processing")
        membrane = np.zeros(self.sensor_size, dtype=np.float32)
        event_times = np.unique(events['t'])
        last_updated_t = 0
        events_lpf = []
        refr = {}
        for idx, evtt in enumerate(event_times):
            # log progress
            if N_PROCESSORS == 1:
                if idx % 100 == 0:
                    print(f"\r{idx/len(event_times):.2%}", end=" "*10)
            # clean up old refractory periods
            for del_key in [tk for tk in refr if tk < (evtt-self.trefr-1)]:
                del refr[del_key]
            # update membrane potentials
            membrane = (self.leak ** (evtt - last_updated_t)) * membrane
            # iterate over events at current timestep to check for resulting spikes
            for ev_idx in map_time_to_evidxs[evtt]:
                evt = events[ev_idx]
                (_,evtx,evty,evtp) = evt
                if evt in refr.get(evtt, []):
                    # ignore events if in refractory period
                    continue
                membrane[evtx,evty,evtp] += self.weight
                if membrane[evtx,evty,evtp] >= self.vthr:
                    # if spike, then add to events_lpf
                    membrane[evtx,evty,evtp] = self.vrest
                    events_lpf.append(evt)
                    # add refractory events
                    for i in range(self.trefr+1):
                        refr[i] = refr.get(i, []) + [(evtx,evty,evtp,evtt+i)]
        print('finished LIF processing')
        return np.array(events_lpf, dtype=events.dtype)


def load_transform_data(filename):
    """Load and transform data from given filename."""
    wizard = Wizard(encoding="evt2", fpath=filename)
    n_chunks = len(wizard.read())
    print(f'{filename}: {n_chunks} chunks')

    ds_factor = 20
    ds_trf = Downsample(spatial_factor=1/ds_factor)
    lp_trf = LowPassLIF(sensor_size=(SENSOR_SIZE[0]//ds_factor, SENSOR_SIZE[1]//ds_factor, 2))

    wizard = Wizard(encoding="evt2", fpath=filename)
    all_evs = lp_trf(ds_trf(wizard.read()))
    print(f'transformed {filename}:', all_evs.shape)

    # save to new directory
    np.save(filename.replace('/raw/', '/compressed/').replace('.raw', '.npy'), all_evs)


if __name__ == "__main__":
    print(f"running with {N_PROCESSORS} processors")

    filenames = [f"./data/raw/{ch}{idx}.raw" for ch in "AB" for idx in range(1,5)]
    os.makedirs('./data/compressed/', exist_ok=True)

    if N_PROCESSORS > 1:
        # multiprocessing
        for bidx in range(0, len(filenames), N_PROCESSORS):
            processes = []
            for fname in [filenames[i] for i in range(bidx, bidx+N_PROCESSORS)]:
                # create and start processes
                kwargs = {'filename': fname}
                process = multiprocessing.Process(target=load_transform_data, kwargs=kwargs)
                processes.append((process, fname))
                process.start()
                print('started process for', fname)
            for process, fname in processes:
                # wait for processes to end
                process.join()
                print('ended process for', fname)
    else:
        # simple sequential processing
        for fname in filenames:
            load_transform_data(fname)
