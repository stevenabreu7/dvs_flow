import numpy as np
import matplotlib.pyplot as plt


DPI = 200


def plot_event_count_ax(ax, evts, n_bins=1_000):
    print(f'{evts.shape[0]:,} events')
    up_data = np.histogram(evts[evts['p'] == 1]['t'], bins=n_bins)
    down_data = np.histogram(evts[evts['p'] == 0]['t'], bins=n_bins)
    ax.bar(up_data[1][:-1], up_data[0], width=up_data[1][1]-up_data[1][0])
    ax.bar(down_data[1][:-1], -1*down_data[0], width=down_data[1][1]-down_data[1][0])
    ax.grid()

for ch in "ABCD":
    for fi in range(1, 5):
        print(f'{ch}{fi}...')
        data_path = f'../data/quat_1ms_comp/{ch}{fi}.npy'
        img_path = f'img/event_count_1ms_comp_{ch}{fi}.png'
        delta_t = 1_000
        evts = np.concatenate(np.load(data_path, allow_pickle=True))
        n_bins = (evts['t'].max() - evts['t'].min()) // delta_t

        fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=DPI)
        ax.set_title('event count over time')
        up_data = np.histogram(evts[evts['p'] == 1]['t'], bins=n_bins)
        down_data = np.histogram(evts[evts['p'] == 0]['t'], bins=n_bins)
        ax.bar(up_data[1][:-1], up_data[0], width=up_data[1][1]-up_data[1][0])
        ax.bar(down_data[1][:-1], -1*down_data[0], width=down_data[1][1]-down_data[1][0])
        ax.grid()
        plt.savefig(img_path)
        plt.close()
