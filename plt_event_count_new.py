import numpy as np
import matplotlib.pyplot as plt
import os


DPI = 200


def plot_event_count_ax(ax, evts, n_bins=1_000):
    print(f'{evts.shape[0]:,} events')
    up_data = np.histogram(evts[evts['p'] == 1]['t'], bins=n_bins)
    down_data = np.histogram(evts[evts['p'] == 0]['t'], bins=n_bins)
    ax.bar(up_data[1][:-1], up_data[0], width=up_data[1][1]-up_data[1][0])
    ax.bar(down_data[1][:-1], -1*down_data[0], width=down_data[1][1]-down_data[1][0])
    ax.grid()


delta_t = 1_000
chars = list("ABCD")
removed_per_particle = {ch: 0. for ch in chars}
for i, ch in enumerate(chars):
    print(ch)
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(17,2), dpi=DPI, sharex=True, sharey=True)
    img_path = f'img/event_count_new_1ms_{ch}_shxy.png'
    if os.path.exists(img_path):
        continue
    for j, fidx in enumerate(range(1, 5)):
        axs[j].set_title(f'trial {fidx}')
        # axs[j].set_ylim(-250_000, 250_000)
        if j == 0:
            axs[j].set_ylabel(f'particle {ch}')
        evts = np.concatenate(np.load(f'../data/bin_1ms/{ch}{fidx}.npy', allow_pickle=True))
        # evts = evts[:evts.shape[0]//2]
        nbins = (evts['t'].max() - evts['t'].min()) // delta_t
        plot_event_count_ax(axs[j], evts, n_bins=nbins)
        del evts
    # save and close plot
    fig.tight_layout()
    fig.savefig(img_path, dpi=DPI)
    plt.clf()
    plt.close(fig)
