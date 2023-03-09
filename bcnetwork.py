import h5py
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import re
import torch


class BinCytometryNetwork(torch.nn.Module):
    def __init__(self, layers:str):
        """Initialize the network.

        Args
        layers: string describing the hidden layers in the network
                '512d-512d' -> two layers with 512 neurons each, with delays
                '512-512'   -> two layers with 512 neurons each, no delays
        """
        super(BinCytometryNetwork, self).__init__()

        neuron_params = {
            'threshold'     : 1.25,
            'current_decay' : 0.25,
            'voltage_decay' : 0.03,
            'tau_grad'      : 0.03,
            'scale_grad'    : 3,
            'requires_grad' : True,
        }

        # parse layer string
        delays = ['d' in l for l in layers.split('-')] + [False]
        layer_dims = [2*32*24, *[int(re.findall(r'\d+', l)[0]) for l in layers.split('-')], 2]

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Dense(
                neuron_params, layer_dims[i], layer_dims[i+1],
                weight_norm=True, delay=delays[i]
            ) for i in range(len(delays))
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

        with open(path + 'gradFlow.txt', 'a') as f:
            f.write(",".join(list(map(str, grad))) + '\n')

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
    
    def __str__(self) -> str:
        x = super().__str__()
        x = '\n'.join([e for e in x.splitlines() if e.strip() != ')'])
        return x
