"""Simple feed-forward snnTorch network for DVS flow project."""
import torch
from torch import nn
import snntorch as snn
from snntorch import utils


class CytometerNetwork(nn.Module):
    """Spiking neural network for cytometer data."""
    def __init__(self, n_inputs, n_hidden, n_outputs, threshold, beta, spk_grad, debug=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, output=True, spike_grad=spk_grad)
        self.linear2 = nn.Linear(n_hidden, n_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, output=True, spike_grad=spk_grad)
        self.debug_mode = debug
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        with torch.no_grad():
            # make weights positive
            self.linear1.weight = nn.Parameter(self.linear1.weight.detach().cpu().abs())
            self.linear2.weight = nn.Parameter(self.linear2.weight.detach().cpu().abs())

    def forward(self, x_batch):
        """Forward pass. Returns spike and membrane voltage outputs."""
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        utils.reset(self)

        # Record final layer
        spk1_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []

        for step in range(x_batch.size(1)):
            cur1 = self.linear1(self.flatten(x_batch[:, step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            if self.debug_mode:
                spk1_rec.append(spk1)
                mem1_rec.append(mem1)
            cur2 = self.linear2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        if self.debug_mode:
            return (
                torch.stack(spk1_rec, dim=1), torch.stack(mem1_rec, dim=1),
                torch.stack(spk2_rec, dim=1), torch.stack(mem2_rec, dim=1)
            )
        return torch.stack(spk2_rec, dim=1), torch.stack(mem2_rec, dim=1)
