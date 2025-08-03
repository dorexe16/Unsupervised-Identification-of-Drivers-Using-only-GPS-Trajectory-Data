import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class TripRnnDataset(Dataset):
    def __init__(self, arrays):          # arrays: list of (C × T) numpy
        self.arrays = arrays

    def __len__(self):  return len(self.arrays)

    def __getitem__(self, idx):
        # return Tensor (T, C)  + length
        arr = torch.tensor(self.arrays[idx].T, dtype=torch.float)  # (T,C)
        return arr, arr.shape[0]

def collate(batch):
    seqs, lens = zip(*batch)             # list of tensors (T,C)
    lens   = torch.tensor(lens, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)           # (B, Tmax, C)
    return padded, lens
        # (x, len, y)

# ---------- collate function ----------
def collate(batch):
    """Pads variable-length batch. Works with or without labels."""
    if len(batch[0]) == 3:          # (x,len,y)
        xs, lens, ys = zip(*batch)
        y = torch.stack(ys)
    else:                           # (x,len)
        xs, lens = zip(*batch)
        y = None

    lens = torch.tensor(lens, dtype=torch.long)
    x_padded = pad_sequence(xs, batch_first=True)          # (B,Tmax,C)
    if y is None:
        return x_padded, lens
    return x_padded, lens, y


import torch.nn as nn

class RnnAutoEncoder(nn.Module):
    def __init__(self, n_features=3, hid=128, latent=128, n_layers=2):
        super().__init__()
        self.encoder = nn.GRU(n_features, hid, n_layers,
                              batch_first=True, bidirectional=False)
        self.to_latent = nn.Linear(hid, latent)

        self.latent = latent
        self.decoder_init = nn.Linear(latent, hid)
        self.decoder = nn.GRU(n_features, hid, n_layers,
                              batch_first=True)
        self.out = nn.Linear(hid, n_features)

    def forward(self, x, lens):
        # ---- encode ----
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h_n = self.encoder(packed)         # h_n : (num_layers,B,hid)
        h_last = h_n[-1]                      # (B, hid)
        z = self.to_latent(h_last)            # (B, latent)

        # ---- decode ----  (teacher forcing with inputs=original)
        h0 = self.decoder_init(z).unsqueeze(0).repeat(
                self.decoder.num_layers, 1, 1)      # (n_layers, B, hid)
        packed_dec = pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                          enforce_sorted=False)
        out_packed, _ = self.decoder(packed_dec, h0)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        recon = self.out(out)                 # (B, Tmax, C)
        return recon, z


import torch.nn as nn
import torch

import torch.nn as nn
import torch

class TripEnd2End(nn.Module):
    """
    GRU encoder  →  latent vector  →  light MLP classifier head
    """
    def __init__(self,
                 encoder: 'RnnAutoEncoder' = None,
                 *,
                 n_feat: int = 3,
                 hid: int = 128,
                 latent: int = 128,
                 n_layers: int = 2,
                 n_classes: int = 2,
                 freeze_encoder: bool = False):
        super().__init__()

        # ----- encoder (reuse or build) -----
        if encoder is None:
            self.encoder = nn.GRU(n_feat, hid, n_layers,
                                  batch_first=True, bidirectional=False)
            self.to_lat = nn.Linear(hid, latent)
        else:
            self.encoder = encoder.encoder        # GRU weights
            self.to_lat = encoder.to_latent
            latent = encoder.latent

        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
            for p in self.to_lat.parameters():   p.requires_grad = False

        # ----- *simpler* classifier -----
        self.clf_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(packed)            # (layers, B, hid)
        z = self.to_lat(h_n[-1])                 # (B, latent)
        return self.clf_head(z)                  # logits

