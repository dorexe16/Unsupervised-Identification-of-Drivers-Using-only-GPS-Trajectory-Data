# -*- coding: utf-8 -*-
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
from src.arnet_unsup.dataset import GPSWindowDataset
from src.arnet_unsup.model   import ARNetUnsup
from src.preprocessing.pre_process_deep_learning import load_and_merge_data
from src.utils.Drives import drives
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKDIR = Path("checkpoints")
CHECKDIR.mkdir(parents=True, exist_ok=True)
# ---------- 1. build dataset ----------
data_dir = r'C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data_backup'
def main():
    merged, _ = load_and_merge_data(data_dir=data_dir)     # uses your helper

    ds   = GPSWindowDataset(merged)

    def collate(batch):
        L = max(b.shape[1] for b in batch)
        x = torch.zeros(len(batch), L, 35)
        mask = torch.zeros(len(batch), L, 1)  # 1 = real timestep
        for i, w in enumerate(batch):
            T = w.shape[1]
            x[i, :T] = w.T
            mask[i, :T] = 1.0
        return x.to(device), mask.to(device)

    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=0, collate_fn=collate)

    # ---------- 2. model ----------
    model = ARNetUnsup().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)


    # ---------- helper: random time-mask augmentation ----------
    class ViewAug:
        def __init__(self, mask_ratio=0.15):
            self.r = mask_ratio

        def __call__(self, x):  # x: B×L×35  (tensor)
            B, L, C = x.shape
            m = torch.rand(B, L, 1, device=x.device) > self.r
            return x * m  # zero-mask 15 % rows

    augment = ViewAug(0.15)

    # ---------- training loop with InfoNCE ----------
    τ = 0.5  # temperature
    β_l1 = 1e-3*2
    for epoch in range(80):
        mse_sum, nce_sum, n_win = 0., 0., 0
        for xb, mask in loader:
            opt.zero_grad()
            # two stochastic views
            v1 = augment(xb)
            v2 = augment(xb)

            _, p1, x1, z1 = model(v1)  # proj emb + recon
            _, p2, x2, z2 = model(v2)

            # reconstruction loss on *one* view (optional)
            mse_loss = ((x1 - v1).pow(2) * mask).sum() / mask.sum()

            # ------- InfoNCE -------
            B = p1.size(0)
            logits = (p1 @ p2.t()) / τ  # B×B
            labels = torch.arange(B, device=xb.device)
            nce_loss = F.cross_entropy(logits, labels)

            l1_loss = β_l1 * z1.abs().mean()
            loss = mse_loss  + l1_loss + nce_loss
            loss.backward()
            opt.step()

            mse_sum += mse_loss.item() * B
            nce_sum += nce_loss.item() * B
            n_win += B
        sched.step()
        print(f"E{epoch:02d}  MSE={mse_sum / n_win:.3f}  NCE={nce_sum / n_win:.3f}")

        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/arnet_unsup_e.pth")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()          # no-op on Linux; required on Windows
    main()