import torch,torch.nn as nn
# ─── src/arnet_unsup/model.py ─────────────────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F
class ARNetUnsup(nn.Module):
    def __init__(self, latent_dim=32, embed_dim=128, p_drop=0.3):
        super().__init__()
        # ─── Encoder ───────────────────────────────────────────
        self.gru1 = nn.GRU(35, embed_dim // 2, batch_first=True,
                           bidirectional=True)
        self.gru2 = nn.GRU(embed_dim, embed_dim // 2, batch_first=True,
                           bidirectional=True)

        # two independent dropouts: encoder & latent
        self.drop      = nn.Dropout(p_drop)       # used on encoder output
        self.lat_drop  = nn.Dropout(p_drop)       # used on z_seq

        self.time_fc   = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU())

        self.bottleneck = nn.Linear(embed_dim, latent_dim)

        # ─── Decoder ───────────────────────────────────────────
        self.decoder_fc  = nn.Linear(latent_dim, embed_dim)
        self.decoder_gru = nn.GRU(embed_dim, 35, num_layers=2,
                                  batch_first=True)
        self.out         = nn.Identity()          # residual + identity

        # ─── Projection head for InfoNCE ───────────────────────
        self.proj = nn.Sequential(
            nn.Linear(latent_dim*3, 128),
            nn.GELU(),
            nn.Linear(128, 128))

    def forward(self, x, lengths: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x        : (B, L, 35)  padded batch of statistics grids
        lengths  : optional 1-D tensor (B,) with the true sequence lengths.
                   If None, plain padded sequences are used.

        Returns
        -------
        trip_emb : (B, 3·latent_dim)   — mean ∥ max ∥ first-4-mean of z_seq
        proj_emb : (B, 128)            — ℓ2-normalised projection for InfoNCE
        xhat     : (B, L, 35)          — reconstruction of x  (residual link)
        z_seq    : (B, L, latent_dim)  — timestep-wise latent for sparsity loss
        """

        # ─── Encoder (optionally packed) ───────────────────────────
        if lengths is not None:
            # safety assertions
            assert lengths.dim() == 1 and lengths.size(0) == x.size(0), "lengths must be (B,)"
            assert (lengths > 0).all(), "zero-length trip!"
            assert (lengths <= x.size(1)).all(), "len[i] exceeds padded width"

            # pack on CPU to avoid CUDA warning
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.gru1(packed)
            h, _ = self.gru2(h)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        else:
            h, _ = self.gru1(x)
            h, _ = self.gru2(h)

        # ─── Bottleneck ────────────────────────────────────────────
        h = self.drop(h)  # B × L × embed_dim
        e_seq = self.time_fc(h)  # B × L × embed_dim
        z_seq = self.bottleneck(e_seq)  # B × L × latent_dim
        z_seq = self.lat_drop(z_seq)  # (optional dropout)

        # ─── Decoder & residual reconstruction ─────────────────────
        recon_in = self.decoder_fc(z_seq)  # B × L × embed_dim
        h_dec, _ = self.decoder_gru(recon_in)  # B × L × 35
        xhat = h_dec + x  # residual connection

        # ─── Trip-level embeddings  ────────────────────────────────
        trip_emb = torch.cat([
            z_seq.mean(dim=1),
            z_seq.max(dim=1).values,
            z_seq[:, :4].mean(dim=1)  # first-few frames
        ], dim=-1)  # 3 × latent_dim

        # projection for InfoNCE (128-D, ℓ2-normalised)
        proj_emb = nn.functional.normalize(self.proj(trip_emb), dim=-1)

        return trip_emb, proj_emb, xhat, z_seq

    # -------------------------------------------------------------
    def loss(self, x, xhat, z_seq, sparsity_w=1e-3):
        recon = F.mse_loss(xhat, x, reduction="mean")
        sparse = sparsity_w * z_seq.abs().mean()
        return recon + sparse, {"recon": recon.item(), "sparse": sparse.item()}
