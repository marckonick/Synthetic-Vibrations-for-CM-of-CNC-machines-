#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:26:55 2025

@author: nikola.markovic
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torch.nn.utils import spectral_norm
import math

# --------------------------
# Sinusoidal timestep embedding
# --------------------------
def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal embeddings (same as in DDPM / Transformer).
    timesteps: (B,)
    Returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


# --------------------------
# Basic conv block
# --------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # conditioning: project cond vector to (out_ch,)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        
        self.act = nn.SiLU()

    def forward(self, x, cond_vec):
        """
        x: (B, C, L)
        cond_vec: (B, cond_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        # inject conditioning as bias
        cond = self.cond_proj(cond_vec).unsqueeze(2)  # (B, C, 1) (32, 160) -> (32, 64, 1)
        h = h + cond # INFORMACIJA cond SE DODAJE SVAKOM KANALU 
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h + cond
        h = self.act(h)
        return h


# --------------------------
# U-Net for 1D signals
# --------------------------
class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        time_emb_dim=128,
        num_ops=10,
        op_emb_dim=32,
    ):
        super().__init__()

        # Embeddings
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * time_emb_dim, time_emb_dim)
        )

        self.op_emb = nn.Embedding(num_ops, op_emb_dim)

        cond_dim = time_emb_dim + op_emb_dim
        C = base_channels
        
        # ---- Encoder ----
        self.down1 = ConvBlock(in_channels, base_channels, cond_dim)
        self.down2 = ConvBlock(base_channels, base_channels * 2, cond_dim)
        self.down3 = ConvBlock(base_channels * 2, base_channels * 4, cond_dim)
        self.down4 = ConvBlock(base_channels * 4, base_channels * 8, cond_dim)

        self.pool = nn.MaxPool1d(2)

        # ----- bottleneck -----
        self.bottleneck = ConvBlock(8 * C, 8 * C, cond_dim)   # 8C -> 8C

        # ----- decoder -----
        self.up4  = nn.ConvTranspose1d(8 * C, 4 * C, 4, 2, 1)  # 8C -> 4C
        self.dec4 = ConvBlock(12 * C, 4 * C, cond_dim)         # (4C + 8C) -> 4C

        self.up3  = nn.ConvTranspose1d(4 * C, 2 * C, 4, 2, 1)  # 4C -> 2C
        self.dec3 = ConvBlock(6 * C, 2 * C, cond_dim)          # (2C + 4C) -> 2C

        self.up2  = nn.ConvTranspose1d(2 * C, C, 4, 2, 1)      # 2C -> C
        self.dec2 = ConvBlock(3 * C, C, cond_dim)              # (C + 2C) -> C

        self.up1  = nn.ConvTranspose1d(C, C, 4, 2, 1)          # C -> C
        self.dec1 = ConvBlock(2 * C, C, cond_dim)              # (C + C) -> C

        self.final_conv = nn.Conv1d(C, in_channels, kernel_size=3, padding=1)

    # --------------------------
    # Forward pass
    # --------------------------
    def forward(self, x, t, op_id):
        """
        x: (B,1,L) noisy sample
        t: (B,) timesteps
        op_id: (B,) operation indices
        label: (B,) class labels
        """

        # --- Embed timestep ---
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # --- Condition embeddings ---
        op_vec = self.op_emb(op_id)

        cond_vec = torch.cat([t_emb, op_vec], dim=1)

        # ---- Encoder ----
        d1 = self.down1(x, cond_vec)   # (B, C, L)
        p1 = self.pool(d1)

        d2 = self.down2(p1, cond_vec)
        p2 = self.pool(d2)

        d3 = self.down3(p2, cond_vec)
        p3 = self.pool(d3)

        d4 = self.down4(p3, cond_vec) # batch_size, 512, 512 | -> | 512 - broj kanala, 512 - sekvenca 
        p4 = self.pool(d4)            # batch_size, 512, 256 | -> | 512 - broj kanala, 256 - sekvenca 

        # ---- Bottleneck ----
        b = self.bottleneck(p4, cond_vec) # dve konvolucije 512 -> 512 kanala  

        # ---- Decoder ----
        u4 = self.up4(b) # 32, 512, 256 -> 32, 256, 512
        u4 = torch.cat([u4, d4], dim=1) # 32, 256+512, 512
        u4 = self.dec4(u4, cond_vec)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3, cond_vec)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2, cond_vec)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1, cond_vec) # 32 x 64 x 4096

        # ---- Output ----
        eps = self.final_conv(u1)
        return eps


    
    def number_of_params(self):
         print('Number of network paramaters:')
         print(sum(p.numel() for p in self.parameters()))
        
        
        
        
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, L)
        return self.act(self.bn(self.conv(x)))
    
    
    
class SimpleTimeSeriesDiffusionModel(nn.Module):
    """
    Small model for DDPM-style ε-prediction on 1D time series.
    Input:  x_t  shape (B, 1, 256)
            t    shape (B,) integer timestep in [0, T-1]
    Output: ε̂   shape (B, 1, 256)
    """
    def __init__(
        self,
        T: int,                 # total diffusion steps (for time embedding)
        in_channels: int = 32,
        hidden_channels: int = 64,
        time_emb_dim: int = 32,
        gru_hidden: int = 128,
    ):
        super().__init__()

        self.T = T

        # --- Time embedding: simple embedding + linear ---
        self.time_embed = nn.Embedding(T, time_emb_dim)
        self.time_proj  = nn.Linear(time_emb_dim, hidden_channels)

        # --- Initial convolution to lift channels ---
        self.conv_in = ConvBlock1D(in_channels, hidden_channels)

        # --- A couple of conv blocks for local patterns ---
        self.conv1 = ConvBlock1D(hidden_channels, hidden_channels)
        self.conv2 = ConvBlock1D(hidden_channels, hidden_channels)

        # --- GRU for longer-range temporal dependencies ---
        # GRU input: hidden_channels, output: 2*gru_hidden (bidirectional)
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=False,     # GRU expects (seq_len, batch, feat)
            bidirectional=True,
        )
        self.gru_out_proj = nn.Linear(2 * gru_hidden, hidden_channels)

        # --- Final convs to get back to 1 channel ---
        self.conv3 = ConvBlock1D(hidden_channels, hidden_channels)
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 1, 256)
        t:   (B,) integer timestep
        returns ε̂: (B, 1, 256)
        """
        B, C, L = x_t.shape
        assert C == 32 and L == 128, "Expected input shape (B, 1, 256)"

        # --- Time embedding ---
        # t: (B,) -> (B, time_emb_dim) -> (B, hidden_channels)
        t_emb = self.time_embed(t.clamp(0, self.T - 1))  # safety clamp
        t_emb = torch.relu(self.time_proj(t_emb))        # (B, hidden_channels)

        # --- Local conv feature extractor ---
        h = self.conv_in(x_t)        # (B, hidden_channels, 256)
        h = self.conv1(h)            # (B, hidden_channels, 256)
        h = self.conv2(h)            # (B, hidden_channels, 256)

        # --- Add time embedding (broadcast over length) ---
        # t_emb: (B, hidden_channels) → (B, hidden_channels, 1)
        h = h + t_emb[:, :, None]

        # --- GRU over time dimension ---
        # GRU expects (L, B, F)
        h_seq = h.permute(2, 0, 1)      # (256, B, hidden_channels)
        h_seq, _ = self.gru(h_seq)      # (256, B, 2*gru_hidden)
        h_seq = self.gru_out_proj(h_seq)  # (256, B, hidden_channels)
        h = h_seq.permute(1, 2, 0)      # back to (B, hidden_channels, 256)

        # --- Final conv layers ---
        h = self.conv3(h)               # (B, hidden_channels, 256)
        eps_pred = self.conv_out(h)     # (B, 1, 256)

        return eps_pred
    
    def number_of_params(self):
         print('Number of network paramaters:')
         print(sum(p.numel() for p in self.parameters()))
        

