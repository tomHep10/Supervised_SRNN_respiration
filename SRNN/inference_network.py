import numpy as np
import torch
import torch.nn as nn
import math
np.random.seed(131)
torch.manual_seed(131)

# Inference network
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Input/Output: [B, T, D]
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

# Inference network
class RNNInfer(nn.Module):
    def __init__(self,
        input_shape: int,
        hidden_shape: int,
        n_layers: int = 2,
        n_heads: int = 1,
        ff_mult: int = 4,
        dropout: float = 0.0,
        pre_norm: bool = True,
                ):
        super(RNNInfer, self).__init__()
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        # We use bidirectional RNN in the inference network, if using forward-rnn only (e.g., for causal inference), please delete backward_rnn
        self.in_proj = nn.Linear(self.input_shape, self.hidden_shape)

        self.pos_enc = PositionalEncoding(d_model=self.hidden_shape, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_shape,
            nhead=n_heads,
            dim_feedforward=ff_mult * self.hidden_shape,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # so we use [B, T, D]
            norm_first=pre_norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mean_proj = nn.Sequential(
            nn.Linear(self.hidden_shape, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_shape),
        )



    def forward(self, y_train, key_padding_mask=None):
        
        batches = int(y_train.shape[0])
        dtype = torch.float32

        device = y_train.device
        B, T, _ = y_train.shape

        # --- Transformer encoder produces "bi_output" equivalent: [B, T, hidden_shape] ---
        x = self.in_proj(y_train)          # [B, T, H]
        x = self.pos_enc(x)                # [B, T, H]
        bi_output = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mean_out = self.mean_proj(bi_output)  # [B, T, H]
 
        # --- Reparameterization trick (vectorised over all timesteps) ---
        covariance_matrix = (1e-4) * torch.eye(self.hidden_shape, device=device)
        std = covariance_matrix.diag()  # [H]
        ep = torch.randn_like(mean_out)  # [B, T, H]
        sampled_h = mean_out + ep * std  # reparameterization trick in order to compute gradients
 
        infer_dist = torch.distributions.multivariate_normal.MultivariateNormal(mean_out, covariance_matrix) # we need this distribution to compute the entropy in our loss
        return infer_dist, sampled_h, mean_out


