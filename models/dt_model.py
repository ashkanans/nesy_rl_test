from __future__ import annotations

import torch
import torch.nn as nn


class DecisionTransformerDiscrete(nn.Module):
    """
    Lightweight discrete DT baseline:
    predicts actions from (state, previous action, RTG, timestep) with causal attention.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        context_len: int,
        n_embd: int = 128,
        n_layer: int = 2,
        n_head: int = 2,
        dropout: float = 0.1,
        max_timestep: int = 1024,
    ):
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.context_len = int(context_len)
        self.max_timestep = int(max_timestep)

        self.state_emb = nn.Embedding(self.num_states, n_embd)
        self.prev_action_emb = nn.Embedding(self.num_actions + 1, n_embd)
        self.rtg_emb = nn.Linear(1, n_embd)
        self.time_emb = nn.Embedding(self.max_timestep, n_embd)
        self.in_ln = nn.LayerNorm(n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.head = nn.Linear(n_embd, self.num_actions)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        states: torch.Tensor,
        prev_actions: torch.Tensor,
        rtg: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps = timesteps.clamp(min=0, max=self.max_timestep - 1)
        x = (
            self.state_emb(states)
            + self.prev_action_emb(prev_actions)
            + self.rtg_emb(rtg.unsqueeze(-1))
            + self.time_emb(timesteps)
        )
        x = self.in_ln(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        causal_mask = self._causal_mask(x.size(1), x.device)
        h = self.encoder(x, mask=causal_mask)
        return self.head(h)
