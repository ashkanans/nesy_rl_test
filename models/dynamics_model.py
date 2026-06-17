from __future__ import annotations

import torch
import torch.nn as nn


class NeuralDiscreteDynamics(nn.Module):
    """
    Discrete next-state model p(s_next | s, a) learned from offline transitions.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.state_emb = nn.Embedding(self.num_states, self.hidden_dim)
        self.action_emb = nn.Embedding(self.num_actions, self.hidden_dim)

        layers: list[nn.Module] = []
        in_dim = 2 * self.hidden_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.num_states))
        self.mlp = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_h = self.state_emb(states.long())
        action_h = self.action_emb(actions.long())
        x = torch.cat([state_h, action_h], dim=-1)
        return self.mlp(x)

    def predict_next_state_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-6)
        logits = self(states=states, actions=actions)
        return torch.softmax(logits / temp, dim=-1)
