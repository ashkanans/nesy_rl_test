from __future__ import annotations

import torch
from trajectory.models.transformers import GPT


def build_tt_model(args, dataset, vocab_size, device):
    class Cfg:
        pass

    cfg = Cfg()
    cfg.vocab_size = vocab_size
    cfg.block_size = args.block_size
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    cfg.observation_dim = dataset.observation_dim
    cfg.action_dim = dataset.action_dim
    cfg.transition_dim = dataset.joined_dim
    cfg.action_weight = args.action_weight
    cfg.reward_weight = args.reward_weight
    cfg.value_weight = args.value_weight
    cfg.embd_pdrop = args.embd_pdrop
    cfg.resid_pdrop = args.resid_pdrop
    cfg.attn_pdrop = args.attn_pdrop

    model = GPT(cfg).to(device)
    return model
