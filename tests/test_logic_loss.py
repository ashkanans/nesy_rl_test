import torch

from logic_loss_tt import apply_acceptance_floor, compute_sample_weights


def test_apply_acceptance_floor_modes():
    p = torch.tensor([0.0, 1e-12, 0.2], dtype=torch.float32)
    eps = 1e-6

    out_clamp = apply_acceptance_floor(p, mode="clamp", eps=eps)
    assert torch.allclose(out_clamp, torch.tensor([eps, eps, 0.2]))

    out_add = apply_acceptance_floor(p, mode="add", eps=eps)
    assert torch.allclose(out_add, torch.tensor([eps, eps + 1e-12, 0.200001]), atol=1e-8)

    out_none = apply_acceptance_floor(p, mode="none", eps=eps)
    assert torch.allclose(out_none, p)


def test_compute_sample_weights_modes():
    logits = torch.tensor([[0.0, -1.0, -2.0]], dtype=torch.float32)

    w_imp = compute_sample_weights(logits, mode="importance")
    assert w_imp.shape == logits.shape
    assert torch.allclose(w_imp.sum(dim=-1), torch.ones(1))

    w_uni = compute_sample_weights(logits, mode="uniform")
    assert torch.allclose(w_uni, torch.full_like(logits, 1.0 / logits.shape[-1]))
