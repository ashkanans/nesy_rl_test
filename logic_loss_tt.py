import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def compute_sample_weights(log_prob_traces, mode):
    """
    Build per-sample weights for acceptance aggregation.

    Args:
        log_prob_traces: [B, K] log-likelihood-like scores
        mode: one of {"importance", "uniform"}
    """
    if mode == "importance":
        return F.softmax(log_prob_traces, dim=-1)
    if mode == "uniform":
        return torch.full_like(log_prob_traces, 1.0 / log_prob_traces.shape[-1])
    raise ValueError(f"Unknown sample weighting mode: {mode}")


def apply_acceptance_floor(prob_acceptance, mode, eps):
    """
    Stabilize acceptance probabilities before log.

    Modes:
      - clamp: max(p, eps)
      - add:   p + eps
      - none:  p
    """
    if mode == "clamp":
        if eps is None or eps <= 0.0:
            return prob_acceptance
        return prob_acceptance.clamp(min=eps)
    if mode == "add":
        if eps is None or eps <= 0.0:
            return prob_acceptance
        return prob_acceptance + eps
    if mode == "none":
        return prob_acceptance
    raise ValueError(f"Unknown acceptance floor mode: {mode}")


class LogicLossModule:
    """
    Logic-aware loss for Trajectory Transformer with Deep DFA constraints.

    This module combines:
        - a standard supervised loss (e.g. next-token cross-entropy from the model)
        - a global logic loss derived from a Deep DFA that checks LTL constraints
          over whole generated sequences.

    The core idea:
        1) The model produces logits over token IDs for each position in the sequence.
        2) We draw num_samples differentiable trajectories from these logits using
           Gumbel-Softmax, obtaining soft one-hot token distributions.
        3) The adapter maps token distributions to DFA symbol distributions.
        4) The Deep DFA processes these symbol distributions and returns, for each
           sampled trace, an acceptance probability.
        5) We use a Monte-Carlo / importance-weighted estimator to approximate
           P_theta(trace satisfies constraint), and define:

               logic_loss = -log( E_{samples} [ acceptance ] )

        6) The final training loss keeps supervised imitation active and adds
           logic as an auxiliary regularizer:

               total_loss = supervised_loss + alpha * logic_loss

    This provides a differentiable way to inject global LTL constraints into
    sequence model training.
    """

    def __init__(
        self,
        deep_dfa,
        adapter,
        mode="global",
        num_samples=10,
        temperature=0.5,
        alpha=0.4,
        append_end_symbol=False,
        eps=1e-10,
        clamp_acceptance=True,
        acceptance_floor_mode=None,
        sample_weighting="importance",
        logic_state_only=False,
    ):
        """
        Initialize the logic loss module.

        Args:
            deep_dfa:
                DeepDFA instance or list of DeepDFAs used to compute acceptance
                probabilities for soft symbol sequences.
            adapter:
                TTDFAAdapter (or compatible) that maps token probabilities to
                symbol probabilities for the DFA.
            mode:
                "global" for full-sequence logic loss (implemented),
                "local" reserved for token-level constraints (not implemented).
            num_samples:
                number of Gumbel-Softmax samples per sequence.
            temperature:
                Gumbel-Softmax temperature; lower = sharper (closer to hard argmax),
                higher = softer distributions.
            alpha:
                Logic regularization weight. Supervised loss is always kept active;
                alpha only scales the auxiliary logic loss.
            eps:
                epsilon used when clamping acceptance probabilities before
                applying the logarithm; if clamp_acceptance is False or eps <= 0,
                no clamping is applied.
            clamp_acceptance:
                if True, clamp acceptance probabilities from below by eps before
                taking the log; if False, log(0) is allowed and may produce -inf.
            acceptance_floor_mode:
                explicit floor strategy {"clamp","add","none"}.
                If None, inferred from clamp_acceptance for backward compatibility.
            sample_weighting:
                acceptance aggregation weights over samples:
                {"importance","uniform"}.
            logic_state_only:
                if True, evaluate logic only on state-token positions, ignoring
                action/reward/value positions.
            append_end_symbol:
                Deprecated compatibility argument. END is now appended exactly once
                in canonical mode regardless of this flag.
        """

        if isinstance(deep_dfa, (list, tuple)):
            self.deep_dfa = [d.to(device) for d in deep_dfa]
        else:
            self.deep_dfa = deep_dfa.to(device)
        self.adapter = adapter
        self.mode = mode
        self.num_samples = num_samples
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps
        self.clamp_acceptance = clamp_acceptance
        if acceptance_floor_mode is None:
            acceptance_floor_mode = "clamp" if clamp_acceptance else "none"
        self.acceptance_floor_mode = str(acceptance_floor_mode)
        self.sample_weighting = str(sample_weighting)
        self.logic_state_only = bool(logic_state_only)
        self.last_logic_stats = {}
        if append_end_symbol:
            warnings.warn(
                "append_end_symbol is deprecated and ignored. "
                "Canonical end semantics always append one explicit END step.",
                DeprecationWarning,
            )
        self.append_end_symbol = append_end_symbol

    def _gumbel_softmax_samples(self, logits, num_samples, temperature):
        """
        Draw differentiable samples from model logits using Gumbel-Softmax.

        Args:
            logits:
                tensor of shape [batch_size, seq_len, num_token_ids], where
                num_token_ids is the size of the model's token ID domain.
            num_samples:
                number of samples to draw per sequence.
            temperature:
                Gumbel-Softmax temperature parameter.

        Returns:
            samples:
                tensor of shape [batch_size, num_samples, seq_len, num_token_ids],
                containing soft one-hot vectors over token IDs.
            log_probs:
                tensor of shape [batch_size, seq_len, num_token_ids], containing
                log p(token_id | prefix) for each position.
        """

        batch_size, seq_len, num_token_ids = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        logits_exp = logits.unsqueeze(1).expand(batch_size, num_samples, seq_len, num_token_ids)
        samples = F.gumbel_softmax(logits_exp, tau=temperature, hard=False, dim=-1)
        return samples, log_probs

    def global_logic_loss_tt(
        self,
        model,
        batch,
        deep_dfa,
        adapter,
        num_samples=10,
        temperature=0.5,
        alpha=0.4,
        return_components=False,
    ):
        """
        Compute global logic loss (and combine it with supervised loss) for a batch.

        Steps:
            1) Forward the model on inputs X to get logits and supervised loss.
            2) Gumbel-softmax sampling:
                   draw 'num_samples' soft trajectories from the logits.
            3) Map token distributions to DFA symbol distributions via the adapter.
            4) Feed symbol distributions to DeepDFA.forward_pi to obtain, for each
               sampled trajectory, an acceptance probability.
            5) Use a weighted Monte-Carlo estimator over samples to estimate
               P_theta(trace satisfies constraint).
            6) Define logic_loss = -log(mean_acceptance) and combine with supervised loss.

        Args:
            model:
                neural model taking (X, targets=Y, mask=mask) and returning
                (logits, supervised_loss).
            batch:
                (X, Y, mask) triple from the dataset.
            deep_dfa:
                DeepDFA instance (typically self.deep_dfa).
            adapter:
                adapter instance (typically self.adapter) for token->symbol mapping.
            num_samples:
                number of Gumbel-Softmax trajectories per sequence.
            temperature:
                Gumbel-Softmax temperature.
            alpha:
                logic regularization weight in total_loss = sup_loss + alpha * logic_loss.
            return_components:
                if True, return (total_loss, sup_loss, logic_loss) separately,
                otherwise return only total_loss.

        Returns:
            total_loss or (total_loss, sup_loss, logic_loss).
        """

        if len(batch) != 3:
            raise ValueError(f"Expected batch to be (X, Y, mask); got length {len(batch)}")

        x, y, mask = batch
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits, sup_loss = model(x, targets=y, mask=mask)
        batch_size, seq_len, num_token_ids = logits.shape

        if num_token_ids != adapter.num_token_ids:
            raise ValueError(
                f"Model logits last dim ({num_token_ids}) != adapter.num_token_ids ({adapter.num_token_ids})"
            )

        samples, log_probs = self._gumbel_softmax_samples(
            logits, num_samples=num_samples, temperature=temperature
        )
        # samples: [batch_size, num_samples, seq_len, num_token_ids]
        traces_soft = samples.view(batch_size * num_samples, seq_len, num_token_ids)

        sym_probs = adapter.token_probs_to_symbol_probs(traces_soft)
        if self.logic_state_only:
            pos = torch.arange(seq_len, device=sym_probs.device)
            state_mask = (pos % int(adapter.transition_dim)) == 0
            sym_probs = sym_probs[:, state_mask, :]
        # Canonical finite-trace semantics: consume explicit END exactly once.
        sym_probs = adapter.append_terminal_end_symbol_probs(sym_probs)

        deep_dfa = deep_dfa.to(device)
        sym_probs = sym_probs.to(device)

        dfa_states, dfa_rew_seq = deep_dfa.forward_pi(sym_probs)
        dfa_final = dfa_rew_seq[:, -1, :]

        if dfa_final.size(-1) < 2:
            raise ValueError("DeepDFA final reward has <2 outputs; expected [reject, accept].")

        acceptance = dfa_final[:, 1].clamp(min=0.0, max=1.0)
        acceptance = acceptance.view(batch_size, num_samples)

        log_probs_exp = log_probs.unsqueeze(1).expand(
            batch_size, num_samples, seq_len, num_token_ids
        )
        log_prob_traces = (samples * log_probs_exp).sum(dim=-1).sum(dim=-1)

        weights = compute_sample_weights(log_prob_traces, mode=self.sample_weighting)
        prob_acceptance = (weights * acceptance).sum(dim=-1)

        prob_safe = apply_acceptance_floor(
            prob_acceptance, mode=self.acceptance_floor_mode, eps=self.eps
        )

        logic_loss = -torch.log(prob_safe).mean()
        with torch.no_grad():
            eps = float(self.eps if self.eps is not None else 0.0)
            self.last_logic_stats = {
                "prob_acceptance_mean": float(prob_acceptance.mean().item()),
                "prob_acceptance_min": float(prob_acceptance.min().item()),
                "prob_acceptance_max": float(prob_acceptance.max().item()),
                "frac_prob_acceptance_le_eps": float((prob_acceptance <= eps).float().mean().item())
                if eps > 0.0
                else None,
            }

        total_loss = sup_loss + alpha * logic_loss

        if return_components:
            return total_loss, sup_loss, logic_loss
        else:
            return total_loss

    def local_logic_loss_tt(self, *args, **kwargs):
        """
        Placeholder for a local logic loss variant.

        Not implemented. Use mode='global' instead.
        """
        raise NotImplementedError(
            "Local logic loss for TT is not implemented yet. Use mode='global'."
        )

    def compute_loss(self, model, batch, return_components=False):
        """
        Dispatch to the appropriate logic loss computation based on self.mode.

        Args:
            model:
                neural model to be trained.
            batch:
                data batch (X, Y, mask).
            return_components:
                if True, return (total_loss, sup_loss, logic_loss).

        Returns:
            total_loss or (total_loss, sup_loss, logic_loss), depending on return_components.

        Raises:
            ValueError if mode is not 'global' or 'local'.
        """

        if self.mode == "global":
            if isinstance(self.deep_dfa, list):
                # compute logic loss for each constraint and average
                total_losses = []
                sup_losses = []
                logic_losses = []
                for dfa_inst in self.deep_dfa:
                    tl, sl, ll = self.global_logic_loss_tt(
                        model,
                        batch,
                        dfa_inst,
                        self.adapter,
                        num_samples=self.num_samples,
                        temperature=self.temperature,
                        alpha=self.alpha,
                        return_components=True,
                    )
                    total_losses.append(tl)
                    sup_losses.append(sl)
                    logic_losses.append(ll)
                total_loss = torch.stack(total_losses).mean()
                sup_loss = torch.stack(sup_losses).mean()
                logic_loss = torch.stack(logic_losses).mean()
                if return_components:
                    return total_loss, sup_loss, logic_loss
                return total_loss
            else:
                return self.global_logic_loss_tt(
                    model,
                    batch,
                    self.deep_dfa,
                    self.adapter,
                    num_samples=self.num_samples,
                    temperature=self.temperature,
                    alpha=self.alpha,
                    return_components=return_components,
                )
        elif self.mode == "local":
            return self.local_logic_loss_tt()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'global' or 'local'.")
