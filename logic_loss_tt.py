import torch
import torch.nn.functional as F
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


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

        6) The final training loss is a convex combination:

               total_loss = (1 - alpha) * supervised_loss + alpha * logic_loss

    This provides a differentiable way to inject global LTL constraints into
    sequence model training.
    """

    def __init__(
        self, deep_dfa, adapter, mode="global",
        num_samples=10, temperature=0.5, alpha=0.4
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
                mixing coefficient in [0, 1] between supervised and logic loss.
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
        self, model, batch, deep_dfa, adapter, num_samples=10,
        temperature=0.5, alpha=0.4, return_components=False
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
                mixing coefficient between supervised and logic loss.
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

        deep_dfa = deep_dfa.to(device)
        sym_probs = sym_probs.to(device)

        dfa_states, dfa_rew_seq = deep_dfa.forward_pi(sym_probs)
        dfa_final = dfa_rew_seq[:, -1, :]

        if dfa_final.size(-1) < 2:
            raise ValueError("DeepDFA final reward has <2 outputs; expected [reject, accept].")

        acceptance = dfa_final[:, 1]
        acceptance = acceptance.view(batch_size, num_samples)

        log_probs_exp = log_probs.unsqueeze(1).expand(batch_size, num_samples, seq_len, num_token_ids)
        log_prob_traces = (samples * log_probs_exp).sum(dim=-1).sum(dim=-1)

        weights = F.softmax(log_prob_traces, dim=-1)
        prob_acceptance = (weights * acceptance).sum(dim=-1)

        eps = 1e-10
        logic_loss = -torch.log(prob_acceptance.clamp(min=eps)).mean()

        total_loss = (1.0 - alpha) * sup_loss + alpha * logic_loss

        if return_components:
            return total_loss, sup_loss, logic_loss
        else:
            return total_loss

    def local_logic_loss_tt(self, *args, **kwargs):
        """
        Placeholder for a local logic loss variant.

        Not implemented. Use mode='global' instead.
        """
        raise NotImplementedError("Local logic loss for TT is not implemented yet. Use mode='global'.")

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
                        model, batch, dfa_inst, self.adapter,
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
                    model, batch, self.deep_dfa, self.adapter, num_samples=self.num_samples,
                    temperature=self.temperature, alpha=self.alpha, return_components=return_components
                )
        elif self.mode == "local":
            return self.local_logic_loss_tt()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'global' or 'local'.")
