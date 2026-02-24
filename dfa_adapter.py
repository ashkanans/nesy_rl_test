import sys
import warnings
from pathlib import Path

import torch
from logic.token_schema import get_num_bins_per_dim as schema_num_bins_per_dim
from logic.token_schema import get_schema_for_env

REPO_ROOT = Path(__file__).parent
NESY_PATH = REPO_ROOT / "suffix-prediction"
sys.path.insert(0, str(NESY_PATH))

import FiniteStateMachine as FSM
from FiniteStateMachine import DFA

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def get_num_bins_per_dim_for_env(env_name, observation_bins, action_bins):
    """Canonical token-schema bins per transition dimension for each supported env."""
    schema = get_schema_for_env(env_name)
    return schema_num_bins_per_dim(schema, observation_bins, action_bins)


def get_end_token_id_from_num_bins(num_bins):
    """Single authoritative END token id derived from adapter-level token schema."""
    bins = [int(b) for b in num_bins]
    if not bins:
        raise ValueError("num_bins must be non-empty")
    return max(bins)


class TTDFAAdapter:
    """
    Adapter between trajectory token IDs and DFA/DeepDFA symbols.

    Canonical end-of-trace semantics:
      1) consume all trace symbols before END
      2) consume exactly one explicit END symbol
      3) evaluate acceptance
    """

    @staticmethod
    def get_num_bins_per_dim_for_env(env_name, observation_bins, action_bins):
        return get_num_bins_per_dim_for_env(env_name, observation_bins, action_bins)

    @staticmethod
    def get_end_token_id_from_num_bins(num_bins):
        return get_end_token_id_from_num_bins(num_bins)

    def __init__(
        self,
        observation_dim,
        action_dim,
        num_bins,
        include_reward=True,
        include_value=True,
        constraint_dims=None,
        abstraction_fn=None,
        use_stop_token=True,
    ):
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.include_reward = bool(include_reward)
        self.include_value = bool(include_value)
        self.constraint_dims = constraint_dims
        self.abstraction_fn = abstraction_fn
        self.use_stop_token = bool(use_stop_token)

        total_dims = self.observation_dim + self.action_dim
        if self.include_reward:
            total_dims += 1
        if self.include_value:
            total_dims += 1

        if isinstance(num_bins, int):
            if num_bins <= 0:
                raise ValueError(f"num_bins must be positive, got {num_bins}")
            self.num_bins_per_dim = [int(num_bins)] * total_dims
        elif isinstance(num_bins, (list, tuple)):
            if len(num_bins) != total_dims:
                raise ValueError(
                    f"len(num_bins)={len(num_bins)} but total scalar dims={total_dims}"
                )
            self.num_bins_per_dim = [int(b) for b in num_bins]
            if any(b <= 0 for b in self.num_bins_per_dim):
                raise ValueError("All num_bins entries must be positive.")
        else:
            raise ValueError("num_bins must be int or list/tuple of ints.")

        self.transition_dim = total_dims
        self.max_num_bins = max(self.num_bins_per_dim)
        self.num_token_ids = self.max_num_bins + (1 if self.use_stop_token else 0)
        self.end_token_id = self.max_num_bins if self.use_stop_token else None

        self.symbolic_vocab = []
        self.symbol_to_idx = {}
        self.pos_bin_to_sym_idx = torch.empty(
            self.transition_dim, self.num_token_ids, dtype=torch.long
        )
        self._build_symbolic_vocab_and_mapping()
        self.num_symbols = len(self.symbolic_vocab)
        self.end_symbol_id = self.symbol_to_idx.get("end")

    def _global_dim_for_pos(self, pos):
        return pos

    def _gen_symbol_name(self, global_dim, bin_id):
        if self.use_stop_token and bin_id == self.max_num_bins:
            return "end"

        if global_dim < self.observation_dim:
            prefix = f"s{global_dim}"
        elif global_dim < self.observation_dim + self.action_dim:
            prefix = f"a{global_dim - self.observation_dim}"
        elif self.include_reward and global_dim == self.observation_dim + self.action_dim:
            prefix = "r"
        elif self.include_value and global_dim == self.observation_dim + self.action_dim + 1:
            prefix = "v"
        else:
            prefix = f"x{global_dim}"

        if self.abstraction_fn is not None and bin_id < self.max_num_bins:
            return f"{prefix}_{self.abstraction_fn(global_dim, bin_id)}"
        return f"{prefix}_bin{bin_id}"

    def _add_symbol(self, symbol_name):
        if symbol_name not in self.symbol_to_idx:
            idx = len(self.symbolic_vocab)
            self.symbolic_vocab.append(symbol_name)
            self.symbol_to_idx[symbol_name] = idx
        return self.symbol_to_idx[symbol_name]

    def _build_symbolic_vocab_and_mapping(self):
        for pos in range(self.transition_dim):
            global_dim = self._global_dim_for_pos(pos)
            for bin_id in range(self.num_token_ids):
                symbol_name = self._gen_symbol_name(global_dim, bin_id)
                symbol_idx = self._add_symbol(symbol_name)
                self.pos_bin_to_sym_idx[pos, bin_id] = symbol_idx

    def token_ids_to_symbol_ids(self, token_ids):
        """
        Hard token path: map token IDs -> symbol IDs.

        Args:
            token_ids: [T] or [B, T], dtype integer

        Returns:
            symbol_ids: [B, T], dtype torch.long
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError("token_ids_to_symbol_ids expects [T] or [B, T].")

        token_ids = token_ids.long()
        if token_ids.numel() == 0:
            return token_ids.new_zeros(token_ids.shape, dtype=torch.long)

        if token_ids.min() < 0 or token_ids.max() >= self.num_token_ids:
            raise ValueError(
                f"token id out of range [0, {self.num_token_ids - 1}] in token_ids_to_symbol_ids"
            )

        batch_size, seq_len = token_ids.shape
        pos = torch.arange(seq_len, device=token_ids.device) % self.transition_dim
        mapper = self.pos_bin_to_sym_idx.to(token_ids.device)[pos]  # [T, num_token_ids]
        mapper = mapper.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, num_token_ids]
        symbol_ids = torch.gather(mapper, 2, token_ids.unsqueeze(-1)).squeeze(-1)
        return symbol_ids.long()

    def tokens_to_symbols(self, tokens):
        symbol_ids = self.token_ids_to_symbol_ids(tokens)
        seqs = []
        for row in symbol_ids:
            seqs.append([self.symbolic_vocab[int(i)] for i in row.tolist()])
        if symbol_ids.shape[0] == 1:
            return seqs[0]
        return seqs

    def token_probs_to_symbol_probs(self, token_probs):
        """
        Soft token path: map token probabilities -> symbol probabilities.

        Args:
            token_probs: [B, T, num_token_ids]

        Returns:
            symbol_probs: [B, T, num_symbols]
        """
        if token_probs.dim() != 3:
            raise ValueError("token_probs_to_symbol_probs expects [B, T, num_token_ids].")
        if token_probs.size(-1) != self.num_token_ids:
            raise ValueError(
                f"token_probs last dim {token_probs.size(-1)} != num_token_ids {self.num_token_ids}"
            )

        batch_size, seq_len, num_token_ids = token_probs.shape
        sym_probs = token_probs.new_zeros(batch_size, seq_len, self.num_symbols)
        pos_bin_to_sym_idx = self.pos_bin_to_sym_idx.to(token_probs.device)

        for t in range(seq_len):
            pos = t % self.transition_dim
            sym_idx_row = pos_bin_to_sym_idx[pos]
            idx = sym_idx_row.unsqueeze(0).expand(batch_size, num_token_ids)
            sym_probs[:, t, :].scatter_add_(1, idx, token_probs[:, t, :])

        return sym_probs

    def append_terminal_end_symbol_probs(self, symbol_probs):
        """Append exactly one deterministic END symbol step."""
        if self.end_symbol_id is None:
            raise ValueError("Adapter has no END symbol; cannot append terminal END.")
        if symbol_probs.dim() != 3:
            raise ValueError("append_terminal_end_symbol_probs expects [B, T, S].")
        end_step = symbol_probs.new_zeros(symbol_probs.shape[0], 1, self.num_symbols)
        end_step[:, 0, self.end_symbol_id] = 1.0
        return torch.cat([symbol_probs, end_step], dim=1)

    def create_dfa_from_ltl(self, ltl_formula, formula_name="constraint", use_safe_dfa=False):
        # Canonical behavior: disable legacy DFA end-state hack.
        FSM.USE_END_HACK = False

        if use_safe_dfa and ltl_formula.startswith("G("):
            return self._build_safe_dfa_from_unsafe_set(ltl_formula, formula_name)

        return DFA(ltl_formula, self.num_symbols, formula_name, dictionary_symbols=self.symbolic_vocab)

    def _parse_unsafe_symbols(self, ltl_formula):
        return [sym for sym in self.symbolic_vocab if sym in ltl_formula]

    def _build_safe_dfa_from_unsafe_set(self, ltl_formula, formula_name):
        unsafe_syms = set(self._parse_unsafe_symbols(ltl_formula))
        transitions = {0: {}, 1: {}}
        for idx, sym in enumerate(self.symbolic_vocab):
            if sym in unsafe_syms:
                transitions[0][idx] = 1
                transitions[1][idx] = 1
            else:
                transitions[0][idx] = 0
                transitions[1][idx] = 1
        acceptance = [True, False]
        return DFA(transitions, acceptance, None, dictionary_symbols=self.symbolic_vocab)

    def _symbols_to_dfa_indices(self, symbol_seq, dfa):
        dfa_symbol_to_idx = {s: i for i, s in enumerate(dfa.dictionary_symbols)}
        fallback = dfa_symbol_to_idx.get("end", 0)
        return [dfa_symbol_to_idx.get(symbol, fallback) for symbol in symbol_seq]

    def _apply_mask_to_state_only(self, token_sequences):
        seq_len = token_sequences.shape[1]
        positions = torch.arange(seq_len, device=token_sequences.device)
        state_positions = (positions % self.transition_dim) == 0
        masked = token_sequences.clone()
        non_state_positions = ~state_positions

        if self.end_token_id is None:
            masked[:, non_state_positions] = 0
            return masked

        # Keep END markers untouched; map all other non-state tokens to 0.
        non_end = masked != self.end_token_id
        masked[:, non_state_positions] = torch.where(
            non_end[:, non_state_positions],
            torch.zeros_like(masked[:, non_state_positions]),
            masked[:, non_state_positions],
        )
        return masked

    def _normalize_end_token_ids(self, seq, allow_missing_end=False, debug=False):
        if self.end_token_id is None:
            raise ValueError("Adapter has no END token configured.")

        end_positions = (seq == self.end_token_id).nonzero(as_tuple=False).flatten()
        if len(end_positions) == 0:
            if not allow_missing_end:
                return None
            prefix = seq
        else:
            first_end = int(end_positions[0].item())
            if len(end_positions) > 1 and debug:
                warnings.warn(
                    "Multiple END tokens found. Truncating at first END per canonical semantics.",
                    RuntimeWarning,
                )
            prefix = seq[:first_end]

        end_token = torch.tensor([self.end_token_id], dtype=seq.dtype, device=seq.device)
        return torch.cat([prefix, end_token], dim=0)

    def check_sat_token_ids(
        self,
        token_ids,
        dfa,
        mask_to_state_only=False,
        allow_missing_end=False,
        debug=False,
    ):
        """
        Check DFA satisfaction on hard token traces.

        Returns:
            bool tensor [B]
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError("check_sat_token_ids expects [T] or [B, T].")

        token_ids = token_ids.long()
        if mask_to_state_only:
            token_ids = self._apply_mask_to_state_only(token_ids)

        out = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
        for b in range(token_ids.shape[0]):
            normalized = self._normalize_end_token_ids(
                token_ids[b], allow_missing_end=allow_missing_end, debug=debug
            )
            if normalized is None:
                out[b] = False
                continue

            symbol_ids = self.token_ids_to_symbol_ids(normalized.unsqueeze(0))[0]
            symbol_names = [self.symbolic_vocab[int(i)] for i in symbol_ids.tolist()]
            dfa_indices = self._symbols_to_dfa_indices(symbol_names, dfa)
            out[b] = bool(dfa.accepts_from_state(0, dfa_indices))

        return out

    def _normalize_symbol_probs(self, seq_probs, allow_missing_end=False, debug=False):
        if self.end_symbol_id is None:
            raise ValueError("Adapter has no END symbol configured.")

        hard_ids = seq_probs.argmax(dim=-1)
        end_positions = (hard_ids == self.end_symbol_id).nonzero(as_tuple=False).flatten()

        if len(end_positions) == 0:
            if not allow_missing_end:
                return None
            prefix = seq_probs
        else:
            first_end = int(end_positions[0].item())
            if len(end_positions) > 1 and debug:
                warnings.warn(
                    "Multiple END symbols found in symbol-prob trace. "
                    "Truncating at first END per canonical semantics.",
                    RuntimeWarning,
                )
            prefix = seq_probs[:first_end, :]

        end_step = seq_probs.new_zeros(1, self.num_symbols)
        end_step[0, self.end_symbol_id] = 1.0
        return torch.cat([prefix, end_step], dim=0)

    def check_sat_symbol_probs(self, symbol_probs, deep_dfa, allow_missing_end=False, debug=False):
        """
        Check soft satisfaction from symbol probabilities using DeepDFA.

        Args:
            symbol_probs: [B, T, S]
            deep_dfa: DeepDFA instance

        Returns:
            acceptance probability tensor [B] in [0, 1]
        """
        if symbol_probs.dim() != 3:
            raise ValueError("check_sat_symbol_probs expects [B, T, S].")
        if symbol_probs.size(-1) != self.num_symbols:
            raise ValueError(
                f"symbol_probs last dim {symbol_probs.size(-1)} != num_symbols {self.num_symbols}"
            )

        normalized = []
        valid = []
        for b in range(symbol_probs.shape[0]):
            seq = self._normalize_symbol_probs(
                symbol_probs[b], allow_missing_end=allow_missing_end, debug=debug
            )
            if seq is None:
                valid.append(False)
                normalized.append(None)
            else:
                valid.append(True)
                normalized.append(seq)

        if not any(valid):
            return symbol_probs.new_zeros(symbol_probs.shape[0])

        max_len = max(seq.shape[0] for seq in normalized if seq is not None)
        end_row = symbol_probs.new_zeros(self.num_symbols)
        end_row[self.end_symbol_id] = 1.0
        batch = end_row.view(1, 1, -1).repeat(symbol_probs.shape[0], max_len, 1)
        lengths = []
        for i, seq in enumerate(normalized):
            if seq is None:
                lengths.append(1)
                continue
            batch[i, : seq.shape[0], :] = seq
            lengths.append(seq.shape[0])

        # DeepDFA keeps transition tensors as plain attributes (not parameters/buffers),
        # so module.to(...) does not guarantee they move devices. Run on DeepDFA's
        # actual tensor device and move outputs back to symbol_probs.device.
        deep_device = symbol_probs.device
        if hasattr(deep_dfa, "trans_prob") and isinstance(deep_dfa.trans_prob, torch.Tensor):
            deep_device = deep_dfa.trans_prob.device
        elif hasattr(deep_dfa, "fin_matrix") and isinstance(deep_dfa.fin_matrix, torch.Tensor):
            deep_device = deep_dfa.fin_matrix.device

        _, dfa_rew_seq = deep_dfa.forward_pi(batch.to(deep_device))
        dfa_rew_seq = dfa_rew_seq.to(symbol_probs.device)
        accept_probs = symbol_probs.new_zeros(symbol_probs.shape[0])
        for i in range(symbol_probs.shape[0]):
            if not valid[i]:
                accept_probs[i] = 0.0
            else:
                accept_probs[i] = dfa_rew_seq[i, lengths[i] - 1, 1]
        return accept_probs.clamp(min=0.0, max=1.0)

    # Backward-compatible wrapper for old call sites.
    def batch_check_dfa_sat(
        self,
        token_sequences,
        dfa,
        mask_to_state_only=False,
    ):
        sat_bool = self.check_sat_token_ids(
            token_sequences, dfa, mask_to_state_only=mask_to_state_only, allow_missing_end=False
        )
        return sat_bool.float().to(device)
