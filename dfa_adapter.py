import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent
NESY_PATH = REPO_ROOT / "suffix-prediction"
sys.path.insert(0, str(NESY_PATH))

from FiniteStateMachine import DFA

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class TTDFAAdapter:
    """
    Adapter between Trajectory Transformer token IDs and DFA symbols.

    This class maps:
      - token IDs produced by a sequence model (interpreted as discretization bins)
      - to symbolic labels used by a DFA (e.g. "s0_bin3", "a1_bin0", "r_bin1", "end").

    Key ideas:
      - We assume a fixed ordering of scalar dimensions in a flattened transition:
            [s_0, s_1, ..., s_{obs-1}, a_0, ..., a_{act-1}, (r), (v), s_0, ...]
        so that each sequence position t corresponds to a "logical dimension"
            pos = t % transition_dim
      - For each scalar dimension i, we specify num_bins_per_dim[i] bins
        (the dimension's cardinality if discrete, or a discretization size if continuous).
      - All dimensions share a global token-ID domain {0, ..., max_num_bins-1}
        plus an optional "end" token at ID max_num_bins.
      - A symbolic vocabulary of strings is built by combining:
            prefix(global_dim, kind)  ×  bin ID or abstraction label
        so that DFA constraints can refer to these symbols in LTL formulas.

    Discrete identity case:
      - For a purely discrete environment with no abstraction, use:
            num_bins_per_dim[i] = cardinality[i]
        and feed the raw discrete values 0..card_i-1 as token IDs per dimension.
        The adapter will then be numerically "identity" at the token level, while
        still providing symbolic names to the DFA.
    """

    def __init__(
        self, observation_dim, action_dim, num_bins, include_reward=True,
        include_value=True, constraint_dims=None, abstraction_fn=None, use_stop_token=True
    ):
        """
        Initialize the DFA adapter.

        Args:
            observation_dim:
                number of scalar observation/state dimensions.
            action_dim:
                number of scalar action dimensions.
            num_bins:
                either:
                  - int: same number of bins for all scalar dimensions
                  - list/tuple of ints: num_bins_per_dim[global_dim] for each dimension
                    in the flattened transition.
            include_reward:
                whether to include a reward scalar dimension in the transition.
            include_value:
                whether to include a value/RTG scalar dimension in the transition.
            constraint_dims:
                reserved for future use; can specify which dimensions constraints refer to.
            abstraction_fn:
                optional function (global_dim, bin_id) -> label string. If provided,
                symbols become "{prefix}_{label}" instead of "{prefix}_bin{bin_id}".
            use_stop_token:
                whether to reserve a special token ID "end" at bin_id = max_num_bins.
        """

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.include_reward = bool(include_reward)
        self.include_value = bool(include_value)

        self.constraint_dims = constraint_dims
        self.abstraction_fn = abstraction_fn
        self.use_stop_token = use_stop_token

        # total scalar dimensions in a single transition
        total_dims = self.observation_dim + self.action_dim
        if self.include_reward:
            total_dims += 1
        if self.include_value:
            total_dims += 1

        # num_bins can be int (same for all dimensions) or list (per dimension)
        if isinstance(num_bins, int):
            if num_bins <= 0:
                raise ValueError("num_bins must be positive, got %d" % num_bins)
        
            self.num_bins_per_dim = [num_bins] * total_dims
        
        elif isinstance(num_bins, (list, tuple)):
            if len(num_bins) != total_dims:
                raise ValueError(
                    "len(num_bins) = %d, but total scalar dims = %d"
                    % (len(num_bins), total_dims)
                )
            
            self.num_bins_per_dim = [int(b) for b in num_bins]
            
            if any(b <= 0 for b in self.num_bins_per_dim):
                raise ValueError("All num_bins entries must be positive.")
        
        else:
            raise ValueError("num_bins must be int or list/tuple of ints.")

        self.transition_dim = total_dims
        self.max_num_bins = max(self.num_bins_per_dim)
        self.num_token_ids = self.max_num_bins + (1 if self.use_stop_token else 0)

        self.symbolic_vocab = []
        self.symbol_to_idx = {}
        self.pos_bin_to_sym_idx = torch.empty(
            self.transition_dim, self.num_token_ids, dtype=torch.long
        )

        self._build_symbolic_vocab_and_mapping()
        self.num_symbols = len(self.symbolic_vocab)

    def _global_dim_for_pos(self, pos):
        """
        Map a sequence position to a global scalar dimension index.

        By default:
            global_dim == pos
        but this helper exists so that more complex mappings (e.g. grouping positions)
        can be implemented without touching the rest of the adapter.
        """
        return pos

    def _gen_symbol_name(self, global_dim, bin_id):
        """
        Generate a symbol name for a given scalar dimension and bin ID.

        Naming scheme:
          - If bin_id == max_num_bins and use_stop_token=True:
                "end"
          - Else:
                prefix(global_dim) = one of:
                    "s{i}" for observation dims
                    "a{j}" for action dims
                    "r"   for reward dim
                    "v"   for value dim
                    "x{global_dim}" for extra/unknown dims
                and:
                    "{prefix}_{label}" if abstraction_fn is provided
                    "{prefix}_bin{bin_id}" otherwise.
        """

        if self.use_stop_token and bin_id == self.max_num_bins:
            return "end"

        if global_dim < self.observation_dim:
            prefix = f"s{global_dim}"
        elif global_dim < self.observation_dim + self.action_dim:
            a_dim = global_dim - self.observation_dim
            prefix = f"a{a_dim}"
        elif self.include_reward and global_dim == self.observation_dim + self.action_dim:
            prefix = "r"
        elif self.include_value and global_dim == self.observation_dim + self.action_dim + 1:
            prefix = "v"
        else:
            prefix = f"x{global_dim}"

        if self.abstraction_fn is not None and bin_id < self.max_num_bins:
            label = self.abstraction_fn(global_dim, bin_id)
            return f"{prefix}_{label}"
        else:
            return f"{prefix}_bin{bin_id}"

    def _add_symbol(self, symbol_name):
        """
        Add a symbol string to the vocabulary if not already present,
        and return its index.
        """

        if symbol_name not in self.symbol_to_idx:
            idx = len(self.symbolic_vocab)
            self.symbolic_vocab.append(symbol_name)
            self.symbol_to_idx[symbol_name] = idx
        
        return self.symbol_to_idx[symbol_name]

    def _build_symbolic_vocab_and_mapping(self):
        """
        Build:
          - the symbolic vocabulary (list of strings)
          - the tensor mapping (pos, bin_id) -> symbol index.

        For each position pos in [0, transition_dim):
          global_dim = _global_dim_for_pos(pos)
          For each bin_id in [0, num_token_ids):
            symbol_name = _gen_symbol_name(global_dim, bin_id)
            symbol_idx  = _add_symbol(symbol_name)
            pos_bin_to_sym_idx[pos, bin_id] = symbol_idx
        """

        for pos in range(self.transition_dim):
            global_dim = self._global_dim_for_pos(pos)

            for bin_id in range(self.num_token_ids):
                symbol_name = self._gen_symbol_name(global_dim, bin_id)
                symbol_idx = self._add_symbol(symbol_name)
                self.pos_bin_to_sym_idx[pos, bin_id] = symbol_idx

    def tokens_to_symbols(self, tokens):
        """
        Convert sequences of token IDs into sequences of symbolic strings.

        Args:
            tokens:
                tensor of shape [seq_len] or [batch_size, seq_len] with integer IDs.

        Returns:
            - if batch_size == 1: list of strings (one sequence)
            - else: list of lists of strings (one per batch element)
        """

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        tokens = tokens.long()
        batch_size, seq_len = tokens.shape
        seqs = []

        for b in range(batch_size):
            seq = []
            for t in range(seq_len):
                pos = t % self.transition_dim
                bin_id = int(tokens[b, t].item())

                if bin_id < 0 or bin_id >= self.num_token_ids:
                    continue

                sym_idx = int(self.pos_bin_to_sym_idx[pos, bin_id].item())
                symbol = self.symbolic_vocab[sym_idx]
                seq.append(symbol)
            seqs.append(seq)

        if batch_size == 1:
            return seqs[0]
        return seqs

    def token_probs_to_symbol_probs(self, token_probs):
        """
        Map token-level probabilities to symbol-level probabilities.

        Args:
            token_probs:
                tensor of shape [batch_size, seq_len, num_token_ids], where the last
                dimension is a probability distribution over token IDs.

        Returns:
            sym_probs:
                tensor of shape [batch_size, seq_len, num_symbols], where each slice
                sym_probs[b, t, :] is the induced distribution over DFA symbols.
        """

        if token_probs.size(-1) != self.num_token_ids:
            raise ValueError(
                f"token_probs last dim {token_probs.size(-1)} != num_token_ids {self.num_token_ids}"
            )

        batch_size, seq_len, num_token_ids = token_probs.shape
        sym_probs = token_probs.new_zeros(batch_size, seq_len, self.num_symbols)

        pos_bin_to_sym_idx = self.pos_bin_to_sym_idx.to(token_probs.device)

        for t in range(seq_len):
            pos = t % self.transition_dim
            sym_idx_row = pos_bin_to_sym_idx[pos]  # [num_token_ids]
            idx = sym_idx_row.unsqueeze(0).expand(batch_size, num_token_ids)
            sym_probs[:, t, :].scatter_add_(1, idx, token_probs[:, t, :])

        return sym_probs

    def create_dfa_from_ltl(self, ltl_formula, formula_name="constraint"):
        """
        Build a DFA from an LTL formula using the current symbolic vocabulary.
        """

        # FiniteStateMachine.DFA signature is (ltl_formula, num_symbols, name, dictionary_symbols)
        dfa = DFA(ltl_formula, self.num_symbols, formula_name, dictionary_symbols=self.symbolic_vocab)

        return dfa

    def _symbols_to_dfa_indices(self, symbol_seq, dfa):
        """
        Map a sequence of symbol strings to DFA dictionary indices.
        """
        
        indices = []

        for symbol in symbol_seq:
            if symbol in dfa.dictionary_symbols:
                idx = dfa.dictionary_symbols.index(symbol)
            else:
                print(f"[WARNING] Symbol '{symbol}' not in DFA vocabulary")
                if "end" in dfa.dictionary_symbols:
                    idx = dfa.dictionary_symbols.index("end")
                else:
                    idx = 0
            indices.append(idx)

        return indices

    def batch_check_dfa_sat(self, token_sequences, dfa):
        """
        Check DFA satisfaction for a batch of token sequences.

        Args:
            token_sequences:
                tensor of shape [batch_size, seq_len] with integer token IDs.
            dfa:
                DFA with method accepts_from_state(initial_state, indices).

        Returns:
            Tensor of shape [batch_size], dtype float32, with 1.0 if accepted, 0.0 otherwise.
        """

        if token_sequences.dim() != 2:
            raise ValueError("batch_check_dfa_sat expects [batch_size, seq_len]")

        token_sequences = token_sequences.long()
        batch_size = token_sequences.shape[0]
        satisfaction = torch.zeros(batch_size, dtype=torch.float32, device=device)

        symbol_sequences = self.tokens_to_symbols(token_sequences)
        if isinstance(symbol_sequences, list) and symbol_sequences and isinstance(symbol_sequences[0], str):
            # single sequence case: wrap in list
            symbol_sequences = [symbol_sequences]

        for i, symbol_seq in enumerate(symbol_sequences):
            indices = self._symbols_to_dfa_indices(symbol_seq, dfa)
            is_accepted = dfa.accepts_from_state(0, indices)
            satisfaction[i] = 1.0 if is_accepted else 0.0

        return satisfaction
