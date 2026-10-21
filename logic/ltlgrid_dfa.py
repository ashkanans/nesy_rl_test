"""
DFA / reward machine for the NRM 4x4 "mine" grid tasks (Neurosymbolic-DT-eval).

The 10 LTLf tasks are over the symbols c0..c3 = pickaxe (P), lava (L), door (D), gem (G); the
fifth grid symbol E (empty) is not mentioned by any formula and leaves the DFA state unchanged.
The DFA is built with flloat (optional dependency) and the "distance" reward is the shaped
potential of the original MooreMachine(reward="distance"): the potential of a DFA state grows with
its proximity to acceptance, 100 in the accepting absorbing state and -100 in the failing one.
"""

import copy
import functools
import json
from pathlib import Path

SYMBOLS = ("P", "L", "D", "G", "E")

# (LTLf formula over c0..c3, number of c-symbols used, name); same order as LTL_tasks.py upstream.
LTLGRID_FORMULAS = [
    ("(F c0) & (F c1)", 2, "task1: visit(P, L)"),
    ("(F c0) & (F c1) & (F c2)", 3, "task2: visit(P, L, D)"),
    ("F(c0 & F(c1))", 2, "task3: seq_visit(P, L)"),
    ("F(c0 & F(c1)) & F(c2 & F(c3))", 4, "task4: seq_visit(P, L) + seq_visit(D, G)"),
    ("F(c0 & F(c1)) & (F c2)", 3, "task5: seq_visit(P, L) + visit(D)"),
    ("F(c0 & F(c1)) & (F c2) & (F c3)", 4, "task6: seq_visit(P, L) + visit(D, G)"),
    ("(F c0) & (F c1) & (G (! c2))", 3, "task7: visit(P, L) + glob_av(D)"),
    ("(F c0) & (F c1) & (G (! c2)) & (G(! c3))", 4, "task8: visit(P, L) + glob_av(D) + glob_av(G)"),
    ("F(c0 & F(c1)) & G (! c2)", 3, "task9: seq_visit(P, L) + glob_av(D)"),
    ("F(c0 & F(c1)) & G (! c2) & G(! c3)", 4, "task10: seq_visit(P, L) + glob_av(D) + glob_av(G)"),
]

REWARD_TYPES = ("distance", "three_value")


TABLES_PATH = Path(__file__).with_name("ltlgrid_dfa_tables.json")


@functools.lru_cache(maxsize=None)
def _load_tables():
    """Precomputed DFAs of the 10 tasks (flloat needs 10-20 s for some, so they are stored)."""
    if not TABLES_PATH.exists():
        return {}
    return json.loads(TABLES_PATH.read_text())


def _build_transitions(formula, num_c_symbols, num_symbols):
    entry = _load_tables().get(formula)
    if entry is not None and entry["num_symbols"] == num_symbols:
        transitions = {int(q): {int(y): int(r) for y, r in row.items()} for q, row in entry["transitions"].items()}
        return transitions, list(entry["acceptance"])
    return _build_transitions_flloat(formula, num_c_symbols, num_symbols)


@functools.lru_cache(maxsize=None)
def _build_transitions_flloat(formula, num_c_symbols, num_symbols):
    from flloat.parser.ltlf import LTLfParser

    dfa = LTLfParser()(formula).to_automaton()
    alphabet = [f"c{i}" for i in range(num_c_symbols)]
    one_hot = []
    for true_sym in alphabet:
        assignment = {sym: False for sym in alphabet}
        assignment[true_sym] = True
        one_hot.append(assignment)

    raw = {s: {} for s in dfa._states}
    for s in dfa._states:
        for target, label in dfa._transition_function[s].items():
            for sym, assignment in enumerate(one_hot):
                if label.subs(assignment):
                    raw[s][sym] = target
    # symbols the formula does not mention (including the empty cell) leave the state unchanged
    for s in raw:
        for sym in range(num_symbols):
            raw[s].setdefault(sym, s)

    # flloat state ids depend on hashing; number states breadth-first from the initial state
    # (symbols in order) so that ids are identical across runs
    order = [dfa._initial_state]
    for s in order:
        for sym in range(num_symbols):
            if raw[s][sym] not in order:
                order.append(raw[s][sym])
    renumber = {s: i for i, s in enumerate(order)}
    states = order
    transitions = {renumber[s]: {sym: renumber[raw[s][sym]] for sym in range(num_symbols)} for s in states}
    acceptance = [s in dfa._final_states for s in states]
    return transitions, acceptance


def _absorbing_states(transitions):
    return [q for q, row in transitions.items() if all(t == q for t in row.values())]


def _distance_rewards(transitions, acceptance, absorbing):
    n = len(acceptance)
    acceptance = list(acceptance)
    rewards = [100] * n
    if not any(acceptance):
        for s in range(n):
            if s in absorbing:
                continue
            if all(t == s or t in absorbing for t in transitions[s].values()):
                acceptance[s] = True
    for s in range(n):
        if acceptance[s]:
            rewards[s] = 0
    while True:
        old = list(rewards)
        for s in range(n):
            if not acceptance[s]:
                nxt = [rewards[t] for t in transitions[s].values() if t != s]
                if nxt:
                    rewards[s] = 1 + min(nxt)
        if rewards == old:
            break
    rewards = [-r for r in rewards]
    minimum = min(r for r in rewards if r != -100)
    rewards = [r if r == -100 else r - minimum for r in rewards]
    maximum = max(rewards)
    return [r if r == -100 else 100 * r / maximum for r in rewards]


def _three_value_rewards(acceptance, absorbing):
    # Same rule as the current upstream MooreMachine: any accepting state is a win (with a G(!x)
    # clause it is not absorbing, since it can still fall into the failure sink); a non-accepting
    # state that can never be left is a failure.
    return [100 if acceptance[q] else (-100 if q in absorbing else 0) for q in range(len(acceptance))]


class LTLGridRewardMachine:
    """Moore machine over the grid symbols: DFA transitions plus a potential per state."""

    def __init__(self, task_id, reward="distance"):
        if reward not in REWARD_TYPES:
            raise ValueError(f"reward must be one of {REWARD_TYPES}, got '{reward}'")
        formula, num_c_symbols, name = LTLGRID_FORMULAS[int(task_id)]
        self.task_id = int(task_id)
        self.formula = formula
        self.name = name
        self.reward_type = reward
        self.num_symbols = len(SYMBOLS)
        transitions, acceptance = _build_transitions(formula, num_c_symbols, self.num_symbols)
        self.transitions, self.acceptance = copy.deepcopy(transitions), list(acceptance)
        self.num_states = len(self.acceptance)
        self.absorbing_states = _absorbing_states(self.transitions)
        if reward == "distance":
            self.potentials = _distance_rewards(self.transitions, self.acceptance, self.absorbing_states)
        else:
            self.potentials = _three_value_rewards(self.acceptance, self.absorbing_states)
        self.initial_state = 0

    def next_state(self, state, symbol):
        return self.transitions[int(state)][int(symbol)]

    def reward(self, state, next_state):
        """Shaped reward of a transition: change of potential, 0 if the state did not change."""
        if next_state == state:
            return 0.0
        return float(self.potentials[next_state] - self.potentials[state])

    def is_terminal(self, state):
        return self.potentials[int(state)] in (100, -100)

    def is_success(self, state):
        return self.potentials[int(state)] == 100


def regenerate_tables(path=TABLES_PATH):
    """Rebuild the stored DFA tables with flloat (python -m logic.ltlgrid_dfa)."""
    tables = {}
    for formula, num_c_symbols, _ in LTLGRID_FORMULAS:
        transitions, acceptance = _build_transitions_flloat(formula, num_c_symbols, len(SYMBOLS))
        tables[formula] = {
            "num_symbols": len(SYMBOLS),
            "transitions": {str(q): {str(y): r for y, r in row.items()} for q, row in transitions.items()},
            "acceptance": acceptance,
        }
    Path(path).write_text(json.dumps(tables, indent=1, sort_keys=True) + "\n")
    return tables


if __name__ == "__main__":
    regenerate_tables()
    print(f"wrote {TABLES_PATH}")
