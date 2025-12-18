import os
import random
import re
import torch
import contextlib
import io
from graphviz import Source
from pythomata import SimpleDFA
from pythomata import SymbolicAutomaton
from ltlf2dfa.parser.ltlf import LTLfParser
from DeepAutoma import DeepDFA, device

# Control whether to add the end-state hack (force acceptance only after seeing "end")
USE_END_HACK = True


def _write_dot(dot_string, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(dot_string)


def dot2dfa(dot):
    # with tempfile.TemporaryFile() as file1:
    #     print(dot)
    #     file1.write(dot.encode())
    #     Lines = file1.readlines()
    #     print(f'lines: {Lines}')

    # with open('tmp.dot', 'w') as file1:
    #     file1.write(dot)

    # with open('tmp.dot', 'r') as file1:
    #     Lines = file1.readlines()
    #     # print(f'lines: {Lines}')

    Lines = dot.split("\n")
    # print(f'lines: {Lines}')

    def _normalize_state(s):
        s = s.strip()
        try:
            return str(int(float(s)))
        except ValueError:
            return s

    count = 0
    states = set()

    for line in Lines:
        count += 1
        if count >= 11:
            # print(f'line<6: {line}')
            if line.strip()[0] == "}":
                break
            action = line.strip().split('"')[1]
            init_state = _normalize_state(line.strip().split(" ")[0])
            final_state = _normalize_state(line.strip().split(" ")[2])
            states.add(init_state)
            states.add(final_state)
        else:
            # print(f'line>6: {line}')
            if "doublecircle" in line.strip():
                nums = re.findall(r"[-+]?\d*\.?\d+", line)
                final_states = [_normalize_state(n) for n in nums]

    automaton = SymbolicAutomaton()
    state_dict = dict()
    state_dict["1"] = 0
    for state in states.union(set(final_states)):
        if state == "1":
            continue
        state_dict.setdefault(state, automaton.create_state())

    final_state_list = []
    for state in final_states:
        final_state_list.append(_normalize_state(state))

    for state in final_state_list:
        automaton.set_accepting_state(state_dict[state], True)

    count = 0
    for line in Lines:
        count += 1
        if count >= 11:
            if line.strip()[0] == "}":
                break
            action = line.strip().split('"')[1]
            init_state = _normalize_state(line.strip().split(" ")[0])
            final_state = _normalize_state(line.strip().split(" ")[2])
            automaton.add_transition((state_dict[init_state], action, state_dict[final_state]))

    automaton.set_initial_state(state_dict["1"])

    return automaton


def _one_hot_bdd(symbols, target_symbol, bdd_dict):
    """
    Build a BDD valuation where exactly one atomic proposition is true.
    """
    bdd = bdd_dict.true()
    for ap in symbols:
        var = bdd_dict.var(ap)
        if ap == target_symbol:
            bdd &= var
        else:
            bdd &= ~var
    return bdd


def _build_dfa_with_spot(ltl_formula, dictionary_symbols, formula_name):
    """
    Try to build a complete DFA with Spot (if installed).

    Returns:
        transitions: dict[state][symbol_idx] -> next_state
        acceptance: list[bool]
        dot_source: str (Graphviz)
    Raises:
        ImportError if spot is not available.
        Exception for any Spot failure.
    """
    import spot  # type: ignore

    alphabet = list(dictionary_symbols)
    ap_list = list(alphabet)

    bdd_dict = spot.make_bdd_dict()
    for ap in ap_list:
        bdd_dict.declare(ap)

    formula = spot.formula(ltl_formula)
    aut = spot.translate(
        formula,
        "deterministic",
        "complete",
        "sbacc",
        "high",
    )
    # ensure the dictionary knows about all symbols
    aut.set_bdd_dict(bdd_dict)

    dot_source = aut.to_str("dot")

    transitions = {}
    acceptance = []
    sink_state = None
    num_states = aut.num_states()

    # Precompute one-hot BDDs for each symbol
    symbol_bdds = {
        sym: _one_hot_bdd(ap_list, sym, bdd_dict) for sym in ap_list
    }

    for s in range(num_states):
        acceptance.append(aut.state_is_accepting(s))
        transitions[s] = {}
        for sym_idx, sym in enumerate(alphabet):
            letter_bdd = symbol_bdds[sym]
            dst = None
            for e in aut.out(s):
                if (e.cond & letter_bdd) != bdd_dict.false():
                    dst = e.dst
                    break
            if dst is None:
                if sink_state is None:
                    sink_state = num_states  # first free index
                dst = sink_state
            transitions[s][sym_idx] = dst

    if sink_state is not None:
        transitions[sink_state] = {
            i: sink_state for i in range(len(alphabet))
        }
        acceptance.append(False)

    return transitions, acceptance, dot_source


class DFA:

    def __init__(self, arg1, arg2, arg3, dictionary_symbols=None):
        if dictionary_symbols == None:
            self.dictionary_symbols = list(range(self.num_of_symbols))
        else:
            # make sure "end" (stop token) is the last symbol for downstream logic
            seen = []
            for sym in dictionary_symbols:
                if sym == "end":
                    continue
                if sym not in seen:
                    seen.append(sym)
            if "end" in dictionary_symbols:
                seen.append("end")
            self.dictionary_symbols = seen
        if isinstance(arg1, str):
            self.init_from_ltl(arg1, arg2, arg3, self.dictionary_symbols)
        elif isinstance(arg1, int):
            self.random_init(arg1, arg2)
        elif isinstance(arg1, dict):
            self.init_from_transacc(arg1, arg2)
        else:
            raise Exception("Uncorrect type for the argument initializing th DFA: {}".format(type(arg1)))

    def init_from_ltl(self, ltl_formula, num_symbols, formula_name, dictionary_symbols):
        self.dictionary_symbols = dictionary_symbols

        # From LTL to DFA
        #   parser = LTLfParser()
        #   ltl_formula_parsed = parser(ltl_formula)
        #   dfa = ltl_formula_parsed.to_automaton()
        #   # print the automaton
        #   graph = dfa.to_graphviz()
        #   #graph.render("symbolicDFAs/"+formula_name)

        #   print(f'formula: {ltl_formula}')

        builder_used = None
        spot_error = None
        try:
            trans, acc, dot_source = _build_dfa_with_spot(
                ltl_formula, dictionary_symbols, formula_name
            )
            builder_used = "spot"
        except Exception as e:
            spot_error = e
            # Fallback to the legacy ltlf2dfa path
            parser = LTLfParser()
            with io.StringIO() as buf_out, io.StringIO() as buf_err, contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                ast = parser(ltl_formula)
                dot = ast.to_dfa()
            dot_source = dot
            dfa = dot2dfa(dot)
            self.alphabet = dictionary_symbols
            trans = self.reduce_dfa(dfa)
            acc = []
            for s in range(len(trans)):
                acc.append(s in dfa._final_states)
            builder_used = "ltlf2dfa"

        if builder_used == "ltlf2dfa" and spot_error is not None:
            print(
                f"[DFA] Spot translation unavailable ({spot_error}); "
                "falling back to ltlf2dfa."
            )

        self.alphabet = dictionary_symbols
        self.transitions = trans
        self.acceptance = acc
        self.num_of_states = len(self.transitions)
        self.num_of_symbols = len(dictionary_symbols)

        # Complete missing transitions so the DFA is total over the given alphabet
        for s in range(self.num_of_states):
            if s not in self.transitions:
                self.transitions[s] = {}
            for sym in range(self.num_of_symbols):
                if sym not in self.transitions[s]:
                    self.transitions[s][sym] = s

        # Save the symbolic DOT (before the end hack)
        try:
            _write_dot(dot_source, "symbolicDFAs/" + formula_name + ".dot")
            graph = Source(dot_source)
            graph.render("symbolicDFAs/" + formula_name)
        except Exception:
            # keep going even if rendering fails (e.g., on headless systems)
            pass

        # Make sure the directory exists
        os.makedirs("simpleDFAs", exist_ok=True)
        self.write_dot_file("simpleDFAs/{}.dot".format(formula_name))

        if USE_END_HACK:
            ########## MANAGE THE END SYMBOL
            ## add two terminal states
            end_with_succes = len(self.transitions)
            end_with_failure = end_with_succes + 1
            self.num_of_states += 2
            # 'end with succes'
            self.transitions[end_with_succes] = {}
            self.acceptance.append(True)
            # ' end with failure'
            self.transitions[end_with_failure] = {}
            self.acceptance.append(False)
            for activity in range(len(self.alphabet)):
                self.transitions[end_with_succes][activity] = end_with_succes
                if activity == len(self.alphabet) - 1:
                    self.transitions[end_with_failure][activity] = end_with_succes
                else:
                    self.transitions[end_with_failure][activity] = end_with_failure
            ## adjust 'end' transitions
            num_of_symbols = len(self.alphabet)
            for state in self.transitions.keys():
                if self.acceptance[state]:
                    self.transitions[state][num_of_symbols - 1] = end_with_succes
                else:
                    self.transitions[state][num_of_symbols - 1] = end_with_failure
            ## adjust finality
            for state in range(end_with_succes):
                self.acceptance[state] = False

        self.write_dot_file("simpleDFAs/{}_final.dot".format(formula_name))

    def reduce_dfa(self, pythomata_dfa):
        # note: I use self.alphabet[:-1] to exclude the end symbol
        dfa = pythomata_dfa

        admissible_transitions = []
        for true_sym in self.alphabet[:-1]:
            trans = {}
            for i, sym in enumerate(self.alphabet[:-1]):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)

        red_trans_funct = {}
        for s0 in dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = dfa._transition_function.get(s0, {})
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        # Complete missing transitions by self-looping, so unseen symbols don't force rejection
        num_syms = len(self.alphabet)
        for s0 in red_trans_funct:
            for sym in range(num_syms):
                if sym not in red_trans_funct[s0]:
                    red_trans_funct[s0][sym] = s0

        return red_trans_funct

    def calculate_non_failure_states(self):
        ########### MARK FAILURE STATES
        non_failure_states = set()

        for i in range(len(self.acceptance)):
            if self.acceptance[i] == True:
                non_failure_states.add(i)
        fix_point = False
        while not fix_point:
            old_non_failure = non_failure_states.copy()
            for state in self.transitions:
                for symbol in self.transitions[state]:
                    if self.transitions[state][symbol] in non_failure_states:
                        non_failure_states.add(state)
            fix_point = old_non_failure == non_failure_states

        return non_failure_states

    def init_from_transacc(self, trans, acc):
        self.num_of_states = len(acc)
        self.num_of_symbols = len(trans[0])
        self.transitions = trans
        self.acceptance = acc

        self.alphabet = []
        for a in range(self.num_of_symbols):
            self.alphabet.append(a)

    def random_init(self, numb_of_states, numb_of_symbols):
        # print(f'num of states: {numb_of_states}')
        self.num_of_states = numb_of_states
        self.num_of_symbols = numb_of_symbols
        transitions = {}
        acceptance = []
        for s in range(numb_of_states):
            trans_from_s = {}
            # Each state is equiprobably set to be accepting or rejecting
            acceptance.append(bool(random.randrange(2)))
            # evenly choose another state from [i + 1; N ] and adds a random-labeled transition
            if s < numb_of_states - 1:
                s_prime = random.randrange(s + 1, numb_of_states)
                a_start = random.randrange(numb_of_symbols)

                trans_from_s[a_start] = s_prime
            else:
                a_start = None
            for a in range(numb_of_symbols):
                # a = str(a)
                if a != a_start:
                    trans_from_s[a] = random.randrange(numb_of_states)
            transitions[s] = trans_from_s.copy()

        self.transitions = transitions
        self.acceptance = acceptance
        self.alphabet = ""
        for a in range(numb_of_symbols):
            self.alphabet += str(a)

    def accepts(self, string):
        if string == "":
            return self.acceptance[0]
        return self.accepts_from_state(0, string)

    def accepts_from_state(self, state, string):
        assert string != ""

        a = string[0]
        next_state = self.transitions[state][a]

        if len(string) == 1:
            return self.acceptance[next_state]

        return self.accepts_from_state(next_state, string[1:])

    def to_pythomata(self):
        trans = self.transitions
        acc = self.acceptance
        # print("acceptance:", acc)
        accepting_states = set()
        for i in range(len(acc)):
            if acc[i]:
                accepting_states.add(i)

        automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

        return automaton

    def write_dot_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(
                'digraph MONA_DFA {\nrankdir = LR;\ncenter = true;\nsize = "7.5,10.5";\nedge [fontname = Courier];\nnode [height = .5, width = .5];\nnode [shape = doublecircle];'
            )
            for i, rew in enumerate(self.acceptance):
                if rew:
                    f.write(str(i) + ";")
            f.write('\nnode [shape = circle]; 0;\ninit [shape = plaintext, label = ""];\ninit -> 0;\n')

            for s in range(self.num_of_states):
                for a in range(self.num_of_symbols):
                    s_prime = self.transitions[s][a]
                    f.write('{} -> {} [label="{}"];\n'.format(s, s_prime, self.dictionary_symbols[a]))
            f.write("}\n")

        s = Source.from_file(file_name)
        # s.view()

    def return_deep_dfa(self):
        ego_dfa = DeepDFA(self.num_of_symbols, self.num_of_states, 2).to(device)
        final_states = [index for index, value in enumerate(self.acceptance) if value == True]
        ego_dfa.initFromDfa(self.transitions, final_states)
        return ego_dfa

    def return_deep_dfa_constraint(self):
        ego_dfa = DeepDFA(self.num_of_symbols, self.num_of_states, self.num_of_symbols).to(device)
        permitted_moves = {}

        # calculate good states
        good_states = self.calculate_non_failure_states()

        # calculate permitted moves
        for s in range(self.num_of_states):
            permitted_moves[s] = []
            for symbol in range(self.num_of_symbols):
                if self.transitions[s][symbol] in good_states:
                    permitted_moves[s].append(symbol)

        ego_dfa.init_constraint_dfa(self.transitions, permitted_moves)
        return ego_dfa


# all dfas must share the same dictionary symbols and num states/num symbols
def save_dfas(dfas, file_name):
    trans_arrays, acc_arrays = list(zip(*[dfa.to_arrays() for dfa in dfas]))
    dfa_dict = {
        "trans": torch.tensor(trans_arrays),
        "acc": torch.tensor(acc_arrays),
        "dictionary_symbols": dfas[0].dictionary_symbols,
    }
    torch.save(dfa_dict, file_name)


"""
def load_dfas(file_name):
    dfa_dict = torch.load(file_name)
    dictionary_symbols = dfa_dict['dictionary_symbols']
    dfas = [
        arrays_to_moore_machine(trans.numpy(), acc.numpy(), dictionary_symbols)
        for trans, acc in zip(dfa_dict['trans'], dfa_dict['acc'])
    ]
    return dfas
"""
