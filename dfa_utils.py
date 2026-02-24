import json
import os
import warnings

from graphviz import Source


def summarize_dfa(dfa):
    num_states = int(getattr(dfa, "num_of_states", 0))
    num_symbols = int(getattr(dfa, "num_of_symbols", 0))
    symbols = list(getattr(dfa, "dictionary_symbols", []))
    acceptance = list(getattr(dfa, "acceptance", []))
    transitions = getattr(dfa, "transitions", {})

    accepting_states = [i for i, is_acc in enumerate(acceptance) if bool(is_acc)]
    sink_states = []
    for s in range(num_states):
        trans = transitions.get(s, {})
        if trans and all(int(trans.get(sym, -1)) == s for sym in range(num_symbols)):
            sink_states.append(s)

    return {
        "num_states": num_states,
        "num_symbols": num_symbols,
        "num_accepting_states": len(accepting_states),
        "accepting_states": accepting_states,
        "sink_states": sink_states,
        "has_end_symbol": "end" in symbols,
        "symbols": symbols,
    }


def dfa_to_dot(dfa):
    num_states = int(getattr(dfa, "num_of_states", 0))
    num_symbols = int(getattr(dfa, "num_of_symbols", 0))
    symbols = list(getattr(dfa, "dictionary_symbols", []))
    acceptance = list(getattr(dfa, "acceptance", []))
    transitions = getattr(dfa, "transitions", {})

    lines = [
        "digraph DFA {",
        "rankdir=LR;",
        'node [shape=circle, fontname="Courier"];',
        'init [shape=point, label=""];',
        "init -> 0;",
    ]

    for s in range(num_states):
        if s < len(acceptance) and acceptance[s]:
            lines.append(f'{s} [shape=doublecircle, label="{s}"];')
        else:
            lines.append(f'{s} [shape=circle, label="{s}"];')

    for s in range(num_states):
        for sym in range(num_symbols):
            next_state = transitions.get(s, {}).get(sym, s)
            label = symbols[sym] if sym < len(symbols) else str(sym)
            label = label.replace('"', '\\"')
            lines.append(f'{s} -> {next_state} [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


def export_dfa_artifacts(dfa, out_dir, stem="dfa", render_png=True):
    os.makedirs(out_dir, exist_ok=True)

    summary = summarize_dfa(dfa)
    summary_path = os.path.join(out_dir, f"{stem}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    dot_path = os.path.join(out_dir, f"{stem}.dot")
    dot_source = dfa_to_dot(dfa)
    with open(dot_path, "w") as f:
        f.write(dot_source)

    png_path = os.path.join(out_dir, f"{stem}.png")
    png_rendered = False
    if render_png:
        try:
            src = Source(dot_source)
            src.render(filename=os.path.join(out_dir, stem), format="png", cleanup=True)
            png_rendered = True
        except Exception as exc:  # pragma: no cover - depends on system graphviz runtime
            warnings.warn(
                f"PNG export failed for '{stem}' ({exc}). "
                f"DOT was still exported to {dot_path}.",
                RuntimeWarning,
            )

    return {
        "summary_path": summary_path,
        "dot_path": dot_path,
        "png_path": png_path if png_rendered else None,
        "summary": summary,
    }
