from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_cb import analyze_dataset, build_adapter_and_dfa, build_dataset, get_arg_parser


def main(argv=None):
    args = get_arg_parser(add_help=True).parse_args(argv)
    dataset = build_dataset(args)
    adapter, _, raw_dfa = build_adapter_and_dfa(args, dataset)
    analyze_dataset(args, dataset, adapter, raw_dfa)


if __name__ == "__main__":
    main()
