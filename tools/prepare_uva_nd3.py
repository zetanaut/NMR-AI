#!/usr/bin/env python3
"""Rebuild the 500-bin UVA-ND3 working example from its preserved source archive."""
import argparse
import json
from pathlib import Path

from uva_nd3_data import ROOT, load_nd3, prepare_nd3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/uva-nd3.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(prepare_nd3(), indent=2, allow_nan=False) + "\n")
    _, audit = load_nd3(args.output)
    print(f"Prepared {audit['rows']} UVA-ND3 records on the exact {audit['bins']}-bin grid")


if __name__ == "__main__":
    main()
