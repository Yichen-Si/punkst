"""Command-line entry point for the multiresolution eigensolver worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .eigensolver import (
    load_eigensolver_request,
    solve_eigensystem,
    write_eigensolver_result,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve and audit a punkst coarse diffusion eigensystem")
    parser.add_argument("--request", type=Path, required=True,
                        help="Path to an eigensolver request manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New result artifact directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        request = load_eigensolver_request(args.request)
        result = solve_eigensystem(request)
        manifest = write_eigensolver_result(args.output, request, result)
    except Exception as error:
        print(json.dumps({
            "status": "error", "error_type": type(error).__name__,
            "message": str(error),
        }, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "ok", "manifest": str(manifest),
        "input_fingerprint": request.fingerprint,
        "modes": len(result.eigenvalues),
        "maximum_residual": result.diagnostics["maximum_residual"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
