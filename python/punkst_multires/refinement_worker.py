"""Command-line entry point for full-data diffusion-mode refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .refinement import (
    load_refinement_request,
    solve_refinement,
    write_refinement_result,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine lifted punkst coarse diffusion modes against the full "
            "embedding operator"))
    parser.add_argument("--request", type=Path, required=True,
                        help="Path to a refinement request manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New refinement result artifact directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        request = load_refinement_request(args.request)
        result = solve_refinement(request)
        manifest = write_refinement_result(args.output, request, result)
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
        "maximum_lifted_residual":
            result.diagnostics["maximum_lifted_residual"],
        "maximum_residual": result.diagnostics["maximum_residual"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
