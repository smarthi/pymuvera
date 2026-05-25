"""Compatibility wrapper for generating the Eigenbasis README plot."""

from __future__ import annotations

from pathlib import Path

from generate_readme_plots import DEFAULT_OUTPUT_DIR, plot7_eigenbasis_spectral_bias


def main() -> None:
    output = plot7_eigenbasis_spectral_bias(DEFAULT_OUTPUT_DIR)
    print(Path(output))


if __name__ == "__main__":
    main()
