#!/usr/bin/env python3
"""
Factompile CLI - Entry point for the Facto compiler.

This module allows running the compiler as:
    python -m dsl_compiler program.facto
    factompile program.facto  (when installed via pip)
"""

from dsl_compiler.cli import main

if __name__ == "__main__":
    main()
