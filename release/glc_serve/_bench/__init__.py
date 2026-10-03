"""Benchmark and correctness-gate drivers, shipped inside the package so that
``glc-bench`` can invoke them as ``python -m glc_serve._bench.<name>`` from an installed
wheel with no checkout and no path juggling.

Each module is also usable on its own; ``--help`` on any of them documents its flags.
"""
