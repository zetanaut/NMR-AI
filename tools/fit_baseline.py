#!/usr/bin/env python3
"""Fit the supplied deuteron baseline using its known n=1, 3.580 m half-wave setup.

Use --setup for another explicitly documented configuration. There is no broad,
unconstrained cable-length search: hardware tuning information is mandatory.
"""
from fit_tuned_baseline import DEFAULT_SETUP, main

if __name__ == "__main__":
    main(default_setup=DEFAULT_SETUP)
