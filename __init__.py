"""
DLX-Cython: DLX (Dancing Links) implementation.

This package provides a DLX algorithm implementation
for solving exact cover problems.
"""

__version__ = "0.1.2"
__author__ = "Robert Brooks"
__email__ = "robmbrooks@gmail.com"

from dlxsolver import DLXSolver, solve_exact_cover

__all__ = ['DLXSolver', 'solve_exact_cover', '__version__']

