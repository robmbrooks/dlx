#!/usr/bin/env python3
"""
Simple example demonstrating how to use DLX in another project.

This file shows the basic usage patterns.
"""

from dlxsolver import DLXSolver, solve_exact_cover
import numpy as np


def example_basic():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    matrix = [
        [1, 0, 1],  # Row 0 covers columns 0 and 2
        [0, 1, 0],  # Row 1 covers column 1
        [1, 1, 0],  # Row 2 covers columns 0 and 1
    ]
    
    solver = DLXSolver(matrix)
    solution = solver.solve_one()
    
    print(f"Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    print(f"\nSolution: {solution}")
    
    if solution:
        # Verify solution
        coverage = np.zeros(3, dtype=np.int32)
        for row_idx in solution:
            coverage += matrix[row_idx]
        print(f"Coverage: {coverage} (should be all 1s)")
        assert np.all(coverage == 1), "Invalid solution!"


def example_all_solutions():
    """Find all solutions example."""
    print("\n" + "=" * 60)
    print("Example 2: Finding All Solutions")
    print("=" * 60)
    
    matrix = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]
    
    solver = DLXSolver(matrix)
    solutions = solver.solve(find_all=True)
    
    print(f"Found {len(solutions)} solution(s):")
    for i, solution in enumerate(solutions):
        print(f"  Solution {i+1}: rows {solution}")


def example_convenience():
    """Using the convenience function."""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Function")
    print("=" * 60)
    
    matrix = [[1, 0, 1], [0, 1, 0]]
    solutions = solve_exact_cover(matrix, find_all=True)
    
    print(f"Solutions: {solutions}")


def example_numpy():
    """Using NumPy arrays."""
    print("\n" + "=" * 60)
    print("Example 4: Using NumPy Arrays")
    print("=" * 60)
    
    matrix = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 1],
    ], dtype=np.int32)
    
    solver = DLXSolver(matrix)
    solution = solver.solve_one()
    
    print(f"Solution: {solution}")
    
    if solution:
        solution_rows = solver.get_solution_rows(solution)
        print(f"\nSolution rows:\n{solution_rows}")


if __name__ == "__main__":
    example_basic()
    example_all_solutions()
    example_convenience()
    example_numpy()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
