"""Genetic-programming symbolic regression for interpretable surrogates.

The implementation is self-contained.
Expressions use protected numerical operators and sklearn estimator semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


_ARITY = {
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    "sin": 1, "cos": 1, "log": 1, "sqrt": 1, "neg": 1,
}


def _evaluate(program, x):
    kind = program[0]
    if kind == "var":
        return x[:, int(program[1])]
    if kind == "const":
        return np.full(len(x), float(program[1]))
    children = [_evaluate(child, x) for child in program[1:]]
    with np.errstate(all="ignore"):
        if kind == "add":
            value = children[0] + children[1]
        elif kind == "sub":
            value = children[0] - children[1]
        elif kind == "mul":
            value = children[0] * children[1]
        elif kind == "div":
            denominator = children[1]
            value = np.divide(
                children[0], denominator,
                out=np.array(children[0], copy=True),
                where=np.abs(denominator) > 1e-6,
            )
        elif kind == "sin":
            value = np.sin(children[0])
        elif kind == "cos":
            value = np.cos(children[0])
        elif kind == "log":
            value = np.log(np.abs(children[0]) + 1e-6)
        elif kind == "sqrt":
            value = np.sqrt(np.abs(children[0]))
        elif kind == "neg":
            value = -children[0]
        else:  # guarded again in fit for actionable configuration errors
            raise ValueError(f"Unknown symbolic operator: {kind}")
    return np.clip(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def _complexity(program):
    return 1 + sum(_complexity(child) for child in program[1:] if isinstance(child, tuple))


def _depth(program):
    children = [child for child in program[1:] if isinstance(child, tuple)]
    return 1 if not children else 1 + max(_depth(child) for child in children)


def _paths(program, prefix=()):
    paths = [prefix]
    for index, child in enumerate(program[1:], start=1):
        if isinstance(child, tuple):
            paths.extend(_paths(child, prefix + (index,)))
    return paths


def _subtree(program, path):
    node = program
    for index in path:
        node = node[index]
    return node


def _replace(program, path, replacement):
    if not path:
        return replacement
    index = path[0]
    values = list(program)
    values[index] = _replace(values[index], path[1:], replacement)
    return tuple(values)


def _terminal(rng, n_features, constant_range):
    if rng.random() < 0.72:
        return ("var", int(rng.integers(n_features)))
    lower, upper = constant_range
    return ("const", round(float(rng.uniform(lower, upper)), 6))


def _random_program(rng, n_features, functions, constant_range, depth, *, grow=True):
    if depth <= 1 or (grow and rng.random() < 0.28):
        return _terminal(rng, n_features, constant_range)
    operator = functions[int(rng.integers(len(functions)))]
    children = tuple(
        _random_program(
            rng, n_features, functions, constant_range, depth - 1, grow=grow
        )
        for _ in range(_ARITY[operator])
    )
    return (operator, *children)


def _format_program(program, feature_names):
    kind = program[0]
    if kind == "var":
        index = int(program[1])
        return str(feature_names[index]) if index < len(feature_names) else f"x{index}"
    if kind == "const":
        return f"{float(program[1]):.5g}"
    children = [_format_program(child, feature_names) for child in program[1:]]
    symbols = {"add": "+", "sub": "−", "mul": "×", "div": "/"}
    if kind in symbols:
        return f"({children[0]} {symbols[kind]} {children[1]})"
    if kind == "neg":
        return f"(−{children[0]})"
    return f"{kind}({children[0]})"


@dataclass(frozen=True)
class _Candidate:
    program: tuple
    slope: float
    intercept: float
    mse: float
    complexity: int
    fitness: float


class GeneticSymbolicRegressor(RegressorMixin, BaseEstimator):
    """Symbolic regressor evolved by tournament selection and tree genetics.

    Fitness combines training MSE and a parsimony penalty. The observed
    non-dominated accuracy/complexity frontier remains available in
    ``pareto_front_`` for scientific inspection.
    """

    is_symbolic_regressor_ = True

    def __init__(
        self,
        population_size=300,
        generations=25,
        tournament_size=7,
        max_depth=5,
        mutation_depth=3,
        crossover_rate=0.7,
        mutation_rate=0.25,
        elitism=0.05,
        parsimony_coefficient=1e-3,
        function_set=("add", "sub", "mul", "div", "sin", "cos", "log", "sqrt"),
        constant_range=(-2.0, 2.0),
        random_state=42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.mutation_depth = mutation_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.parsimony_coefficient = parsimony_coefficient
        self.function_set = function_set
        self.constant_range = constant_range
        self.random_state = random_state

    def _score(self, program, x, y):
        raw = _evaluate(program, x)
        centered = raw - np.mean(raw)
        denominator = float(np.dot(centered, centered))
        slope = 0.0 if denominator <= 1e-14 else float(np.dot(centered, y - np.mean(y)) / denominator)
        intercept = float(np.mean(y) - slope * np.mean(raw))
        prediction = slope * raw + intercept
        mse = float(np.mean((prediction - y) ** 2))
        complexity = _complexity(program)
        fitness = mse + float(self.parsimony_coefficient) * complexity
        return _Candidate(program, slope, intercept, mse, complexity, fitness)

    def _tournament(self, candidates, rng):
        size = min(max(2, int(self.tournament_size)), len(candidates))
        indices = rng.choice(len(candidates), size=size, replace=False)
        return min((candidates[int(index)] for index in indices), key=lambda item: item.fitness)

    def fit(self, X, y):
        x = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(target):
            raise ValueError("X must be 2-D and paired with a one-dimensional y.")
        if len(x) < 4:
            raise ValueError("Genetic symbolic regression requires at least four samples.")
        functions = tuple(str(name).lower() for name in self.function_set)
        unknown = sorted(set(functions) - set(_ARITY))
        if not functions or unknown:
            raise ValueError(f"Invalid symbolic function_set; unknown operators: {unknown}")
        population_size = max(10, int(self.population_size))
        generations = max(1, int(self.generations))
        maximum_depth = max(2, int(self.max_depth))
        rng = np.random.default_rng(self.random_state)
        population = [("var", index) for index in range(x.shape[1])]
        while len(population) < population_size:
            depth = int(rng.integers(2, maximum_depth + 1))
            population.append(_random_program(
                rng, x.shape[1], functions, self.constant_range, depth,
                grow=bool(rng.integers(2)),
            ))

        archive = {}
        evolution = []
        for generation in range(generations):
            candidates = [self._score(program, x, target) for program in population]
            for candidate in candidates:
                key = repr(candidate.program)
                previous = archive.get(key)
                if previous is None or candidate.mse < previous.mse:
                    archive[key] = candidate
            best = min(candidates, key=lambda item: item.fitness)
            evolution.append({
                "generation": generation,
                "best_mse": best.mse,
                "best_complexity": best.complexity,
                "mean_mse": float(np.mean([item.mse for item in candidates])),
            })
            if generation == generations - 1:
                break
            elite_count = max(1, min(population_size, int(round(float(self.elitism) * population_size))))
            next_population = [item.program for item in sorted(candidates, key=lambda item: item.fitness)[:elite_count]]
            while len(next_population) < population_size:
                parent = self._tournament(candidates, rng).program
                child = parent
                if rng.random() < float(self.crossover_rate):
                    donor = self._tournament(candidates, rng).program
                    child = _replace(
                        parent,
                        _paths(parent)[int(rng.integers(len(_paths(parent))))],
                        _subtree(donor, _paths(donor)[int(rng.integers(len(_paths(donor))))]),
                    )
                if rng.random() < float(self.mutation_rate):
                    mutation = _random_program(
                        rng, x.shape[1], functions, self.constant_range,
                        max(1, int(self.mutation_depth)), grow=True,
                    )
                    paths = _paths(child)
                    child = _replace(child, paths[int(rng.integers(len(paths)))], mutation)
                if _depth(child) > maximum_depth:
                    child = parent
                next_population.append(child)
            population = next_population

        candidates = list(archive.values())
        selected = min(candidates, key=lambda item: item.fitness)
        # Sorting by complexity and retaining strictly improving errors yields
        # the non-dominated front for the two minimization objectives.
        best_error = np.inf
        frontier = []
        for candidate in sorted(candidates, key=lambda item: (item.complexity, item.mse)):
            if candidate.mse < best_error - 1e-14:
                frontier.append(candidate)
                best_error = candidate.mse
        generic_names = [f"x{index}" for index in range(x.shape[1])]
        self.program_ = selected.program
        self.slope_ = selected.slope
        self.intercept_ = selected.intercept
        self.training_mse_ = selected.mse
        self.complexity_ = selected.complexity
        self.equation_ = self.format_equation(generic_names)
        self.pareto_front_ = [{
            "program": item.program,
            "complexity": item.complexity,
            "mse": item.mse,
            "slope": item.slope,
            "intercept": item.intercept,
            "equation": self._format_candidate(item, generic_names),
            "selected": item.program == selected.program,
        } for item in frontier]
        self.evolution_ = evolution
        self.n_features_in_ = x.shape[1]
        return self

    def _format_candidate(self, candidate, feature_names):
        expression = _format_program(candidate.program, feature_names)
        return f"{candidate.slope:.6g} × {expression} + {candidate.intercept:.6g}"

    def format_program(self, program=None, feature_names=None):
        check_is_fitted(self, "program_")
        names = list(feature_names or [f"x{index}" for index in range(self.n_features_in_)])
        return _format_program(self.program_ if program is None else program, names)

    def format_equation(self, feature_names=None):
        check_is_fitted(self, "program_")
        names = list(feature_names or [f"x{index}" for index in range(self.n_features_in_)])
        expression = _format_program(self.program_, names)
        return f"{self.slope_:.6g} × {expression} + {self.intercept_:.6g}"

    def predict(self, X):
        check_is_fitted(self, "program_")
        x = np.asarray(X, dtype=float)
        return self.slope_ * _evaluate(self.program_, x) + self.intercept_
