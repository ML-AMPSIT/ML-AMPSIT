"""JSON-configurable Bayesian hyperparameter search spaces."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real as RealNumber

from skopt.space import Categorical, Integer, Real


def _unexpected_keys(definition, allowed, *, location):
    unexpected = set(definition) - set(allowed)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Unknown tuning-space field(s) for {location}: {names}")


def _finite_number(value, *, location):
    if isinstance(value, bool) or not isinstance(value, RealNumber):
        raise ValueError(f"{location} must be numeric")
    value = float(value)
    if not (-float("inf") < value < float("inf")):
        raise ValueError(f"{location} must be finite")
    return value


def _integer(value, *, location):
    numeric = _finite_number(value, location=location)
    if numeric != int(numeric):
        raise ValueError(f"{location} must be an integer")
    return int(numeric)


def _category(value, *, location):
    if isinstance(value, list):
        return tuple(_category(item, location=location) for item in value)
    if isinstance(value, Mapping):
        raise ValueError(f"{location} categories cannot be JSON objects")
    return value


def parse_dimension(definition, *, model_key, parameter):
    """Convert one JSON dimension declaration to a scikit-optimize space."""
    location = f"tuning_spaces.{model_key}.{parameter}"
    if isinstance(definition, list):
        if not definition:
            raise ValueError(f"{location} categorical values cannot be empty")
        return Categorical([
            _category(value, location=location) for value in definition
        ])
    if not isinstance(definition, Mapping):
        raise ValueError(
            f"{location} must be a dimension object or a categorical array"
        )

    kind = str(definition.get("type", "")).strip().lower()
    if kind == "real":
        _unexpected_keys(
            definition, {"type", "low", "high", "prior", "base"},
            location=location,
        )
        if "low" not in definition or "high" not in definition:
            raise ValueError(f"{location} requires low and high")
        low = _finite_number(definition["low"], location=f"{location}.low")
        high = _finite_number(definition["high"], location=f"{location}.high")
        if low >= high:
            raise ValueError(f"{location} requires low < high")
        prior = str(definition.get("prior", "uniform"))
        if prior not in {"uniform", "log-uniform"}:
            raise ValueError(f"{location}.prior must be uniform or log-uniform")
        if prior == "log-uniform" and low <= 0:
            raise ValueError(f"{location} requires low > 0 with log-uniform prior")
        base = _finite_number(
            definition.get("base", 10), location=f"{location}.base"
        )
        if base <= 0 or base == 1:
            raise ValueError(f"{location}.base must be positive and different from 1")
        return Real(
            low, high, prior=prior, base=base,
        )
    if kind == "integer":
        _unexpected_keys(
            definition, {"type", "low", "high", "prior", "base"},
            location=location,
        )
        if "low" not in definition or "high" not in definition:
            raise ValueError(f"{location} requires low and high")
        low = _integer(definition["low"], location=f"{location}.low")
        high = _integer(definition["high"], location=f"{location}.high")
        if low > high:
            raise ValueError(f"{location} requires low <= high")
        prior = str(definition.get("prior", "uniform"))
        if prior not in {"uniform", "log-uniform"}:
            raise ValueError(f"{location}.prior must be uniform or log-uniform")
        if prior == "log-uniform" and low <= 0:
            raise ValueError(f"{location} requires low > 0 with log-uniform prior")
        base = _finite_number(
            definition.get("base", 10), location=f"{location}.base"
        )
        if base <= 0 or base == 1:
            raise ValueError(f"{location}.base must be positive and different from 1")
        return Integer(
            low, high, prior=prior, base=base,
        )
    if kind == "categorical":
        _unexpected_keys(
            definition, {"type", "values", "weights"}, location=location,
        )
        values = definition.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{location}.values must be a non-empty array")
        weights = definition.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(values):
                raise ValueError(
                    f"{location}.weights must contain one value per category"
                )
            weights = [
                _finite_number(value, location=f"{location}.weights")
                for value in weights
            ]
            if any(value < 0 for value in weights) or sum(weights) <= 0:
                raise ValueError(
                    f"{location}.weights must be non-negative with a positive sum"
                )
        return Categorical(
            [_category(value, location=location) for value in values],
            prior=weights,
        )
    raise ValueError(
        f"{location}.type must be real, integer, or categorical"
    )


def configured_search_space(model_key, config, default_space):
    """Return a model's configured replacement space or its registered default."""
    spaces = (config or {}).get("tuning_spaces")
    if spaces is None:
        return dict(default_space)
    if not isinstance(spaces, Mapping):
        raise ValueError("tuning_spaces must be a JSON object")
    if model_key not in spaces:
        return dict(default_space)
    definition = spaces[model_key]
    if not isinstance(definition, Mapping):
        raise ValueError(f"tuning_spaces.{model_key} must be a JSON object")
    return {
        parameter: parse_dimension(
            dimension, model_key=model_key, parameter=parameter,
        )
        for parameter, dimension in definition.items()
    }
