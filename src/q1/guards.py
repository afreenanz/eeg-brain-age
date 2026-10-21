"""Leakage guards used by every evaluation loop.

A violation raises immediately, so a leaky experiment cannot silently produce numbers.
"""
from __future__ import annotations

from collections.abc import Iterable


class LeakageError(AssertionError):
    pass


def assert_disjoint(**groups: Iterable) -> None:
    """assert_disjoint(train=ids, val=ids, test=ids): no subject in two groups."""
    sets = {k: set(v) for k, v in groups.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            both = sets[a] & sets[b]
            if both:
                raise LeakageError(f"subjects in both '{a}' and '{b}': {sorted(both)[:5]}"
                                   f"{' ...' if len(both) > 5 else ''}")


def assert_no_labels(ids: Iterable, groups: dict, forbidden: Iterable[str]) -> None:
    """Ensure no subject with a forbidden label (e.g. AD) is in a training set."""
    bad = [s for s in ids if groups.get(s) in set(forbidden)]
    if bad:
        raise LeakageError(f"forbidden-label subjects in training: {bad[:5]}")
