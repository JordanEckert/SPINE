"""Shared contracts for prototype selection / generation methods.

Every method in this repository (SPINE and all baselines)
implements the same interface so the runner can treat them uniformly:

    X_proto, y_proto, indices = method.select(X_train, y_train, **params)
"""

import numpy as np

#: Sentinel index marking a synthetic (generated) prototype.
SYNTHETIC_INDEX = -1


class PrototypeSelectorBase:
    """Abstract base for all reduction methods in the study."""

    is_generator = False
    deterministic = True
    order_dependent = False

    def select(self, X_train, y_train, **params):
        """Return ``(X_proto, y_proto, indices)`` for one training fold."""
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def param_grid(self):
        """Hyperparameter grid for inner-CV tuning (empty = untuned)."""
        return {}


def allocate_per_class(y, total):
    """Split a TOTAL prototype budget across classes proportionally.

    A largest-remainder apportionment of ``total`` over the class
    frequencies.  This is the protocol's per-class budget rule: the
    runner splits the anchor's total this way for K-Means and Random,
    and LVQ3 applies it to its own matched total.  The split depends on
    the training labels only -- never on a competing method's prototype
    labels -- so no method can pass its own class allocation to a
    comparator.  Two constraints are applied in this order:

    1. every class receives at least 1 (floor), and
    2. no class receives more than its size (cap).

    If ``total < n_classes`` the floor wins and the returned counts sum
    to ``n_classes`` (documented rule: the budget floor is one prototype
    per class; the anchor's total is never below ``n_classes`` on this
    roster, so this arises only under a capped ``total``).
    Deterministic: remainder ties resolve to class order.

    Returns ``dict {class_label: count}``.
    """
    y = np.asarray(y)
    classes, sizes = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = int(min(total, sizes.sum()))

    quotas = total * sizes / float(sizes.sum())
    counts = np.floor(quotas).astype(int)
    counts = np.maximum(counts, 1)
    counts = np.minimum(counts, sizes)

    # Largest-remainder correction toward the exact total, respecting
    # the floor and cap.  Ties resolve to class order (stable argsort).
    remainders = quotas - np.floor(quotas)
    while counts.sum() < total:
        order = np.argsort(-remainders, kind="stable")
        for i in order:
            if counts[i] < sizes[i]:
                counts[i] += 1
                remainders[i] = -1.0  # spent
                break
        else:
            break  # every class capped
    while counts.sum() > total:
        order = np.argsort(remainders, kind="stable")
        moved = False
        for i in order:
            if counts[i] > 1:
                counts[i] -= 1
                remainders[i] = 2.0  # spent
                moved = True
                break
        if not moved:
            break  # every class at the floor of 1
    return {c: int(k) for c, k in zip(classes, counts)}


def validate_selection(method, X_train, y_train, X_proto, y_proto,
                       indices):
    """Hard contract checks on a method's output (no soft failures)."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_proto = np.asarray(X_proto)
    y_proto = np.asarray(y_proto)
    indices = np.asarray(indices)

    if X_proto.ndim != 2 or X_proto.shape[1] != X_train.shape[1]:
        raise AssertionError(
            "{0}: X_proto has shape {1}, expected (*, {2})".format(
                method.name, X_proto.shape, X_train.shape[1]))
    if len(X_proto) == 0:
        raise AssertionError("{0}: empty prototype set".format(method.name))
    if not (len(X_proto) == len(y_proto) == len(indices)):
        raise AssertionError(
            "{0}: length mismatch X_proto={1} y_proto={2} indices={3}"
            .format(method.name, len(X_proto), len(y_proto), len(indices)))
    if len(X_proto) > len(X_train):
        raise AssertionError(
            "{0}: more prototypes ({1}) than training rows ({2})".format(
                method.name, len(X_proto), len(X_train)))

    if method.is_generator:
        if not (indices == SYNTHETIC_INDEX).all():
            raise AssertionError(
                "{0}: generator must return the synthetic sentinel for "
                "every index".format(method.name))
    else:
        if indices.min() < 0 or indices.max() >= len(X_train):
            raise AssertionError(
                "{0}: selection indices out of range".format(method.name))
        if not np.array_equal(X_proto, X_train[indices]):
            raise AssertionError(
                "{0}: X_proto rows do not equal X_train[indices]".format(
                    method.name))
        if not np.array_equal(y_proto, y_train[indices]):
            raise AssertionError(
                "{0}: y_proto does not equal y_train[indices]".format(
                    method.name))
