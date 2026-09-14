"""SPINE -- Skeletal Prototypes on Iterated Nerve Expansions.

SPINE is the method under test.  The construction itself is this
package's top level (the phase modules, the geometry and topology
primitives, the model); the campaign that runs it lives in
``harness``, which reaches SPINE through the same
``select(X, y, total_count=..., random_state=...)`` contract every
comparator in ``baselines`` implements.

The dependency runs one way in each direction and nowhere else:
``spine.model`` imports ``harness.base`` for the prototype
contract, and ``harness.runner`` imports ``SPINE`` to run it.

SPINE is fitted once per fold and scored two ways -- its vertices as an
ordinary prototype set (``SPINE+1NN``) and the same fitted model under
the skeleton decision rule (``SPINE+Graph``).  These are two evaluations
of one fit, and ``harness.stats`` keeps them in separate
batteries for that reason.
"""

from .model import SPINE, SkeletonClassifier

__all__ = ["SPINE", "SkeletonClassifier"]
