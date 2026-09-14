"""Per-fold evaluation metrics.

Accuracy is the primary metric and the only formally ranked one (it is
also the tuning criterion). Reduction rate does NOT carry formal
statistics -- every budget-matched method emits the anchor's count by
construction, so the rank test would be a tautology; it is written as a
descriptive table instead (see harness.stats).  Preprocessing
CPU time is a reference column, measured in the runner rather than
here.

Classifier roster: 1-NN primary; SVM and MLP at library
defaults for the "generalization across classifiers" section.  The two
stated deviations from bare defaults, both for reproducibility only:
``MLPClassifier`` receives the fold's derived ``random_state`` (its
init/shuffling is stochastic), and neither SVC nor MLP has any other
parameter set.
"""

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def make_classifier(kind, random_state=0):
    if kind == "1nn":
        return KNeighborsClassifier(n_neighbors=1)
    if kind == "svm":
        return SVC()
    if kind == "mlp":
        return MLPClassifier(random_state=random_state)
    raise ValueError("unknown classifier kind {0!r}".format(kind))


def evaluate_prototypes(X_proto, y_proto, X_test, y_test,
                        classifiers=("1nn",), random_state=0):
    """Fit each classifier on the prototype set; score on the test fold.

    Returns ``{f"acc_{kind}": ...}``.  A prototype set with a single
    class is legal (a degenerate reduction outcome): the classifier then
    predicts that class everywhere and the score records the consequence
    -- no special-casing.
    """
    out = {}
    for kind in classifiers:
        clf = make_classifier(kind, random_state=random_state)
        clf.fit(X_proto, y_proto)
        y_pred = clf.predict(X_test)
        out["acc_" + kind] = float(accuracy_score(y_test, y_pred))
    return out


def reduction_rate(n_proto, n_train):
    return 1.0 - float(n_proto) / float(n_train)
