"""Dataset acquisition and loading 

Every loader either returns the real dataset exactly as the manifest
(:data:`harness.config.DATASETS`) declares it, or raises.
There is deliberately no fallback of any kind: a download failure, a
parse failure, or a shape mismatch is a stop-the-world error to be
investigated, never papered over.

Sources
-------
* KEEL classification repository.  The whole-set loader reads
  ``<keel_name>.zip`` (containing ``<keel_name>.dat``); the outer CV uses
  KEEL's own published partitions, ``<keel_name>-10-fold.zip`` (or
  ``-5-fold.zip``), read by :func:`load_keel_folds`.
* UCI archive for EEG Eye State (ARFF; not present in KEEL), split by a
  seeded stratified k-fold in the runner.
* OpenML for waveform (ARFF; KEEL does not host it), split by a seeded
  stratified k-fold in the runner.

Labels are returned as integer codes (stable order of first appearance
is NOT used; codes follow the sorted unique original labels so runs are
independent of row order).
"""

import os
import re
import urllib.error
import urllib.request
import zipfile

import numpy as np

from .config import DATASETS, KEEL_FOLD_URL, KEEL_URL

#: Some mirrors reject the default urllib agent; a plain browser UA is
#: sufficient for KEEL and UCI.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _download(url, dest_path, timeout=300):
    """Download ``url`` to ``dest_path`` (atomic; hard error on failure)."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    tmp = dest_path + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise IOError("download of {0} failed with HTTP {1}".format(
                url, resp.status))
        data = resp.read()
    if not data:
        raise IOError("download of {0} returned an empty body".format(url))
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest_path)


def parse_keel_dat(text):
    """Parse a KEEL ``.dat`` file.

    Returns ``(X_raw, y_raw, attr_names, attr_is_nominal)`` where
    ``X_raw`` is an object array of the input columns as strings
    (numeric conversion is left to the caller) and ``y_raw`` the output
    column as strings.

    The KEEL header is ARFF-like: ``@attribute name type[range]`` lines,
    ``@inputs``/``@outputs`` declarations, then ``@data``.  The parser
    is strict: an output attribute other than the last declared
    attribute, or a data row of the wrong arity, is a hard error.
    """
    attr_names, attr_types = [], []
    inputs_line, outputs_line = None, None
    data_rows = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if in_data:
            data_rows.append([tok.strip() for tok in line.split(",")])
        elif low.startswith("@attribute"):
            # "@attribute NAME rest" -- NAME may be followed by a type
            # keyword (real/integer) with a range, or a {..} nominal set.
            body = line.split(None, 1)[1].strip()
            if "{" in body:
                name = body.split("{", 1)[0].strip().rstrip(",")
                attr_names.append(name)
                attr_types.append("nominal")
            else:
                parts = body.split(None, 1)
                name = parts[0].strip()
                rest = parts[1].lower() if len(parts) > 1 else "real"
                attr_names.append(name)
                if rest.startswith("integer") or rest.startswith("real"):
                    attr_types.append("numeric")
                else:
                    raise ValueError(
                        "unrecognised KEEL attribute type in line: "
                        "{0!r}".format(line))
        elif low.startswith("@inputs"):
            inputs_line = [t.strip() for t in
                           line.split(None, 1)[1].split(",")]
        elif low.startswith("@outputs") or low.startswith("@output"):
            outputs_line = [t.strip() for t in
                            line.split(None, 1)[1].split(",")]
        elif low.startswith("@data"):
            in_data = True
        elif low.startswith("@relation"):
            continue
        else:
            raise ValueError(
                "unrecognised KEEL header line: {0!r}".format(line))

    if not data_rows:
        raise ValueError("KEEL file contains no data rows")
    if outputs_line is None:
        # KEEL convention: the class is the last attribute.
        outputs_line = [attr_names[-1]]
    if len(outputs_line) != 1:
        raise ValueError(
            "expected exactly one output attribute, got {0!r}".format(
                outputs_line))
    out_idx = attr_names.index(outputs_line[0])
    if inputs_line is not None:
        expected_inputs = [n for i, n in enumerate(attr_names)
                           if i != out_idx]
        if inputs_line != expected_inputs:
            raise ValueError(
                "@inputs does not match the non-output attributes: "
                "{0!r} vs {1!r}".format(inputs_line, expected_inputs))

    arity = len(attr_names)
    for i, row in enumerate(data_rows):
        if len(row) != arity:
            raise ValueError(
                "data row {0} has {1} fields, expected {2}: {3!r}".format(
                    i, len(row), arity, row))

    data = np.asarray(data_rows, dtype=object)
    in_cols = [i for i in range(arity) if i != out_idx]
    X_raw = data[:, in_cols]
    y_raw = data[:, out_idx]
    in_names = [attr_names[i] for i in in_cols]
    in_nominal = [attr_types[i] == "nominal" for i in in_cols]
    return X_raw, y_raw, in_names, in_nominal


def _to_float(X_raw, names):
    """Strict numeric conversion; a non-numeric cell is a hard error."""
    try:
        return X_raw.astype(float)
    except ValueError:
        for j in range(X_raw.shape[1]):
            try:
                X_raw[:, j].astype(float)
            except ValueError:
                raise ValueError(
                    "column {0!r} is not numeric; if it is nominal it "
                    "must be handled by an explicit recipe, not by "
                    "silent coercion".format(names[j]))
        raise


def _encode_labels(y_raw):
    """Integer-encode labels by sorted unique value (row-order free)."""
    classes = np.unique(y_raw)
    lut = {c: i for i, c in enumerate(classes)}
    y = np.asarray([lut[v] for v in y_raw], dtype=int)
    return y, classes


def _keel_dat_path(name, data_dir):
    spec = DATASETS[name]
    return os.path.join(data_dir, spec["keel_name"] + ".dat")


def download_dataset(name, data_dir):
    """Fetch one dataset's raw file into ``data_dir`` (skip if present)."""
    spec = DATASETS[name]
    os.makedirs(data_dir, exist_ok=True)

    if spec["source"] == "keel":
        dat_path = _keel_dat_path(name, data_dir)
        if os.path.exists(dat_path):
            return dat_path
        zip_path = os.path.join(data_dir, spec["keel_name"] + ".zip")
        if not os.path.exists(zip_path):
            _download(KEEL_URL.format(keel_name=spec["keel_name"]),
                      zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist()
                       if m.endswith(spec["keel_name"] + ".dat")]
            if len(members) != 1:
                raise IOError(
                    "expected exactly one {0}.dat inside {1}, found "
                    "{2!r}".format(spec["keel_name"], zip_path, members))
            with zf.open(members[0]) as src, \
                    open(dat_path + ".part", "wb") as dst:
                dst.write(src.read())
        os.replace(dat_path + ".part", dat_path)
        return dat_path

    if spec["source"] == "uci_eeg":
        arff_path = os.path.join(data_dir, "eeg-eye-state.arff")
        if not os.path.exists(arff_path):
            _download(spec["url"], arff_path)
        return arff_path

    if spec["source"] == "openml":
        arff_path = os.path.join(data_dir, name + ".arff")
        if not os.path.exists(arff_path):
            _download(spec["url"], arff_path)
        return arff_path

    raise ValueError("unknown source {0!r} for dataset {1!r}".format(
        spec["source"], name))


def _load_openml_arff(path):
    """Load an OpenML ARFF: numeric features + a nominal class column.

    OpenML classification ARFFs put the class in the last attribute.
    Features are read as float; the class is decoded to strings for the
    shared integer-encoding step.
    """
    from scipy.io import arff as scipy_arff

    with open(path, "r") as fh:
        data, meta = scipy_arff.loadarff(fh)
    names = list(meta.names())
    feat_names, class_col = names[:-1], names[-1]
    X = np.column_stack([np.asarray(data[n], dtype=float)
                         for n in feat_names])
    y_raw = np.asarray([v.decode() if isinstance(v, bytes) else str(v)
                        for v in data[class_col]], dtype=object)
    return X, y_raw


def _load_eeg(arff_path):
    """EEG Eye State from the UCI ARFF (14 numeric inputs, binary class)."""
    from scipy.io import arff as scipy_arff

    with open(arff_path, "r") as fh:
        data, meta = scipy_arff.loadarff(fh)
    names = list(meta.names())
    class_col = names[-1]
    if class_col.lower() not in ("eyedetection", "class"):
        raise ValueError(
            "unexpected EEG class column {0!r}".format(class_col))
    feat_names = names[:-1]
    X = np.column_stack([np.asarray(data[n], dtype=float)
                         for n in feat_names])
    y_raw = np.asarray([v.decode() if isinstance(v, bytes) else str(v)
                        for v in data[class_col]], dtype=object)
    return X, y_raw


def load_dataset(name, data_dir, download=True):
    """Load one dataset; verify it against the manifest; hard-fail else.

    Returns ``(X, y, class_labels)`` with ``X`` float64, ``y`` integer
    codes, ``class_labels`` the original label of each code.
    """
    if name not in DATASETS:
        raise KeyError("unknown dataset {0!r}; manifest has {1}".format(
            name, sorted(DATASETS)))
    spec = DATASETS[name]

    if spec["source"] == "uci_eeg":
        path = (download_dataset(name, data_dir) if download
                else os.path.join(data_dir, "eeg-eye-state.arff"))
        if not os.path.exists(path):
            raise IOError("missing raw file {0}".format(path))
        X, y_raw = _load_eeg(path)
    elif spec["source"] == "openml":
        path = (download_dataset(name, data_dir) if download
                else os.path.join(data_dir, name + ".arff"))
        if not os.path.exists(path):
            raise IOError("missing raw file {0}".format(path))
        X, y_raw = _load_openml_arff(path)
    else:
        path = (download_dataset(name, data_dir) if download
                else _keel_dat_path(name, data_dir))
        if not os.path.exists(path):
            raise IOError("missing raw file {0}".format(path))
        with open(path, "r") as fh:
            text = fh.read()
        X_raw, y_raw, names, nominal = parse_keel_dat(text)
        if any(nominal):
            raise ValueError(
                "{0}: unexpected nominal input attributes {1!r} -- "
                "the roster is numeric-only".format(
                    name, [n for n, f in zip(names, nominal) if f]))
        X = _to_float(X_raw, names)

    if not np.isfinite(X).all():
        raise ValueError("{0}: non-finite values in X".format(name))

    y, class_labels = _encode_labels(y_raw)

    # Manifest verification -- every mismatch is a hard stop.
    if len(X) != spec["n"]:
        raise AssertionError(
            "{0}: {1} rows loaded but manifest declares {2}".format(
                name, len(X), spec["n"]))
    if spec["d"] is not None and X.shape[1] != spec["d"]:
        raise AssertionError(
            "{0}: {1} attributes loaded but manifest declares {2}".format(
                name, X.shape[1], spec["d"]))
    if len(class_labels) != spec["n_classes"]:
        raise AssertionError(
            "{0}: {1} classes loaded but manifest declares {2}".format(
                name, len(class_labels), spec["n_classes"]))
    return X, y, class_labels


#: Matches KEEL partition member names: ``<name>-<K>-<k>tra.dat`` /
#: ``-<k>tst.dat`` (K = total folds, k = the fold index).
_FOLD_RE = re.compile(r"-(\d+)-(\d+)t(ra|st)\.dat$", re.IGNORECASE)


def _fold_zip_path(name, data_dir, k):
    """Local path of the K=k partition archive for a dataset."""
    return os.path.join(
        data_dir, "{0}-{1}-fold.zip".format(DATASETS[name]["keel_name"], k))


def _fetch_fold_zip(name, data_dir, k):
    """Local path of the K=k partition archive, downloading if needed.

    Returns ``None`` when KEEL does not publish this fold count, so the
    caller can fall back to a different one.  KEEL signals a missing
    partition in two ways: a 404, or (for some datasets) an HTTP 200 with
    an empty body.  Both, and a 200 that is not a valid zip, are treated
    as "not available".  A genuine network failure (``URLError``)
    propagates loudly rather than being read as a missing partition.
    """
    zip_path = _fold_zip_path(name, data_dir, k)
    if os.path.exists(zip_path):
        return zip_path
    url = KEEL_FOLD_URL.format(keel_name=DATASETS[name]["keel_name"], k=k)
    try:
        _download(url, zip_path)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError:
        raise
    except OSError as exc:
        # _download raises IOError("... empty body ...") on a 200 with no
        # content -- KEEL's way of saying the partition does not exist.
        if "empty body" in str(exc):
            return None
        raise
    if not zipfile.is_zipfile(zip_path):
        # A 200 that is not a zip (e.g. an error page) is not a partition.
        os.remove(zip_path)
        return None
    return zip_path


def load_keel_folds(name, data_dir, download=True):
    """Load KEEL's pre-computed cross-validation partitions for a dataset.

    Prefers the 10-fold partition and falls back to the 5-fold one if
    KEEL does not publish 10-fold for the dataset.  Returns
    ``(folds, k, class_labels)`` where ``folds`` is a list of
    ``(X_tr, y_tr, X_te, y_te)`` (float64 features, integer-coded
    labels), one per outer fold, and ``k`` is the fold count used.

    Labels are integer-coded from the union of labels across every
    partition file, so a class absent from one test fold keeps a stable
    code.  Each fold's train + test is verified to reconstruct the full
    manifest ``n`` (the partitions tile the dataset); ``d`` and
    ``n_classes`` are checked against the manifest.  Any mismatch, or a
    missing / short partition, is a hard error -- no silent adaptation.
    """
    if name not in DATASETS:
        raise KeyError("unknown dataset {0!r}".format(name))
    spec = DATASETS[name]
    if spec["source"] != "keel":
        raise ValueError(
            "load_keel_folds is only for KEEL datasets; {0!r} is source "
            "{1!r}".format(name, spec["source"]))
    os.makedirs(data_dir, exist_ok=True)

    zip_path, k = None, None
    for kk in (10, 5):
        existing = _fold_zip_path(name, data_dir, kk)
        if os.path.exists(existing):
            zip_path, k = existing, kk
            break
        if download:
            got = _fetch_fold_zip(name, data_dir, kk)
            if got is not None:
                zip_path, k = got, kk
                break
    if zip_path is None:
        raise IOError(
            "no KEEL 10-fold or 5-fold partition available for {0!r}"
            .format(name))

    tra, tst = {}, {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            m = _FOLD_RE.search(member)
            if not m or int(m.group(1)) != k:
                continue
            idx, kind = int(m.group(2)), m.group(3).lower()
            with zf.open(member) as fh:
                text = fh.read().decode("latin-1")
            (tra if kind == "ra" else tst)[idx] = text
    expect = list(range(1, k + 1))
    if sorted(tra) != expect or sorted(tst) != expect:
        raise IOError(
            "{0}: expected {1} train and {1} test partition files, found "
            "tra={2}, tst={3} in {4}".format(
                name, k, sorted(tra), sorted(tst), zip_path))

    # Parse every partition first; encode labels from the global union so
    # codes are identical across folds even when a rare class is absent
    # from a given test fold.
    parsed, all_labels = {}, set()
    for idx in expect:
        for kind, text in (("ra", tra[idx]), ("st", tst[idx])):
            X_raw, y_raw, col_names, nominal = parse_keel_dat(text)
            if any(nominal):
                raise ValueError(
                    "{0}: unexpected nominal input attributes -- the "
                    "roster is numeric-only".format(name))
            Xf = _to_float(X_raw, col_names)
            if not np.isfinite(Xf).all():
                raise ValueError(
                    "{0}: non-finite values in partition {1}{2}".format(
                        name, idx, kind))
            parsed[(idx, kind)] = (Xf, y_raw)
            all_labels.update(y_raw.tolist())

    class_labels = np.array(sorted(all_labels), dtype=object)
    lut = {c: i for i, c in enumerate(class_labels)}

    def _enc(y_raw):
        return np.asarray([lut[v] for v in y_raw], dtype=int)

    folds = []
    for idx in expect:
        X_tr, ytr = parsed[(idx, "ra")]
        X_te, yte = parsed[(idx, "st")]
        if len(X_tr) + len(X_te) != spec["n"]:
            raise AssertionError(
                "{0} fold {1}: train+test = {2} rows but manifest "
                "declares {3}".format(
                    name, idx, len(X_tr) + len(X_te), spec["n"]))
        folds.append((X_tr, _enc(ytr), X_te, _enc(yte)))

    d = folds[0][0].shape[1]
    if spec["d"] is not None and d != spec["d"]:
        raise AssertionError(
            "{0}: {1} attributes loaded but manifest declares {2}".format(
                name, d, spec["d"]))
    if len(class_labels) != spec["n_classes"]:
        raise AssertionError(
            "{0}: {1} classes loaded but manifest declares {2}".format(
                name, len(class_labels), spec["n_classes"]))
    return folds, k, class_labels


def imbalance_ratio(y):
    """Majority/minority class-size ratio (for the dataset table)."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max()) / float(counts.min())
