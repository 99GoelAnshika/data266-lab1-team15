"""Numeric metrics matched to the supplied team's evaluation protocol."""
import numpy as np


def fid_sample_space(real, generated):
    """Sample FID with an exact low-rank trace computation.

    Covariances are A.T@A and B.T@B, using the n-1 sample denominator.
    The covariance cross-term equals the nuclear norm of A@B.T.
    This avoids a numerically fragile square root of a singular 2048x2048
    covariance product when there are only 300 evaluation images.
    """
    x, y = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if min(len(x), len(y)) < 2 or x.shape[1] != y.shape[1]:
        raise ValueError('FID requires >=2 samples and matching feature dimensions')
    mx, my = x.mean(0), y.mean(0)
    a = (x - mx) / np.sqrt(len(x) - 1)
    b = (y - my) / np.sqrt(len(y) - 1)
    nuclear = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    value = np.sum((mx - my)**2) + np.sum(a*a) + np.sum(b*b) - 2*nuclear
    if not np.isfinite(value) or value < -1e-6:
        raise FloatingPointError('Invalid FID value')
    return float(max(0, value))


def normalize(features):
    x = np.asarray(features, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def class_mifid(real, generated):
    """Class-specific MiFID: mean same-index feature cosine distance.

    This reproduces Anshika's supplied cross-check; it is not the general
    memorization-informed FID definition. Source order stays fixed.
    """
    if np.asarray(real).shape != np.asarray(generated).shape:
        raise ValueError('Class MiFID requires matching sample/feature shape')
    return float(np.mean(1 - np.sum(normalize(real) * normalize(generated), axis=1)))


def kid(real, generated, subset_size=100, subsets=50, seed=266):
    x, y = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    m = min(subset_size, len(x), len(y))
    if m < 2:
        raise ValueError('KID requires >=2 samples')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        kaa, kbb, kab = (a@a.T/x.shape[1]+1)**3, (b@b.T/x.shape[1]+1)**3, (a@b.T/x.shape[1]+1)**3
        values.append((kaa.sum()-np.trace(kaa))/(m*(m-1)) +
                      (kbb.sum()-np.trace(kbb))/(m*(m-1)) - 2*kab.mean())
    return float(np.mean(values)), float(np.std(values, ddof=1))


def squared_distances(x, y):
    return np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(y*y, axis=1)[None, :] - 2*x@y.T, 0)


def generative_precision_recall(real, generated, k=3):
    x, y = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if k >= min(len(x), len(y)):
        raise ValueError('Too few images for k-neighbor precision/recall')
    xx, yy = squared_distances(x, x), squared_distances(y, y)
    np.fill_diagonal(xx, np.inf)
    np.fill_diagonal(yy, np.inf)
    rx, ry = np.partition(xx, k-1, axis=1)[:, k-1], np.partition(yy, k-1, axis=1)[:, k-1]
    xy = squared_distances(x, y)
    precision = (xy <= rx[:, None]).any(axis=0).mean()
    recall = (xy <= ry[None, :]).any(axis=1).mean()
    return float(precision), float(recall)


def agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = float(np.mean(a == b))
    labels = np.union1d(a, b)
    expected = sum(float(np.mean(a == label) * np.mean(b == label)) for label in labels)
    kappa = None if expected == 1 else float((observed - expected) / (1 - expected))
    return observed, kappa
