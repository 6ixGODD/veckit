"""Virtual Embryo Challenge — Task 1 evaluation metrics.

Faithful to the paper's Task-1 metric families (single-cell -> gene-expression accuracy + cell-type
composition; no spatial FGW). A prediction is a set of cells (expression matrix). Given the ground-truth
target cells, we report:
  * pseudobulk Pearson correlation                (PRIMARY, gene-expression accuracy)
  * pseudobulk Pearson correlation per cell type
  * cell-type composition base-2 Jensen-Shannon divergence, via a FROZEN probe classifier q(.)
    trained on the target embryo and applied to the PREDICTED EXPRESSION (participants submit no labels)
  * RBF-MMD distribution distance (in PCA space)
  * DEG recovery (target vs preceding stage): top-K up/down Jaccard + signed rank-agreement (Spearman)

All metrics operate on expression only (matching the submission protocol: submit expression, organizers classify).
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---- helpers ----
def to_dense(X): return X.toarray() if sparse.issparse(X) else np.asarray(X)
def pseudobulk(X): return np.asarray(to_dense(X).mean(0)).ravel()
def ct_means(X, ct):
    X = to_dense(X); return {c: X[ct == c].mean(0) for c in np.unique(ct)}

# ---- gene-expression accuracy ----
def pseudobulk_pearson(pred_X, true_X):
    return float(np.corrcoef(pseudobulk(pred_X), pseudobulk(true_X))[0, 1])

def pseudobulk_pearson_celltype(pred_X, pred_ct, true_X, true_ct):
    pm, tm = ct_means(pred_X, pred_ct), ct_means(true_X, true_ct)
    rs = [np.corrcoef(pm[c], tm[c])[0, 1] for c in pm if c in tm]
    rs = [r for r in rs if np.isfinite(r)]
    return (float(np.mean(rs)) if rs else float("nan"))

def rbf_mmd(pred_X, true_X, n=2000, seed=0):
    pred_X, true_X = to_dense(pred_X), to_dense(true_X); rng = np.random.default_rng(seed)
    Z = PCA(n_components=30, random_state=0).fit(np.vstack([pred_X, true_X]))
    A = Z.transform(pred_X[rng.choice(pred_X.shape[0], min(n, pred_X.shape[0]), replace=False)])
    B = Z.transform(true_X[rng.choice(true_X.shape[0], min(n, true_X.shape[0]), replace=False)])
    d2 = np.sum((A[:, None] - A[None, :]) ** 2, -1); gamma = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)
    Kaa, Kbb, Kab = rbf_kernel(A, A, gamma), rbf_kernel(B, B, gamma), rbf_kernel(A, B, gamma)
    return float(Kaa.mean() + Kbb.mean() - 2 * Kab.mean())

def deg_recovery(pred_X, true_X, ref_X, topk=50):
    """target-vs-preceding DEG recovery: (top-K up/down Jaccard, signed Spearman). A 'no change'
    prediction (delta==0, e.g. copy) honestly recovers no DEGs -> (0, 0)."""
    from scipy.stats import spearmanr
    d_true = pseudobulk(true_X) - pseudobulk(ref_X); d_pred = pseudobulk(pred_X) - pseudobulk(ref_X)
    if float(np.std(d_pred)) < 1e-2 * float(np.std(d_true)): return 0.0, 0.0
    top = lambda d: set(np.argsort(d)[:topk]) | set(np.argsort(d)[-topk:])
    st, sp = top(d_true), top(d_pred)
    return float(len(st & sp) / len(st | sp)), float(spearmanr(d_true, d_pred).correlation)

# ---- cell-type composition (frozen probe, official protocol) ----
def _jsd2(p, q, eps=1e-6):
    p = p + eps; q = q + eps; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(x * np.log2(x / y))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))

def train_frozen_probe(target_X, target_ct, cap=60000, seed=0):
    """A frozen cell-type classifier q(.) trained once on the target embryo (organizer side)."""
    target_X = to_dense(target_X); rng = np.random.default_rng(seed); n = target_X.shape[0]
    idx = rng.choice(n, min(cap, n), replace=False) if n > cap else np.arange(n)
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, n_jobs=-1))
    clf.fit(target_X[idx], target_ct[idx]); return clf

def composition_jsd(pred_X, probe, true_ct):
    """Composition JSD: predicted proportions = mean soft q(x_hat); observed = target label fractions."""
    classes = probe.classes_
    pi_hat = probe.predict_proba(to_dense(pred_X)).mean(0)
    u, cnt = np.unique(true_ct, return_counts=True); dd = dict(zip(u, cnt))
    pi_obs = np.array([dd.get(c, 0) for c in classes], float)
    return _jsd2(pi_obs, pi_hat)

# ---- one-call scorer ----
def score_task1(pred_X, pred_ct, true_X, true_ct, ref_X, probe=None):
    """Compute all Task-1 metrics. ref_X = the preceding stage (for DEG). Returns a dict."""
    if probe is None: probe = train_frozen_probe(true_X, true_ct)
    dj, dr = deg_recovery(pred_X, true_X, ref_X)
    return {
        "pseudobulk_pearson":          round(pseudobulk_pearson(pred_X, true_X), 4),
        "pseudobulk_pearson_celltype": round(pseudobulk_pearson_celltype(pred_X, pred_ct, true_X, true_ct), 4),
        "composition_JSD":             round(composition_jsd(pred_X, probe, true_ct), 4),
        "rbf_mmd":                     round(rbf_mmd(pred_X, true_X), 5),
        "deg_jaccard":                 round(dj, 4),
        "deg_rank_spearman":           round(dr, 4),
    }
