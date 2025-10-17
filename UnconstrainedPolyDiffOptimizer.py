import numpy as np
import scipy as sp

def poly_from_gram_no_drift(G_re: np.ndarray, G_im: np.ndarray, estimator) -> np.ndarray:
    """
    Evaluate a FastDiskCombiEstimator on Gram pieces without np.angle/np.cos.
    estimator must have .fastRadialEstimatorList[g].values (1D tables) and .gammaMax.
    """
    r = np.hypot(G_re, G_im)
    ct = np.ones_like(G_re)
    nz = r > 0
    ct[nz] = G_re[nz] / r[nz]

    res = estimator.fastRadialEstimatorList[0].values.shape[0]
    scaled = np.clip(r, 0.0, 1.0) * (res - 1)
    i = np.floor(scaled).astype(np.int64)
    i2 = np.minimum(i + 1, res - 1)
    frac = scaled - i

    out = _radial_interp(estimator.fastRadialEstimatorList[0].values, i, i2, frac)
    gammaMax = getattr(estimator, "gammaMax", len(estimator.fastRadialEstimatorList) - 1)
    if gammaMax >= 1:
        c0 = np.ones_like(ct); c1 = ct
        out += _radial_interp(estimator.fastRadialEstimatorList[1].values, i, i2, frac) * c1
        for g in range(1, gammaMax):
            c2 = 2.0 * ct * c1 - c0
            out += _radial_interp(estimator.fastRadialEstimatorList[g + 1].values, i, i2, frac) * c2
            c0, c1 = c1, c2
    return out


def pick_h_lambda_from_E(E_on_grid, r_grid, gamma=0.5,
                         h_min=0.02, h_max=0.15):
    """
    E_on_grid: array of E(r)^2 sampled on r_grid in [0,1]
    r_grid   : increasing grid in [0,1]
    gamma    : 0.2..0.8   (strength of diversity)
    Returns: (h, lam)
    """
    e = np.asarray(E_on_grid)  # this is E(r)^2
    Emax = float(np.max(e))
    if Emax <= 0:
        # degenerate: no signal -> pick mild defaults
        h = 0.08
        lam = gamma * (2*np.sqrt(np.pi)*h) * 1.0
        return h, lam

    # find rough peaks: local maxima vs neighbors
    # (simple robust method; replace by scipy.signal.find_peaks if you like)
    pad = np.r_[e[0], e, e[-1]]
    is_peak = (e >= pad[:-2]) & (e >= pad[2:])
    peak_idx = np.where(is_peak & (e > 0.1*Emax))[0]
    if len(peak_idx) == 0:
        peak_idx = [int(np.argmax(e))]

    # estimate width per peak: full width at half max (FWHM) ~ 2*s where e >= 0.5*peak
    widths = []
    for idx in peak_idx:
        thr = 0.5 * e[idx]
        # extend left/right to threshold
        L = idx
        while L > 0 and e[L] >= thr: L -= 1
        R = idx
        n = len(e)
        while R < n-1 and e[R] >= thr: R += 1
        w = 0.5 * (r_grid[min(R, n-1)] - r_grid[max(L, 0)])
        widths.append(max(w, 1e-6))
    w_min = float(min(widths)) if widths else 0.08

    # min separation between peaks (if multiple)
    sep = []
    if len(peak_idx) >= 2:
        for i in range(len(peak_idx)-1):
            for j in range(i+1, len(peak_idx)):
                sep.append(abs(r_grid[peak_idx[i]] - r_grid[peak_idx[j]]))
    min_sep = float(min(sep)) if sep else w_min

    # bandwidth choice
    h = 0.7 * min(w_min, 0.5*min_sep)   # eta = 0.7
    h = float(np.clip(h, h_min, h_max))

    # lambda scaled by Emax and h
    lam = float(gamma * (2.0*np.sqrt(np.pi)*h) * Emax)
    return h, lam

def normalize_to_X(U, eps=1e-15):
    norms = np.linalg.norm(U, axis=0, keepdims=True)
    return U / np.maximum(norms, eps)

def gram_re_im_from_X(X):
    n2, m = X.shape
    n = n2 // 2
    A = X[:n]; B = X[n:]
    G_re = A.T @ A + B.T @ B
    cross = A.T @ B
    G_im = cross - cross.T
    return G_re, G_im

def _radial_interp(values, i, i2, frac):
    v0 = np.take(values, i, mode='clip')
    v1 = np.take(values, i2, mode='clip')
    return (1.0 - frac) * v0 + frac * v1

# def poly_from_gram_no_drift(G_re, G_im, startingPolynomial):
#     r = np.hypot(G_re, G_im)
#     ct = np.ones_like(G_re); nz = r > 0; ct[nz] = G_re[nz] / r[nz]
#     res = startingPolynomial.fastRadialEstimatorList[0].values.shape[0]
#     scaled = np.clip(r, 0.0, 1.0) * (res - 1)
#     i = np.floor(scaled).astype(np.int64)
#     i2 = np.minimum(i + 1, res - 1)
#     frac = scaled - i
#     out = _radial_interp(startingPolynomial.fastRadialEstimatorList[0].values, i, i2, frac)
#     gammaMax = getattr(startingPolynomial, "gammaMax",
#                        len(startingPolynomial.fastRadialEstimatorList) - 1)
#     if gammaMax >= 1:
#         c0 = np.ones_like(ct); c1 = ct
#         out += _radial_interp(startingPolynomial.fastRadialEstimatorList[1].values, i, i2, frac) * c1
#         for g in range(1, gammaMax):
#             c2 = 2.0 * ct * c1 - c0
#             out += _radial_interp(startingPolynomial.fastRadialEstimatorList[g+1].values, i, i2, frac) * c2
#             c0, c1 = c1, c2
#     return out  # (m,m), real

# --- CLOSED-FORM L2 crowding penalty over r=|<vi,vj>| ---
def crowding_L2_closed_form(r_vals, h):
    """
    r_vals: (K,) with K = m(m-1)/2  magnitudes in [0,1]
    returns: int rho(r)^2 dr  with Gaussian kernel (grid-free)
    """
    K = r_vals.shape[0]
    if K == 0:
        return 0.0
    # pairwise squared diffs
    d = r_vals[:, None] - r_vals[None, :]
    S = np.exp(- (d * d) / (4.0 * h * h)).sum()
    return (1.0 / (K * K)) * (1.0 / (2.0 * np.sqrt(np.pi) * h)) * S

# --- Objective to MAXIMIZE: mean(E^2) - lam * crowd ---
def objective_diversity_normalize(params_flat,
                                  n, m,
                                  startingPolynomial_E,
                                  lam=0.2, h=0.05):
    """
    params_flat: flattened U in R^{2n x m}. We normalize columns to unit X.
    Returns NEGATIVE value (so you can minimize it):  -[gain - lam * crowd]
    """
    U = params_flat.reshape(2*n, m)
    X = normalize_to_X(U)                         # unit columns (constraint-free)
    G_re, G_im = gram_re_im_from_X(X)
    # magnitudes r (upper triangle without diag)
    r = np.hypot(G_re, G_im)
    mask = np.triu(np.ones((m, m), dtype=bool), 1)
    r_vals = r[mask].ravel()

    # E = A - A0 via your fast evaluator
    Evals = poly_from_gram_no_drift(G_re, G_im, startingPolynomial_E)
    gain = np.mean((Evals[mask] ** 2))           # p = 2

    crowd = crowding_L2_closed_form(r_vals, h)   # grid-free L2 crowding
    J = gain - lam * crowd                       # maximize J
    return -J



def real_to_complex_unit(X: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """X: (2n, m) -> Z: (n, m) complex with unit columns."""
    n2, m = X.shape
    n = n2 // 2
    Z = X[:n] + 1j * X[n:]
    norms = np.linalg.norm(Z, axis=0, keepdims=True)
    Z /= np.maximum(norms, eps)
    return Z


def Z_from_result(res_x: np.ndarray, n: int, m: int) -> np.ndarray:
    """Convert optimizer params back to complex unit vectors Z (n x m)."""
    U = res_x.reshape(2 * n, m)
    X = normalize_to_X(U)
    return real_to_complex_unit(X)


