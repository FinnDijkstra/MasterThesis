# ============================================================
#  Unconstrained vs constrained solver for your 6-vector setup
#  - real Gram path (no complex K, no np.angle)
#  - Chebyshev / cosine recurrence (no-drift harmonics)
#  - drop-in benchmark harness
# ============================================================

import time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from functools import partial
from scipy.special import logsumexp

# ---------------------------
# Utilities: parametrizations
# ---------------------------

# ===== drag-drop helper layer for custom starts =====



def ensure_unit_columns(S0_flat, shapeStartingGuess, eps=1e-15):
    """Normalize columns of a flattened (2n, m) start so it's feasible for the constrained run."""
    X = S0_flat.reshape(*shapeStartingGuess).copy()
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    X /= np.maximum(norms, eps)
    return X.ravel()

def stereo_from_X(X, eps=1e-12):
    """
    Inverse stereographic from the pole at x0=+1.
    For each column: y = x_rest / (1 - x0). Guards near the pole.
    X: (2n, m) with unit columns.
    Returns Y: (2n-1, m).
    """
    x0 = X[0]            # (m,)
    rest = X[1:, :]      # (2n-1, m)
    denom = 1.0 - x0     # (m,)
    Y = np.empty_like(rest)
    mask = denom > eps
    Y[:, mask] = rest[:, mask] / denom[mask]
    # Near the pole, fall back to a very large parameter pointing in 'rest' direction
    # (rare unless a column ≈ e1). This keeps things finite.
    if np.any(~mask):
        scale = 1.0 / eps
        # if rest is ~0 too, just put zero
        mag = np.linalg.norm(rest[:, ~mask], axis=0, keepdims=True)
        dirn = np.divide(rest[:, ~mask], np.maximum(mag, eps), out=np.zeros_like(rest[:, ~mask]), where=True)
        Y[:, ~mask] = scale * dirn
    return Y

def params_from_flat_start(S0_flat, shapeStartingGuess, mode="normalize"):
    """
    Convert your flattened (2n, m) start into the parameter vector
    expected by the unconstrained solve.
    - mode="normalize": params are U (2n, m) → we can reuse X itself
    - mode="stereo":    params are Y (2n-1, m) via inverse stereographic
    Returns (params0_flat, n, m)
    """
    d_tot, m = shapeStartingGuess
    n = d_tot // 2
    X = S0_flat.reshape(d_tot, m)
    if mode == "normalize":
        # Using X itself is fine; objective renormalizes inside each eval
        params0_flat = X.ravel()
    elif mode == "stereo":
        Y = stereo_from_X(X)
        params0_flat = Y.ravel()
    else:
        raise ValueError("mode must be 'normalize' or 'stereo'")
    return params0_flat, n, m

def run_from_flattened_start(flattenedStartingGuess,
                             shapeStartingGuess,
                             startingPolynomial,
                             W,
                             *,
                             unconstrained_mode="normalize",
                             unconstrained_method="L-BFGS-B",
                             workers_trust_constr=1,
                             finite_diff_rel_step=1e-6,
                             gtol=1e-6,
                             xtol=1e-12,
                             maxiter=1000,
                             verbose=0):
    """
    Convenience: run BOTH
      (1) constrained trust-constr from your flattened start, and
      (2) unconstrained from converted params (normalize/stereo).
    Returns (res_constrained, res_unconstrained)
    """
    # 1) constrained — make sure start is feasible (unit columns)
    S0_flat_feasible = ensure_unit_columns(flattenedStartingGuess, shapeStartingGuess)


    # 2) unconstrained — convert start appropriately
    params0_flat, n, m = params_from_flat_start(S0_flat_feasible, shapeStartingGuess, mode=unconstrained_mode)
    res_u = solve_unconstrained(
        params0_flat, n, m, startingPolynomial, W,
        mode=unconstrained_mode, method=unconstrained_method,
        workers=workers_trust_constr,
        finite_diff_rel_step=finite_diff_rel_step,
        gtol=gtol, xtol=xtol, maxiter=maxiter, verbose=verbose
    )
    return res_u


def stereo_to_X(Y: np.ndarray) -> np.ndarray:
    """
    Inverse stereographic projection from e1 in R^{2n}.

    Parameters
    ----------
    Y : (2n-1, m) array

    Returns
    -------
    X : (2n, m) array with unit-norm columns in R^{2n}
    """
    s = np.sum(Y * Y, axis=0)         # (m,)
    d = 1.0 + s
    x0 = (s - 1.0) / d                # (m,)
    xr = (2.0 / d) * Y                # (2n-1, m)
    return np.vstack([x0[np.newaxis, :], xr])

def real_to_complex_unit(X: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    X: (2n, m) real array with (approximately) unit columns.
    Returns Z: (n, m) complex with unit columns, Z = A + iB.
    """
    n2, m = X.shape
    n = n2 // 2
    A, B = X[:n], X[n:]
    Z = A + 1j * B
    # safeguard: renormalize to unit length (helps after FD noise)
    norms = np.linalg.norm(Z, axis=0, keepdims=True)
    Z /= np.maximum(norms, eps)
    return Z

def normalize_to_X(U: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    U: (2n, m) free real params -> X: (2n, m) unit columns
    """
    norms = np.linalg.norm(U, axis=0, keepdims=True)
    return U / np.maximum(norms, eps)

# --- public entry points you’ll call ---

def Z_from_constrained(S_flat: np.ndarray, shapeStartingGuess: tuple) -> np.ndarray:
    """
    From constrained solve (trust-constr): res.x -> complex unit vectors.
    """
    X = S_flat.reshape(*shapeStartingGuess)
    return real_to_complex_unit(X)

def Z_from_unconstrained_normalize(params_flat: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    From unconstrained 'normalize' params U_flat: (2n*m,) -> Z in C^{n x m}.
    """
    U = params_flat.reshape(2*n, m)
    X = normalize_to_X(U)
    return real_to_complex_unit(X)

def Z_from_unconstrained_stereo(params_flat: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    From unconstrained 'stereo' params Y_flat: ((2n-1)*m,) -> Z in C^{n x m}.
    """
    Y = params_flat.reshape(2*n - 1, m)
    X = stereo_to_X(Y)
    return real_to_complex_unit(X)





# --------------------------------------------
# Real Gram pieces from X = [a; b] \in R^{2n}
# --------------------------------------------

def gram_re_im_from_X(X: np.ndarray):
    """
    Given X = [A; B] with A, B in R^{n x m}, compute
      Re K = A^T A + B^T B
      Im K = A^T B - B^T A

    Parameters
    ----------
    X : (2n, m) array

    Returns
    -------
    G_re, G_im : (m, m) arrays
    """
    n2, m = X.shape
    n = n2 // 2
    A = X[:n]
    B = X[n:]
    G_re = A.T @ A + B.T @ B
    cross = A.T @ B
    G_im = cross - cross.T
    return G_re, G_im


# ------------------------------------------------------
# Radial interpolation & polynomial via no-drift cosines
# ------------------------------------------------------

def _radial_interp(values: np.ndarray, i: np.ndarray, i2: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of a 1D table 'values' at fractional indices defined by (i, i2, frac).
    Uses np.take to avoid view-chains (thread-safe under parallel FD).
    """
    v0 = np.take(values, i, mode='clip')
    v1 = np.take(values, i2, mode='clip')
    return (1.0 - frac) * v0 + frac * v1


def poly_from_gram_no_drift(G_re: np.ndarray,
                            G_im: np.ndarray,
                            startingPolynomial) -> np.ndarray:
    """
    Compute P from the Gram pieces using Chebyshev (no trig, no drift).

    startingPolynomial must have:
      - gammaMax : int
      - fastRadialEstimatorList[g].values : (resolution,) precomputed table
    """
    # radius and cos(theta)
    r = np.hypot(G_re, G_im)
    # cos theta; define as 1 where r == 0
    ct = np.ones_like(G_re)
    nz = r > 0
    ct[nz] = G_re[nz] / r[nz]

    # interpolation indices once
    res = startingPolynomial.fastRadialEstimatorList[0].values.shape[0]
    scaled = np.clip(r, 0.0, 1.0) * (res - 1)
    i = np.floor(scaled).astype(np.int64)
    i2 = np.minimum(i + 1, res - 1)
    frac = scaled - i

    # γ = 0
    out = _radial_interp(startingPolynomial.fastRadialEstimatorList[0].values, i, i2, frac)

    # higher harmonics via Chebyshev recurrence:
    # c0 = 1, c1 = cos(theta), c_{g+1} = 2*ct*c_g - c_{g-1}
    gammaMax = getattr(startingPolynomial, "gammaMax", len(startingPolynomial.fastRadialEstimatorList) - 1)
    if gammaMax >= 1:
        c0 = np.ones_like(ct)
        c1 = ct
        out += _radial_interp(startingPolynomial.fastRadialEstimatorList[1].values, i, i2, frac) * c1
        for g in range(1, gammaMax):
            c2 = 2.0 * ct * c1 - c0
            out += _radial_interp(startingPolynomial.fastRadialEstimatorList[g + 1].values, i, i2, frac) * c2
            c0, c1 = c1, c2

    return out  # (m, m), real


# --------------------------------------------
# Objectives (top-level, picklable)
# --------------------------------------------

def objective_constrained_S(S_flat: np.ndarray,
                            shapeStartingGuess: tuple,
                            startingPolynomial,
                            W_flat: np.ndarray) -> float:
    """
    Constrained baseline: S_flat reshaped to X in R^{2n x m} (unit columns enforced by constraints).
    Uses real Gram to avoid complex work.
    """
    X = S_flat.reshape(*shapeStartingGuess)
    G_re, G_im = gram_re_im_from_X(X)
    P = poly_from_gram_no_drift(G_re, G_im, startingPolynomial)
    return -np.dot(W_flat, P.ravel()).real


def objective_unconstrained_stereo(Y_flat: np.ndarray,
                                   n: int,
                                   m: int,
                                   startingPolynomial,
                                   W_flat: np.ndarray) -> float:
    """
    Unconstrained via stereographic params Y in R^{(2n-1) x m}.
    """
    Y = Y_flat.reshape(2 * n - 1, m)
    X = stereo_to_X(Y)  # (2n, m)
    G_re, G_im = gram_re_im_from_X(X)
    P = poly_from_gram_no_drift(G_re, G_im, startingPolynomial)
    return -np.dot(W_flat, P.ravel()).real


def objective_unconstrained_normalize(U_flat: np.ndarray,
                                      n: int,
                                      m: int,
                                      startingPolynomial,
                                      W_flat: np.ndarray) -> float:
    """
    Unconstrained via normalize-then-project params U in R^{(2n) x m}.
    """
    U = U_flat.reshape(2 * n, m)
    X = normalize_to_X(U)  # (2n, m)
    G_re, G_im = gram_re_im_from_X(X)
    P = poly_from_gram_no_drift(G_re, G_im, startingPolynomial)
    return -np.dot(W_flat, P.ravel()).real


def objective_facets_softmin(params_flat, n, m, estimator_A, W_stack, b_vec, tau):
    """
    Minimize soft-min_tau_j [ b_j - <W_j, A(V,V)> ] over V.
    params_flat: flattened U in R^{2n x m} (normalize parametrization).
    """
    U = params_flat.reshape(2*n, m)
    X = normalize_to_X(U)
    G_re, G_im = gram_re_im_from_X(X)
    P = poly_from_gram_no_drift(G_re, G_im, estimator_A)  # (m, m)

    # All facets at once: g_j = b_j - <W_j, P>
    # einsum: j,i,i -> j for Frobenius inner products
    s = b_vec - np.einsum('Kij,ij->K', W_stack, P, optimize=True)

    # soft-min (≈ min_j s_j when tau is small)
    f = -tau * logsumexp(-s / tau)
    return f


# ------------------------------------------------
# Exact unit-norm constraints (for constrained run)
# ------------------------------------------------

def c_fun(S_flat: np.ndarray, d_tot: int, m: int):
    X = S_flat.reshape(d_tot, m)
    return np.sum(X * X, axis=0) - 1.0


def c_jac(S_flat: np.ndarray, d_tot: int, m: int):
    X = S_flat.reshape(d_tot, m)
    nvars = d_tot * m
    # mapping: flat index k = i*m + j  -> row j = k % m
    rows = np.tile(np.arange(m), d_tot)
    cols = np.arange(nvars)
    data = 2.0 * X.ravel(order='C')
    return sparse.coo_matrix((data, (rows, cols)), shape=(m, nvars)).tocsr()


def c_hess(S_flat: np.ndarray, v: np.ndarray, d_tot: int, m: int):
    diag = 2.0 * np.tile(np.asarray(v), d_tot)
    return sparse.diags(diag, format='csr')


def make_norm_constraints(shapeStartingGuess: tuple) -> NonlinearConstraint:
    d_tot, m = shapeStartingGuess
    return NonlinearConstraint(
        fun=partial(c_fun, d_tot=d_tot, m=m),
        lb=0.0, ub=0.0,
        jac=partial(c_jac, d_tot=d_tot, m=m),
        hess=partial(c_hess, d_tot=d_tot, m=m),
    )


# ------------------------------
# Solver wrappers (pick & run)
# ------------------------------

def solve_facets_softmin(flat_start, shapeStartingGuess, estimator_A, W_stack, b_vec,
                         tau=None, maxiter=1000, ftol=1e-9):
    d_tot, m = shapeStartingGuess
    n = d_tot // 2

    # default tau: small fraction of current spread (robust)
    if tau is None:
        # quick pilot eval to scale tau
        U0 = flat_start.reshape(2*n, m)
        X0 = normalize_to_X(U0)
        G_re0, G_im0 = gram_re_im_from_X(X0)
        P0 = poly_from_gram_no_drift(G_re0, G_im0, estimator_A)
        s0 = b_vec - np.einsum('Kij,ij->K', W_stack, P0, optimize=True)
        rng = np.percentile(s0, 90) - np.percentile(s0, 10)
        tau = max(1e-4, 0.05 * (rng if np.isfinite(rng) and rng > 0 else 1.0))

    res = minimize(
        objective_facets_softmin,
        flat_start,
        args=(n, m, estimator_A, W_stack, b_vec, float(tau)),
        method="L-BFGS-B",
        jac=None,
        options=dict(maxiter=maxiter, ftol=ftol, maxls=50)
    )
    return res, tau

def solve_unconstrained(params0_flat: np.ndarray,
                        n: int,
                        m: int,
                        startingPolynomial,
                        W: np.ndarray,
                        *,
                        mode: str = "normalize",       # "normalize" or "stereo"
                        method: str = "L-BFGS-B",      # or "trust-constr"
                        workers: int = 1,              # used only if method="trust-constr"
                        finite_diff_rel_step: float = 1e-6,
                        gtol: float = 1e-6,
                        xtol: float = 1e-12,
                        maxiter: int = 1000,
                        verbose: int = 0):
    """
    Unconstrained solve. Choose parametrization and method.
    - "normalize" uses 2n vars/column (robust).
    - "stereo"    uses 2n-1 vars/column (no constraints; can be ill-conditioned near e1).
    """
    W_flat = np.asarray(W).ravel()

    if mode == "normalize":
        fun = objective_unconstrained_normalize
        args = (n, m, startingPolynomial, W_flat)
        shape = (2 * n, m)
    elif mode == "stereo":
        fun = objective_unconstrained_stereo
        args = (n, m, startingPolynomial, W_flat)
        shape = (2 * n - 1, m)
    else:
        raise ValueError("mode must be 'normalize' or 'stereo'")

    if method == "L-BFGS-B":
        if n < 20:
            meth = "BFGS"
            opt = dict(
                maxiter=maxiter,
                gtol=gtol,  # L-BFGS-B uses 'ftol' / 'gtol' differently; ftol governs f-convergence
                finite_diff_rel_step=finite_diff_rel_step,
                disp=bool(verbose)
            )
        else:
            meth = "L-BFGS-B"
            opt = dict(
                maxiter=maxiter,
                ftol=gtol,  # L-BFGS-B uses 'ftol' / 'gtol' differently; ftol governs f-convergence
                maxls=50,
                disp=bool(verbose)
            )
        res = minimize(
            fun, params0_flat, args=args,
            method=meth,
            jac=None,  # FD by default; no workers knob here
            options=opt
        )
        res.shape = shape  # stash for reporting
        return res

    elif method == "trust-constr":
        res = minimize(
            fun, params0_flat, args=args,
            method="trust-constr",
            jac='2-point',
            constraints=[],  # unconstrained
            options=dict(
                finite_diff_rel_step=finite_diff_rel_step,
                gtol=gtol, xtol=xtol, barrier_tol=1e-12,
                initial_tr_radius=1.0,
                maxiter=maxiter,
                verbose=verbose
            ),
            workers=workers
        )
        res.shape = shape
        return res

    else:
        raise ValueError("method must be 'L-BFGS-B' or 'trust-constr'")






# ------------------------------------------
# Benchmark harness (optional, easy to use)
# ------------------------------------------

def _randn(*shape, rng=None):
    if rng is None:
        rng = np.random.default_rng(1234)
    return rng.standard_normal(shape)


def make_feasible_S0(n: int, m: int, rng=None) -> np.ndarray:
    """
    Return S0_flat in shape (2n, m) flattened, with unit-norm columns.
    """
    X0 = _randn(2 * n, m, rng=rng)
    X0 = normalize_to_X(X0)
    return X0.ravel()


def make_params0(n: int, m: int, mode: str = "normalize", rng=None) -> np.ndarray:
    """
    Initial parameters for the unconstrained run.
    """
    if mode == "normalize":
        U0 = _randn(2 * n, m, rng=rng)
        return U0.ravel()
    elif mode == "stereo":
        Y0 = 0.1 * _randn(2 * n - 1, m, rng=rng)  # small radius around pole
        return Y0.ravel()
    else:
        raise ValueError("mode must be 'normalize' or 'stereo'")




# ---------------------------
# Example usage (comments)
# ---------------------------
# from your code:
#   startingPolynomial = FastDiskCombiEstimator(alpha=dim-2, coefficientArray=..., resolution=1000)
#   W = facetIneqs[facetIdx]         # shape (6, 6)
#   dim = n
#
# # 1) Compare performance quickly:
# res_constr, res_unconstr = compare_runs(n=dim, startingPolynomial=startingPolynomial, W=W,
#                                         m=6, use_workers_for_trust_constr=1)
#
# # 2) If you want unconstrained + trust-constr (to use workers):
# params0 = make_params0(dim, 6, mode="normalize")
# res_uc_tr = solve_unconstrained(params0, dim, 6, startingPolynomial, W,
#                                 mode="normalize", method="trust-constr",
#                                 workers=4, finite_diff_rel_step=1e-6, verbose=3)
#
# Notes:
# - If you set workers>1 for trust-constr, keep BLAS threads to 1 to avoid oversubscription
#   (e.g., OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1).
# - For the constrained run you can also set workers>1 to parallelize FD of the objective.
