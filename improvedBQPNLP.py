import os
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, NonlinearConstraint, Bounds

# --- (optional) set BLAS threads to physical cores before importing numpy elsewhere ---
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"

# ---------- Helpers ----------
def _S_to_Z(S, shapeStartingGuess):
    """Flattened real/imag -> complex matrix Z."""
    d_tot, m = shapeStartingGuess      # d_tot = 2*dim
    X = S.reshape(d_tot, m)
    dim = d_tot // 2
    Z = X[:dim] + 1j * X[dim:]         # shape (dim, m)
    return Z

def _radial_interp(values, i, i2, frac):
    """Linear interpolation on a precomputed radial grid for a whole (m x m) field of radii."""
    v0 = values[i]                     # broadcasted via fancy indexing
    v1 = values[i2]
    return (1.0 - frac) * v0 + frac * v1

def polyVals_fast(S, shapeStartingGuess, startingPolynomial):
    """
    Fast evaluation of your disk-poly combination on the Gram entries.
    Avoids np.angle/np.cos(γθ) by using unit phasors and a recurrence.
    """
    # Gram from Z = X[:dim] + i*X[dim:]
    Z = _S_to_Z(S, shapeStartingGuess)
    K = Z.conj().T @ Z                 # (m, m) Hermitian

    # r, unit phasor u
    r = np.abs(K)
    u = np.ones_like(K, dtype=np.complex128)
    nz = r > 0
    u[nz] = K[nz] / r[nz]              # e^{iθ}, safe at r>0

    # Precompute interpolation indices once
    # Get resolution from first FastRadialEstimator
    res = startingPolynomial.fastRadialEstimatorList[0].values.shape[0]
    scaled = np.clip(r, 0.0, 1.0) * (res - 1)
    i   = np.floor(scaled).astype(np.int64)
    i2  = np.minimum(i + 1, res - 1)
    frac = scaled - i

    # γ = 0 term
    out = _radial_interp(startingPolynomial.fastRadialEstimatorList[0].values, i, i2, frac)

    # Accumulate higher harmonics using powers of the unit phasor (no trig calls)
    phase = np.ones_like(u)
    for g in range(1, startingPolynomial.gammaMax + 1):
        phase *= u  # u^g
        radg = _radial_interp(startingPolynomial.fastRadialEstimatorList[g].values, i, i2, frac)
        out += radg * phase.real

    return out   # shape (m, m), real-valued

# ---------- Objective (drop the constant; same minimizer) ----------
# W = facetIneqs[facetIdx]  (keep dense)
def make_objective(W, shapeStartingGuess, startingPolynomial):
    W = np.asarray(W)
    def fun(S):
        P = polyVals_fast(S, shapeStartingGuess, startingPolynomial)
        # faster than sum(multiply(...)); vdot flattens and conjugates W (W is real anyway)
        return -np.dot(W.ravel(), P.ravel()).real
        # return np.dot(W.ravel(), P.ravel()).real
    return fun

# ---------- Unit-norm-per-column constraints: exact sparse Jacobian & Hessian ----------
# def make_norm_constraints(shapeStartingGuess):
#     d_tot, m = shapeStartingGuess
#     n = d_tot * m
#     rows = np.repeat(np.arange(m), d_tot)
#     cols = np.arange(n)
#
#     def c_fun(S):
#         X = S.reshape(d_tot, m)
#         return np.sum(X * X, axis=0) - 1.0     # (m,)
#
#     def c_jac(S):
#         X = S.reshape(d_tot, m)
#         data = 2.0 * X.ravel(order='C')
#         return sparse.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
#
#     def c_hess(S, v):
#         # sum_j v[j] * ∇²c_j  → diagonal with entries 2*v[col_index]
#         diag = 2.0 * np.tile(np.asarray(v), d_tot)  # length n
#         return sparse.diags(diag, format='csr')
#
#     return NonlinearConstraint(c_fun, 0.0, 0.0, jac=c_jac, hess=c_hess)

def make_norm_constraints(shapeStartingGuess):
    d_tot, m = shapeStartingGuess
    n = d_tot * m

    # map flat var index k -> column j = k % m  (C-order flatten)
    rows = np.tile(np.arange(m), d_tot)   # <-- FIX: tile, not repeat
    cols = np.arange(n)

    def c_fun(S):
        X = S.reshape(d_tot, m)           # C-order
        return np.sum(X*X, axis=0) - 1.0  # (m,)

    def c_jac(S):
        X = S.reshape(d_tot, m)           # C-order
        data = 2.0 * X.ravel(order='C')
        return sparse.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()

    def c_hess(S, v):
        # for k=i*m + j, column is j = k % m -> diag = 2*v[j]
        diag = 2.0 * np.tile(np.asarray(v), d_tot)
        return sparse.diags(diag, format='csr')

    return NonlinearConstraint(c_fun, 0.0, 0.0, jac=c_jac, hess=c_hess)

# ---------- Example wiring ----------
# Given from your context:
# - shapeStartingGuess = (2*dim, m) with m == 6
# - startingPolynomial = FastDiskCombiEstimator(...)
# - facetIneqs, facetIdx defined (W = facetIneqs[facetIdx])
# - x0 initial guess is feasible (each column has norm 1)

def solve_problem(x0, shapeStartingGuess, startingPolynomial, facetIneqs, facetIdx,
                  lb_x=None, ub_x=None):
    W = facetIneqs[facetIdx]
    fun = make_objective(W, shapeStartingGuess, startingPolynomial)
    nlc = make_norm_constraints(shapeStartingGuess)

    bounds = None
    if lb_x is not None or ub_x is not None:
        # Provide Bounds if you have simple variable-wise limits (cheap for trust-constr)
        lb = -np.inf if lb_x is None else lb_x
        ub =  np.inf if ub_x is None else ub_x
        bounds = Bounds(lb, ub)

    # Finite-diff for the objective gradient (fast & stable enough for your linear interpolant).
    # If you later supply an analytic/AD jac, drop 'jac' and pass 'jac=your_jac'.
    # options = dict(
    #     gtol=1e-6,
    #     xtol=1e-8,
    #     barrier_tol=1e-8,
    #     initial_tr_radius=1.0,
    #     maxiter=500,
    #     finite_diff_rel_step=1e-6,  # tune if needed; keep comfortably smaller than ~1/(resolution-1)
    #     verbose=0
    # )
    options= dict(
        agents=4,  # number of parallel workers
        finite_diff_rel_step=1e-6,  # keep as you tuned it
        gtol=1e-6, xtol=1e-12, barrier_tol=1e-8
    )

    res = minimize(fun, x0,
                   method="trust-constr",
                   jac='2-point',
                   constraints=[nlc],
                   bounds=bounds,
                   options=options)

    return res
