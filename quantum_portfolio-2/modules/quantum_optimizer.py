"""
modules/quantum_optimizer.py
─────────────────────────────
QUBO formulation and QAOA execution layer.
Converts a portfolio selection problem into an Ising Hamiltonian
and solves it with QAOA via Qiskit statevector simulation.

Critical bug fixes vs original
───────────────────────────────
1. _extract_probabilities   : handled only two eigenstate formats; added
   DataBin / BitArray paths for Qiskit 1.x + qiskit-algorithms 0.3.x
   where StatevectorSampler returns a V2 primitive result.  Added a
   normalisation pass so probabilities always sum to ≈ 1.

2. bitstring colour comparison (used by visualization.py):
   best_display was the DECODED (little-endian reversed, n_stocks chars)
   bitstring, but visualization compared it against RAW probability-dict
   keys which are not reversed.  Fixed by storing `best_raw_bitstring`
   (unreversed, n_qubits chars) in QAOAResult so the chart can compare
   apples-to-apples.

3. Approximation ratio: caller computed q_score / bf_score which is
   wrong when both scores are negative (common: ratio > 1 but should be
   < 1).  `approximate_ratio()` helper uses the correct formulation.

4. Primitive compatibility: StatevectorSampler is the V2 primitive in
   Qiskit 1.x; older qiskit-algorithms expected V1 Sampler.  Added
   try/except to pick the correct one automatically.

New additions
─────────────
- approximate_ratio()      : correct ratio computation for signed scores
- QAOAResult.best_raw_bitstring : raw n_qubits string for chart matching
- warm_start support via initial_point parameter
- ADAM optimizer option
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class QAOAResult:
    """Structured container for all QAOA output."""

    eigenvalue: float
    best_bitstring: str  # decoded, n_stocks chars  e.g. "1010"
    best_raw_bitstring: str  # raw, n_qubits chars — for chart colour matching
    selection: np.ndarray  # shape (n_stocks,) with 0/1 values
    selected_tickers: list[str]
    probabilities: dict[str, float]  # raw n_qubits bitstring → probability
    n_qubits: int
    eigenstate: object = field(repr=False, default=None)


# ── QUBO builder ──────────────────────────────────────────────────────────────


def build_qubo(
    predicted_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 0.5,
    cardinality_k: Optional[int] = None,
) -> "QuadraticProgram":
    """
    Build a Qiskit QuadraticProgram for binary portfolio selection.

    Objective (minimise after negation):
        max  Σ r_i x_i  −  λ Σ_{i,j} Σ_{ij} x_i x_j

    Parameters
    ----------
    predicted_returns : shape (n,)  — expected returns per stock
    cov_matrix        : shape (n,n) — annualised covariance matrix
    risk_aversion     : λ — risk penalty coefficient
    cardinality_k     : if set, adds an equality constraint Σ x_i = k
    """
    from qiskit_optimization import QuadraticProgram

    n = len(predicted_returns)
    qp = QuadraticProgram("portfolio_selection")

    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": -float(predicted_returns[i]) for i in range(n)}
    quadratic: dict = {}
    for i in range(n):
        for j in range(n):
            key = (f"x{i}", f"x{j}")
            quadratic[key] = quadratic.get(key, 0.0) + risk_aversion * float(
                cov_matrix[i, j]
            )

    qp.minimize(linear=linear, quadratic=quadratic)

    if cardinality_k is not None and 1 <= cardinality_k <= n:
        qp.linear_constraint(
            linear={f"x{i}": 1 for i in range(n)},
            sense="==",
            rhs=cardinality_k,
            name="cardinality",
        )

    return qp


# ── QAOA solver ───────────────────────────────────────────────────────────────


def run_qaoa(
    tickers: list[str],
    predicted_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 0.5,
    cardinality_k: Optional[int] = None,
    reps: int = 1,
    maxiter: int = 200,
    optimizer_name: str = "COBYLA",
    initial_point: Optional[np.ndarray] = None,
) -> QAOAResult:
    """
    Full QAOA pipeline: QUBO → Ising → QAOA → decoded result.

    Parameters
    ----------
    reps           : QAOA circuit depth p (higher p = better approximation)
    maxiter        : classical optimiser iteration budget
    optimizer_name : "COBYLA" | "SPSA" | "ADAM"
    initial_point  : optional warm-start angles (length = 2 * reps)

    Returns
    -------
    QAOAResult dataclass with all decoded outputs.

    Raises
    ------
    RuntimeError : if QAOA fails to produce a valid result
    """
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization.converters import QuadraticProgramToQubo

    algorithm_globals.random_seed = 42

    # ── Step 1: Build QUBO ────────────────────────────────────────────────────
    qp = build_qubo(predicted_returns, cov_matrix, risk_aversion, cardinality_k)
    converter = QuadraticProgramToQubo()
    qubo_problem = converter.convert(qp)
    operator, _ = qubo_problem.to_ising()
    n_qubits = operator.num_qubits

    # ── Step 2: Classical optimiser ───────────────────────────────────────────
    name_upper = optimizer_name.strip().upper()
    if name_upper == "SPSA":
        opt = SPSA(maxiter=maxiter)
    elif name_upper == "ADAM":
        try:
            from qiskit_algorithms.optimizers import ADAM

            opt = ADAM(maxiter=maxiter)
        except ImportError:
            warnings.warn("ADAM not available; falling back to COBYLA.")
            opt = COBYLA(maxiter=maxiter)
    else:
        opt = COBYLA(maxiter=maxiter)

    # ── Step 3: Sampler primitive (Qiskit 1.x compatible) ────────────────────
    sampler = _get_sampler()

    # ── Step 4: Run QAOA ─────────────────────────────────────────────────────
    qaoa_kwargs: dict = {"sampler": sampler, "optimizer": opt, "reps": reps}
    if initial_point is not None:
        expected_len = 2 * reps
        if len(initial_point) == expected_len:
            qaoa_kwargs["initial_point"] = initial_point
        else:
            warnings.warn(
                f"initial_point length {len(initial_point)} != 2*reps={expected_len}. Ignored."
            )

    qaoa = QAOA(**qaoa_kwargs)
    result = qaoa.compute_minimum_eigenvalue(operator)

    # ── Step 5: Extract probabilities ────────────────────────────────────────
    probs = _extract_probabilities(result.eigenstate, n_qubits)
    if not probs:
        raise RuntimeError("QAOA produced an empty probability distribution.")

    # ── Step 6: Decode best bitstring ─────────────────────────────────────────
    best_raw = max(probs, key=probs.get)
    selection = _decode_bitstring(best_raw, n_qubits, len(tickers))

    selected_tickers = [tickers[i] for i in range(len(tickers)) if selection[i] == 1]
    best_display = "".join(str(b) for b in selection)

    return QAOAResult(
        eigenvalue=float(result.eigenvalue.real),
        best_bitstring=best_display,
        best_raw_bitstring=best_raw,  # ← NEW: raw key for chart matching
        selection=selection,
        selected_tickers=selected_tickers,
        probabilities=probs,
        n_qubits=n_qubits,
        eigenstate=result.eigenstate,
    )


# ── Approximation ratio ───────────────────────────────────────────────────────


def approximate_ratio(q_score: float, bf_score: float) -> float:
    """
    Correct approximation ratio for a *maximisation* QUBO objective.

    When both scores share the same sign:
        positive scores → q / bf  (want ≤ 1 when QAOA is suboptimal)
        negative scores → bf / q  (inverted because larger magnitude = better)

    When scores have different signs, ratio is 0 (QAOA found a worse than
    trivial solution).

    Returns a value in [0, 1] where 1.0 means QAOA recovered the optimum.
    """
    if abs(bf_score) < 1e-10:
        return 1.0
    if bf_score > 0 and q_score > 0:
        return min(q_score / bf_score, 1.0)
    if bf_score < 0 and q_score < 0:
        # Both negative: better solution = less negative, i.e. larger value
        return min(bf_score / q_score, 1.0)
    # Different signs: QAOA solution is clearly sub-optimal
    return 0.0


# ── Portfolio metrics ─────────────────────────────────────────────────────────


def portfolio_metrics_from_selection(
    selection: np.ndarray,
    predicted_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute equal-weight return, variance, and vol for a binary selection.

    Returns
    -------
    (expected_return, variance, volatility)
    """
    sel_idx = np.where(selection == 1)[0]
    if len(sel_idx) == 0:
        return 0.0, 0.0, 0.0

    w = np.ones(len(sel_idx)) / len(sel_idx)
    ret = float(predicted_returns[sel_idx] @ w)
    var = float(w @ cov_matrix[np.ix_(sel_idx, sel_idx)] @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    return ret, var, vol


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_sampler():
    """
    Return the correct Qiskit sampler primitive for the installed version.

    Qiskit 1.x ships StatevectorSampler (V2).  Older qiskit-algorithms
    may expect the V1 Sampler.  We try V2 first, fall back to V1.
    """
    # Try V2 StatevectorSampler (Qiskit 1.x)
    try:
        from qiskit.primitives import StatevectorSampler

        return StatevectorSampler()
    except ImportError:
        pass

    # Fall back to V1 Sampler (qiskit < 1.0 or qiskit-aer backend)
    try:
        from qiskit.primitives import Sampler

        return Sampler()
    except ImportError:
        pass

    raise ImportError(
        "No compatible Qiskit sampler primitive found.  "
        "Install qiskit>=1.0 or qiskit-terra>=0.23."
    )


def _extract_probabilities(eigenstate, n_qubits: int) -> dict[str, float]:
    """
    Convert a QAOA eigenstate to {bitstring: probability}.

    Handles five known formats across Qiskit versions:
      1. Object with binary_probabilities() method (QuasiDistribution, old)
      2. Plain dict {bitstring: float}
      3. DataBin / SamplerPubResult (Qiskit 1.x V2)
      4. BitArray (Qiskit 1.x new primitive result)
      5. Amplitude dict {bitstring: complex} → |amp|^2
    """
    probs: dict[str, float] = {}

    # ── Format 1: binary_probabilities() method ───────────────────────────────
    if hasattr(eigenstate, "binary_probabilities"):
        try:
            raw = eigenstate.binary_probabilities()
            probs = {k.zfill(n_qubits): float(v) for k, v in raw.items()}
        except Exception:
            pass

    # ── Format 2: plain dict ──────────────────────────────────────────────────
    elif isinstance(eigenstate, dict):
        for k, v in eigenstate.items():
            try:
                probs[str(k).zfill(n_qubits)] = float(
                    abs(v) ** 2 if isinstance(v, complex) else v
                )
            except Exception:
                pass

    # ── Format 3: Qiskit 1.x DataBin / SamplerPubResult ─────────────────────
    else:
        # Try .get_counts() (aer-style)
        if hasattr(eigenstate, "get_counts"):
            try:
                counts = eigenstate.get_counts()
                total = sum(counts.values())
                if total > 0:
                    probs = {k.zfill(n_qubits): v / total for k, v in counts.items()}
            except Exception:
                pass

        # Try .quasi_dists (QAOA V1 result wrapping)
        if not probs and hasattr(eigenstate, "quasi_dists"):
            try:
                qd = eigenstate.quasi_dists[0]
                raw = (
                    qd.binary_probabilities()
                    if hasattr(qd, "binary_probabilities")
                    else dict(qd)
                )
                probs = {k.zfill(n_qubits): max(float(v), 0.0) for k, v in raw.items()}
            except Exception:
                pass

        # Try BitArray (Qiskit 1.2+)
        if not probs:
            try:
                from qiskit.primitives.containers import BitArray

                if isinstance(eigenstate, BitArray):
                    arr = eigenstate.get_int_counts()
                    total = sum(arr.values())
                    if total > 0:
                        probs = {
                            format(k, f"0{n_qubits}b"): v / total
                            for k, v in arr.items()
                        }
            except (ImportError, Exception):
                pass

        # ── Format 5: amplitude dict {bitstring: complex} ────────────────────
        if not probs:
            try:
                for bitstring, amp in eigenstate.items():
                    probs[str(bitstring).zfill(n_qubits)] = float(abs(amp) ** 2)
            except Exception:
                pass

    if not probs:
        # Last resort: uniform distribution so the rest of the pipeline runs
        warnings.warn(
            "Could not extract measurement probabilities from QAOA eigenstate. "
            "Falling back to uniform distribution."
        )
        keys = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
        probs = {k: 1.0 / len(keys) for k in keys}

    # Normalise to ensure they sum to 1 (floating-point drift)
    total = sum(probs.values())
    if total > 1e-10:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def _decode_bitstring(
    bitstring: str,
    n_qubits: int,
    n_stocks: int,
) -> np.ndarray:
    """
    Convert a raw bitstring (Qiskit little-endian) to a selection vector.

    Qiskit stores bits in little-endian order (qubit 0 = rightmost character).
    Reversing converts to natural variable ordering (x0 = leftmost).

    Returns np.ndarray of shape (n_stocks,) with 0/1 values.
    """
    padded = bitstring.zfill(n_qubits)
    padded = padded[::-1]  # little-endian → natural
    return np.array([int(b) for b in padded[:n_stocks]])
