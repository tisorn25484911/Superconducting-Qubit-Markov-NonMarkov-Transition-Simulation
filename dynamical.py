"""
dynamical.py
============
Dynamical evolution, multi-shot measurement simulation, quantum state tomography
(log-likelihood MLE), and quantum process tomography for the open quantum system.

Solvers provided
────────────────
  evolve_lindblad()    — GKSL master equation (QuTiP mesolve)
  evolve_born_markov() — Bloch-Redfield structured bath (QuTiP brmesolve)
  evolve_mcsolve()     — Monte Carlo quantum trajectories (QuTiP mcsolve)

Measurement
───────────
  measure_state()          — single-basis projective measurement (n_shots)
  full_tomography_shots()  — all 9 Pauli-basis combinations → .txt files
  load_shots()             — reload saved shot files

Tomography
──────────
  qst_log_likelihood()  — QST via maximum-likelihood (single qubit + 2-qubit)
  qpt_chi_matrix()      — QPT χ-matrix reconstruction via process tomography

Utilities
─────────
  build_c_ops()      — Lindblad collapse operators for Q-A-E system
  gamma_c_zeno()     — Zeno scaling prediction  Γ_c = Ω²/(4γ) + Γ_0
  fit_decay_rate()   — extract effective exponential decay rate from concurrence
"""

import os
import re
import numpy as np
import qutip as qt
import pandas as pd
import scipy.optimize as opt
import scipy.linalg as la
import warnings
warnings.filterwarnings('ignore')

from state_prep import (
    Q, A, E, sx, sy, sz, s_plus, s_minus,
    H_for_simulation, DEFAULT_PARAMS,
    prepare_initial_state, nm_measure,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Directory for saved measurement results
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "measurement_results"

# ──────────────────────────────────────────────────────────────────────────────
#  Lindblad collapse operators
# ──────────────────────────────────────────────────────────────────────────────
def build_c_ops(params, gamma_E=0.0):
    """
    Lindblad collapse operators for the 3-qubit Q-A-E system.

    Pure dephasing on Q and A (always present), plus optional dephasing on E
    controlled by gamma_E (rad/μs).

    Parameters
    ----------
    params  : dict   — must contain 'T2Q', 'T2A'
    gamma_E : float  — E dephasing rate.  0 = non-Markovian (no E noise).

    Returns
    -------
    list of QuTiP Qobj collapse operators
    """
    T2Q = params.get("T2Q", DEFAULT_PARAMS["T2Q"])
    T2A = params.get("T2A", DEFAULT_PARAMS["T2A"])
    ops = [
        np.sqrt(1.0 / T2Q) * Q(sz),   # Q pure dephasing
        np.sqrt(1.0 / T2A) * A(sz),   # A pure dephasing
    ]
    if gamma_E > 0.0:
        ops.append(np.sqrt(gamma_E) * E(sz))
    return ops


# ──────────────────────────────────────────────────────────────────────────────
#  Solver 1: Lindblad master equation  (mesolve)
# ──────────────────────────────────────────────────────────────────────────────
def evolve_lindblad(params, gamma_E=0.0, tlist=None, rho0=None):
    """
    Solve the GKSL master equation via QuTiP mesolve.

    Parameters
    ----------
    params  : dict
    gamma_E : float  — E dephasing rate (rad/μs)
    tlist   : array  — time points (μs). Default: linspace(0, 10, 400)
    rho0    : Qobj   — initial 3-qubit density matrix. Default: paper's Bell⊗|0⟩_E

    Returns
    -------
    tlist  : np.ndarray
    states : list[Qobj]   — 3-qubit density matrices at each time point
    conc   : list[float]  — Q-A concurrence at each time point
    """
    if tlist is None:
        tlist = np.linspace(0, 10, 400)
    if rho0 is None:
        rho0 = prepare_initial_state(params)

    H     = H_for_simulation(params)
    c_ops = build_c_ops(params, gamma_E)

    result = qt.mesolve(
        H=H, rho0=rho0, tlist=tlist,
        c_ops=c_ops, e_ops=[],
        options={'nsteps': 8000, 'atol': 1e-9, 'rtol': 1e-7},
    )
    conc = [qt.concurrence(s.ptrace([0, 1])) for s in result.states]
    return np.array(tlist), result.states, conc


# ──────────────────────────────────────────────────────────────────────────────
#  Solver 2: Bloch-Redfield  (Born-Markov, structured bath)
# ──────────────────────────────────────────────────────────────────────────────
def evolve_born_markov(params, spectral_density=None, tlist=None, rho0=None):
    """
    Bloch-Redfield master equation via QuTiP brmesolve.
    Goes beyond GKSL by retaining frequency-dependent bath coupling
    (no flat-spectrum assumption) — appropriate for transmon 1/f noise.

    Parameters
    ----------
    params           : dict
    spectral_density : callable(ω) → float   — bath spectral function S(ω).
                       Default: Ohmic with cutoff calibrated to match T2Q.
    tlist            : array
    rho0             : Qobj

    Returns
    -------
    tlist, states, conc
    """
    if tlist is None:
        tlist = np.linspace(0, 10, 400)
    if rho0 is None:
        rho0 = prepare_initial_state(params)

    if spectral_density is None:
        T2Q     = params.get("T2Q", DEFAULT_PARAMS["T2Q"])
        alpha   = 1.0 / (2.0 * np.pi * T2Q)
        omega_c = 2.0 * np.pi * 5.0    # 5 GHz cutoff
        kBT     = 0.04                  # ~20 mK

        def spectral_density(omega):
            if abs(omega) < 1e-12:
                return alpha * omega_c * kBT
            S = alpha * abs(omega) * np.exp(-abs(omega) / omega_c)
            if omega > 0:
                n = 1.0 / (np.exp(omega / kBT) - 1.0 + 1e-12)
                return S * (n + 1.0)
            else:
                n = 1.0 / (np.exp(-omega / kBT) - 1.0 + 1e-12)
                return S * n

    H     = H_for_simulation(params)
    a_ops = [[Q(sz), spectral_density], [A(sz), spectral_density]]

    try:
        result = qt.brmesolve(H=H, psi0=rho0, tlist=tlist,
                              a_ops=a_ops, e_ops=[])
        conc = [qt.concurrence(s.ptrace([0, 1])) for s in result.states]
        return np.array(tlist), result.states, conc
    except Exception as exc:
        print(f"[brmesolve] {exc}  — falling back to mesolve.")
        return evolve_lindblad(params, gamma_E=0.0, tlist=tlist, rho0=rho0)


# ──────────────────────────────────────────────────────────────────────────────
#  Solver 3: Monte Carlo quantum trajectories  (mcsolve)
# ──────────────────────────────────────────────────────────────────────────────
def evolve_mcsolve(params, gamma_E=0.0, tlist=None, rho0=None, ntraj=200):
    """
    Monte Carlo quantum trajectory solver.
    Each trajectory = one experimental shot (stochastic quantum jump sequence).
    Ensemble average over ntraj trajectories → density matrix.

    Parameters
    ----------
    params  : dict
    gamma_E : float
    tlist   : array
    rho0    : Qobj    — pure state ket, or density matrix (then falls back)
    ntraj   : int     — number of trajectories

    Returns
    -------
    tlist, states (list of 2-qubit Q-A rho), conc
    """
    if tlist is None:
        tlist = np.linspace(0, 10, 400)
    if rho0 is None:
        rho0 = prepare_initial_state(params)

    # mcsolve requires a ket for the initial state
    if rho0.type == 'oper':
        purity = (rho0 * rho0).tr()
        if abs(purity - 1.0) < 1e-6:
            evals, evecs = rho0.eigenstates()
            psi0 = evecs[np.argmax(np.real(evals))]
        else:
            # Mixed: fall back to mesolve
            return evolve_lindblad(params, gamma_E, tlist, rho0)
    else:
        psi0 = rho0

    H     = H_for_simulation(params)
    c_ops = build_c_ops(params, gamma_E)

    # 9 two-qubit Pauli correlators for Q-A tomography (trace over E)
    paulis = [sx, sy, sz]
    e_ops  = [qt.tensor(p1, p2, qt.qeye(2)) for p1 in paulis for p2 in paulis]

    result = qt.mcsolve(
        H=H, state=psi0, tlist=tlist,
        c_ops=c_ops, e_ops=e_ops,
        ntraj=ntraj,
        options={'nsteps': 8000, 'progress_bar': False},
    )

    conc   = []
    states = []
    for i in range(len(tlist)):
        rho_QA = _rho_from_pauli_expect(result.expect, i)
        states.append(rho_QA)
        conc.append(qt.concurrence(rho_QA))

    return np.array(tlist), states, conc


def _rho_from_pauli_expect(expect, t_idx):
    """Reconstruct 2-qubit Q-A density matrix from 9 Pauli expectation values."""
    paulis = [sx, sy, sz]
    rho    = qt.qeye([2, 2]) / 4.0
    pairs  = [(p1, p2) for p1 in paulis for p2 in paulis]
    for k, (p1, p2) in enumerate(pairs):
        rho = rho + 0.25 * float(np.real(expect[k][t_idx])) * qt.tensor(p1, p2)
    rho.dims = [[2, 2], [2, 2]]
    return rho


# ──────────────────────────────────────────────────────────────────────────────
#  Measurement simulation  (experimental-style multi-shot)
# ──────────────────────────────────────────────────────────────────────────────
def measure_state(rho_QA, basis='z', n_shots=1024, save_file=None, seed=None):
    """
    Simulate projective measurement of a 2-qubit Q-A state.

    Mimics a real experiment: each shot independently prepares the same state
    and records the outcome.  (Implemented as sampling from the exact probability
    distribution, equivalent to multinomial sampling.)

    Parameters
    ----------
    rho_QA    : Qobj   — 2-qubit density matrix
    basis     : str or list[str]   — 'x','y','z' per qubit, or single string
                  e.g. 'z' applies Z-basis to both, ['x','z'] measures Q in X, A in Z
    n_shots   : int    — number of measurement shots
    save_file : str or None   — filename within RESULTS_DIR to save outcomes (.txt)
    seed      : int or None   — random seed for reproducibility

    Returns
    -------
    outcomes : np.ndarray shape (n_shots, 2)  — ±1 outcomes per qubit
    counts   : dict   — summary (basis, n_shots, counts per outcome)
    """
    rng = np.random.default_rng(seed)
    rho_QA = rho_QA.copy()
    rho_QA.dims = [[2, 2], [2, 2]]

    if isinstance(basis, str):
        basis = [basis, basis]

    def _to_z_rotation(b):
        """Unitary that rotates the given basis into Z-basis."""
        if b == 'x':   # |±x⟩ → |0/1⟩ via Hadamard-like
            return qt.Qobj(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        elif b == 'y': # |±y⟩ → |0/1⟩ via Y-basis rotation
            return qt.Qobj(np.array([[1, -1j], [1, 1j]]) / np.sqrt(2))
        else:
            return qt.qeye(2)

    R1 = _to_z_rotation(basis[0])
    R2 = _to_z_rotation(basis[1])
    R  = qt.tensor(R1, R2)
    rho_meas = R * rho_QA * R.dag()

    probs = np.real(np.diag(rho_meas.full()))
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()

    # outcome ordering: |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3
    # QuTiP basis: basis(2,0)=|0⟩ → eigenvalue +1 of σ_z
    #              basis(2,1)=|1⟩ → eigenvalue -1 of σ_z
    outcome_map = {0: (1, 1), 1: (1, -1), 2: (-1, 1), 3: (-1, -1)}
    indices  = rng.choice(4, size=n_shots, p=probs)
    outcomes = np.array([outcome_map[idx] for idx in indices])

    counts = {
        "basis_Q" : basis[0],
        "basis_A" : basis[1],
        "n_shots" : n_shots,
        "N_pp"    : int(np.sum((outcomes[:, 0] == 1) & (outcomes[:, 1] == 1))),
        "N_pm"    : int(np.sum((outcomes[:, 0] == 1) & (outcomes[:, 1] == -1))),
        "N_mp"    : int(np.sum((outcomes[:, 0] == -1) & (outcomes[:, 1] == 1))),
        "N_mm"    : int(np.sum((outcomes[:, 0] == -1) & (outcomes[:, 1] == -1))),
    }

    if save_file is not None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, save_file)
        header = (f"# basis_Q={basis[0]} basis_A={basis[1]} n_shots={n_shots}\n"
                  f"# q_outcome a_outcome\n")
        with open(path, 'w') as f:
            f.write(header)
            for row in outcomes:
                f.write(f"{int(row[0])} {int(row[1])}\n")

    return outcomes, counts


def full_tomography_shots(rho_QA, n_shots=1024, exp_label="exp",
                          t_label="t0", seed=None):
    """
    Collect shots in all 9 Pauli-basis combinations for 2-qubit state tomography.
    Saves each basis combination to its own .txt file.

    Returns
    -------
    df : pd.DataFrame   columns = [basis_Q, basis_A, q_out, a_out]
    """
    bases  = ['x', 'y', 'z']
    frames = []
    for bQ in bases:
        for bA in bases:
            fname    = f"{exp_label}_{t_label}_{bQ}{bA}.txt"
            s, _     = measure_state(rho_QA, basis=[bQ, bA],
                                     n_shots=n_shots, save_file=fname, seed=seed)
            df_part  = pd.DataFrame(s, columns=['q_out', 'a_out'])
            df_part['basis_Q'] = bQ
            df_part['basis_A'] = bA
            frames.append(df_part)
    return pd.concat(frames, ignore_index=True)


def load_shots(filepath):
    """
    Load measurement shots from a saved .txt file produced by measure_state().

    Returns
    -------
    df : pd.DataFrame   columns = [q_out, a_out, basis_Q, basis_A]
    """
    basis_Q = 'z'; basis_A = 'z'
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') and 'basis_Q' in line:
                m = re.search(r'basis_Q=(\w)', line)
                if m: basis_Q = m.group(1)
                m = re.search(r'basis_A=(\w)', line)
                if m: basis_A = m.group(1)
    df = pd.read_csv(filepath, comment='#', sep=r'\s+',
                     header=None, names=['q_out', 'a_out'])
    df['basis_Q'] = basis_Q
    df['basis_A'] = basis_A
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Quantum State Tomography  (log-likelihood MLE)
# ──────────────────────────────────────────────────────────────────────────────
def qst_log_likelihood(df_tomo):
    """
    Maximum-likelihood quantum state tomography for a 2-qubit (Q-A) state.

    Algorithm (matching HW2.ipynb approach):
      1. For each qubit independently, maximise the log-likelihood over the
         Bloch vector (r_x, r_y, r_z) given the shot counts in each basis.
      2. Reconstruct the 2-qubit density matrix from:
           ρ = (I⊗I + Σ_i r_i^Q σ_i⊗I + Σ_j r_j^A I⊗σ_j + Σ_{ij} c_{ij} σ_i⊗σ_j) / 4
         where c_{ij} = ⟨σ_i^Q σ_j^A⟩ from the shot data.

    Parameters
    ----------
    df_tomo : pd.DataFrame   — output of full_tomography_shots()

    Returns
    -------
    rho_QA  : Qobj (2×2×2 density matrix)
    bloch_Q : np.ndarray(3)  — MLE Bloch vector of Q  [r_x, r_y, r_z]
    bloch_A : np.ndarray(3)  — MLE Bloch vector of A
    """
    def _mle_single(df, out_col, basis_col):
        """MLE Bloch vector for one qubit from its marginal shot counts."""
        counts = {}
        for b in ['x', 'y', 'z']:
            sub  = df[df[basis_col] == b][out_col].values
            Np   = int((sub == 1).sum())
            Nm   = int((sub == -1).sum())
            N    = Np + Nm
            sg   = (Np - Nm) / N if N > 0 else 0.0
            counts[b] = (Np, Nm, sg)

        def neg_log_like(params):
            rx, ry, rz = params
            if rx**2 + ry**2 + rz**2 > 1.0:
                return 1e10
            eps = 1e-15
            val = 0.0
            for b, r in [('x', rx), ('y', ry), ('z', rz)]:
                Np, Nm, _ = counts[b]
                val -= Np * np.log(max((1 + r) / 2, eps))
                val -= Nm * np.log(max((1 - r) / 2, eps))
            return val

        res = opt.minimize(neg_log_like, [0.0, 0.0, 0.0], method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000})
        return res.x

    bloch_Q = _mle_single(df_tomo, 'q_out', 'basis_Q')
    bloch_A = _mle_single(df_tomo, 'a_out', 'basis_A')

    # Two-qubit density matrix from 9 Pauli correlators
    pauli_map = {'x': sx, 'y': sy, 'z': sz}
    rho_QA    = qt.qeye([2, 2]) / 4.0

    for bQ in ['x', 'y', 'z']:
        for bA in ['x', 'y', 'z']:
            sub  = df_tomo[(df_tomo['basis_Q'] == bQ) & (df_tomo['basis_A'] == bA)]
            if len(sub) == 0:
                continue
            corr  = float((sub['q_out'].values * sub['a_out'].values).mean())
            rho_QA = rho_QA + 0.25 * corr * qt.tensor(pauli_map[bQ], pauli_map[bA])

    # Add single-qubit marginal contributions
    rx_Q, ry_Q, rz_Q = bloch_Q
    rx_A, ry_A, rz_A = bloch_A
    rho_QA = (rho_QA
              + 0.25 * rx_Q * qt.tensor(sx, qt.qeye(2))
              + 0.25 * ry_Q * qt.tensor(sy, qt.qeye(2))
              + 0.25 * rz_Q * qt.tensor(sz, qt.qeye(2))
              + 0.25 * rx_A * qt.tensor(qt.qeye(2), sx)
              + 0.25 * ry_A * qt.tensor(qt.qeye(2), sy)
              + 0.25 * rz_A * qt.tensor(qt.qeye(2), sz))

    rho_QA.dims = [[2, 2], [2, 2]]
    return rho_QA, bloch_Q, bloch_A


# ──────────────────────────────────────────────────────────────────────────────
#  Quantum Process Tomography  (χ-matrix reconstruction)
# ──────────────────────────────────────────────────────────────────────────────
def qpt_chi_matrix(channel_fn, n_input_states=4):
    """
    Quantum Process Tomography: reconstruct the χ-matrix of a single-qubit channel.

    Standard QPT protocol:
      For each input basis state ρ_j  (|0⟩, |1⟩, |+⟩, |+i⟩):
        1. Apply the channel: ρ_out = Λ(ρ_j)
        2. Solve the linear system  β · vec(χ) = vec(λ)
           where  λ_j = ρ_out  and  β encodes  Λ(ρ) = Σ_{mn} χ_{mn} E_m ρ E_n†

    Process basis: {I, X, iY, Z}   (Pauli operator set)

    Parameters
    ----------
    channel_fn : callable(rho_in: Qobj) → rho_out: Qobj
                 The channel to characterise.  Operates on a single-qubit state.
    n_input_states : int  — number of input states (4 for full QPT)

    Returns
    -------
    chi : np.ndarray (4×4, real)  — process matrix in the Pauli basis
    """
    # Input basis states
    rho_basis = [
        qt.ket2dm(qt.basis(2, 0)),                                       # |0⟩
        qt.ket2dm(qt.basis(2, 1)),                                       # |1⟩
        qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit()),            # |+⟩
        qt.ket2dm((qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()),      # |+i⟩
    ]
    # Operator basis {I, X, iY, Z}
    E_ops = [qt.qeye(2), qt.sigmax(),
             qt.Qobj(np.array([[0, -1j], [1j, 0]])),   # iY
             qt.sigmaz()]
    n = len(E_ops)

    Lambda_vec = np.zeros(4 * n, dtype=complex)
    Beta       = np.zeros((4 * n, n * n), dtype=complex)

    for j, rho_j in enumerate(rho_basis):
        rho_out    = channel_fn(rho_j)
        rho_out_v  = rho_out.full().reshape(4)
        for k in range(4):
            Lambda_vec[j * 4 + k] = rho_out_v[k]
        for m in range(n):
            for p in range(n):
                block = (E_ops[m] * rho_j * E_ops[p].dag()).full().reshape(4)
                for k in range(4):
                    Beta[j * 4 + k, m * n + p] = block[k]

    chi_vec, _, _, _ = np.linalg.lstsq(Beta, Lambda_vec, rcond=None)
    chi = chi_vec.reshape(n, n)
    return np.real(chi)


# ──────────────────────────────────────────────────────────────────────────────
#  Zeno scaling utilities
# ──────────────────────────────────────────────────────────────────────────────
def gamma_c_zeno(gamma_E, params=None):
    """
    Zeno-regime prediction for the effective Q-A concurrence decay rate:
        Γ_c = Ω_QE² / (4 γ_E) + Γ_0
    where Γ_0 = 1/T2Q + 1/T2A.
    """
    if params is None:
        params = DEFAULT_PARAMS
    Om_QE  = params.get("Om_QE", DEFAULT_PARAMS["Om_QE"])
    T2Q    = params.get("T2Q",   DEFAULT_PARAMS["T2Q"])
    T2A    = params.get("T2A",   DEFAULT_PARAMS["T2A"])
    Gamma0 = 1.0 / T2Q + 1.0 / T2A
    return Om_QE**2 / (4.0 * gamma_E) + Gamma0


def fit_decay_rate(conc, tlist, skip_frac=0.05):
    """
    Fit log(C(t)) ~ −Γ t  to extract the effective concurrence decay rate Γ_c.

    skip_frac : fraction of initial time points to skip (transient).
    Returns np.nan if insufficient finite data.
    """
    skip  = max(1, int(skip_frac * len(tlist)))
    C     = np.maximum(conc[skip:], 1e-10)
    t     = tlist[skip:]
    valid = np.isfinite(np.log(C)) & (C > 1e-9)
    if valid.sum() < 5:
        return np.nan
    coeff = np.polyfit(t[valid], np.log(C[valid]), 1)
    return float(-coeff[0])


# ──────────────────────────────────────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=== dynamical.py self-test ===")
    params = DEFAULT_PARAMS.copy()
    tlist  = np.linspace(0, 5, 100)

    print("  Running mesolve (Exp.1 baseline)...")
    t, states, conc = evolve_lindblad(params, gamma_E=0.0, tlist=tlist)
    print(f"  C(0)={conc[0]:.4f}  C(5μs)={conc[-1]:.4f}  N={nm_measure(conc, t):.4f}")

    print("  Testing measurement simulation...")
    rho_QA = states[0].ptrace([0, 1])
    outcomes, counts = measure_state(rho_QA, basis='z', n_shots=512, seed=42)
    print(f"  n_shots={counts['n_shots']}  N++={counts['N_pp']}  N--={counts['N_mm']}")

    print("  Testing full tomography shots...")
    df_tomo = full_tomography_shots(rho_QA, n_shots=256, exp_label="selftest",
                                     t_label="t0", seed=42)
    print(f"  DataFrame shape: {df_tomo.shape}")

    print("  Running MLE QST...")
    rho_mle, bQ, bA = qst_log_likelihood(df_tomo)
    C_sim = qt.concurrence(rho_QA)
    C_mle = qt.concurrence(rho_mle)
    print(f"  C_exact={C_sim:.4f}  C_MLE={C_mle:.4f}")

    print("  Testing QPT...")
    def id_channel(rho): return rho
    chi = qpt_chi_matrix(id_channel)
    print(f"  χ[0,0] (identity channel, expect 1.0): {chi[0,0]:.4f}")

    print("=== OK ===")


if __name__ == "__main__":
    main()
