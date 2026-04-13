"""
state_prep.py
=============
State preparation, Hamiltonian definitions, and frame-transformation utilities
for the Gaikwad et al. (PRL 132, 200401, 2024) simulation platform.

Qubit ordering convention: 0 = Q (system qubit), 1 = A (ancilla), 2 = E (environment)
Time unit  : microseconds (μs)
Frequency  : rad/μs   (= 2π × MHz)

Approximation degrees (params["aprox_deg"]):
  0  — Interaction picture / RWA: H = H_QA + H_QE (time-independent).
         Reproduces the paper's exact results with minimal numerical cost.
  1  — Doubly-rotating frame with finite detunings: includes small residual
         δ_i σ_z^i / 2 terms that open access to lab-frame gate definitions
         and frame transformations.  Realistic qubit frequencies are used
         (scaled to rad/μs) together with drive modulation at Δ_QA/2 and Δ_QE/4.
"""

import numpy as np
import qutip as qt
import pennylane as qml
from scipy.linalg import expm
from scipy.integrate import trapezoid

# ──────────────────────────────────────────────────────────────────────────────
#  Default physical parameters   (Table I  +  text of Gaikwad et al.)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    # Coherence times (μs)
    "T2Q"   : 39.0,
    "T2A"   : 41.0,
    # iSWAP coupling rates (rad/μs = 2π × MHz)
    "Om_QA" : 2 * np.pi * 0.477,   # Q-A  (2π × 0.477 MHz)
    "Om_QE" : 2 * np.pi * 0.473,   # Q-E  (2π × 0.473 MHz)
    # Dispersive coupling chi/2π = 200 kHz
    "chi"   : 2 * np.pi * 0.200,
    # Qubit transition frequencies for aprox_deg = 1
    # Values scaled so 1 "scaled-GHz" = 1 rad/μs (allows finite-step integration)
    # Physically these represent ~4.1 GHz transmon qubits.
    "om_Q"  : 2 * np.pi * 4.100,   # rad/μs  (scaled: 4.100 "GHz")
    "om_A"  : 2 * np.pi * 4.100,   # will be shifted by Om_QA in H_free
    "om_E"  : 2 * np.pi * 4.100,   # will be shifted by -Om_QE in H_free
    # Approximation degree selector
    "aprox_deg" : 0,
}

# ──────────────────────────────────────────────────────────────────────────────
#  Single-qubit Pauli building blocks
# ──────────────────────────────────────────────────────────────────────────────
I       = qt.qeye(2)
sx      = qt.sigmax()
sy      = qt.sigmay()
sz      = qt.sigmaz()
s_plus  = qt.sigmap()    # |0⟩⟨1|  raising
s_minus = qt.sigmam()    # |1⟩⟨0|  lowering

# ──────────────────────────────────────────────────────────────────────────────
#  Embedding operators into 3-qubit  Q ⊗ A ⊗ E  space
# ──────────────────────────────────────────────────────────────────────────────
def Q(op):   return qt.tensor(op, I, I)
def A(op):   return qt.tensor(I, op, I)
def E(op):   return qt.tensor(I, I, op)

# ──────────────────────────────────────────────────────────────────────────────
#  Hamiltonians
# ──────────────────────────────────────────────────────────────────────────────
def H_free(params):
    """
    Free (lab-frame) Hamiltonian for aprox_deg = 1.

    We set  ω_A = ω_Q + Δ_QA  and  ω_E = ω_Q − Δ_QE  so the natural
    detunings match the iSWAP coupling frequencies from the paper.

    H_0 = ω_Q σ_z^Q/2 + ω_A σ_z^A/2 + ω_E σ_z^E/2
    """
    om_Q  = params.get("om_Q",  DEFAULT_PARAMS["om_Q"])
    Om_QA = params.get("Om_QA", DEFAULT_PARAMS["Om_QA"])
    Om_QE = params.get("Om_QE", DEFAULT_PARAMS["Om_QE"])
    om_A  = om_Q + Om_QA    # detuned so Δ_QA = Om_QA
    om_E  = om_Q - Om_QE    # detuned so Δ_QE = Om_QE
    return (om_Q * Q(sz) / 2.0 +
            om_A * A(sz) / 2.0 +
            om_E * E(sz) / 2.0)


def H_QA_int(params):
    """Q-A iSWAP coupling: Ω_QA (σ+^Q σ-^A + σ-^Q σ+^A)"""
    Om = params.get("Om_QA", DEFAULT_PARAMS["Om_QA"])
    return Om * (qt.tensor(s_plus, s_minus, I) + qt.tensor(s_minus, s_plus, I))


def H_QE_int(params):
    """Q-E iSWAP coupling: Ω_QE (σ+^Q σ-^E + σ-^Q σ+^E)"""
    Om = params.get("Om_QE", DEFAULT_PARAMS["Om_QE"])
    return Om * (qt.tensor(s_plus, I, s_minus) + qt.tensor(s_minus, I, s_plus))


def H_interaction(params):
    """Pure interaction Hamiltonian (used in aprox_deg = 0, RWA / interaction picture)."""
    return H_QA_int(params) + H_QE_int(params)


def H_detuning(params):
    """
    Residual detuning terms for aprox_deg = 1 (doubly-rotating frame).

    Drive frequencies:  ω_d_QA = Δ_QA/2,   ω_d_QE = Δ_QE/4
    In the frame rotating at ω_d for each pair, the residual detuning is
    δ_i = ω_i − ω_drive projected onto each qubit.

    Concretely:
      δ_Q_QA = −Δ_QA/2,  δ_A_QA = +Δ_QA/2   (for Q-A drive)
      δ_Q_QE = −Δ_QE/4,  δ_E_QE = +Δ_QE/4   (for Q-E drive)
    Combined on Q:  δ_Q = −Δ_QA/2 − Δ_QE/4
    """
    Om_QA = params.get("Om_QA", DEFAULT_PARAMS["Om_QA"])
    Om_QE = params.get("Om_QE", DEFAULT_PARAMS["Om_QE"])
    # Drive frequencies (plan: ω_drive = Δ/2 for QA, Δ/4 for QE)
    dQ = -Om_QA / 2.0 - Om_QE / 4.0
    dA = +Om_QA / 2.0
    dE = +Om_QE / 4.0
    return (dQ * Q(sz) / 2.0 +
            dA * A(sz) / 2.0 +
            dE * E(sz) / 2.0)


def H_for_simulation(params):
    """
    Return the Hamiltonian appropriate for the chosen approximation degree.

    aprox_deg = 0 → H_int only (interaction picture, RWA — reproduces paper)
    aprox_deg = 1 → H_int + H_detuning (doubly-rotating frame with drive
                    modulation at Δ_QA/2 and Δ_QE/4; access to frame transforms)
    """
    deg = params.get("aprox_deg", 0)
    if deg == 0:
        return H_interaction(params)
    else:
        return H_interaction(params) + H_detuning(params)


# ──────────────────────────────────────────────────────────────────────────────
#  Frame transformations  (rotating ↔ stationary lab frame)
# ──────────────────────────────────────────────────────────────────────────────
def _U0(params, t):
    """
    Free-evolution unitary  U_0(t) = exp(−i H_detuning · t)  (3-qubit).
    Used to transform between the doubly-rotating frame and a reference frame
    in which the qubit operators are stationary.
    """
    Om_QA = params.get("Om_QA", DEFAULT_PARAMS["Om_QA"])
    Om_QE = params.get("Om_QE", DEFAULT_PARAMS["Om_QE"])
    dQ = -Om_QA / 2.0 - Om_QE / 4.0
    dA = +Om_QA / 2.0
    dE = +Om_QE / 4.0
    # Individual rotations exp(−i δ σ_z t / 2)
    Uq = qt.Qobj(expm(-1j * dQ * sz.full() * t / 2.0))
    Ua = qt.Qobj(expm(-1j * dA * sz.full() * t / 2.0))
    Ue = qt.Qobj(expm(-1j * dE * sz.full() * t / 2.0))
    return qt.tensor(Uq, Ua, Ue)


def to_stationary_frame(rho_rot, params, t):
    """
    Transform a 3-qubit density matrix from the rotating (simulation) frame
    to the stationary frame: ρ_stat(t) = U_0(t) ρ_rot(t) U_0†(t).

    This is only meaningful for aprox_deg = 1.
    """
    U = _U0(params, t)
    return U * rho_rot * U.dag()


def to_rotating_frame(rho_stat, params, t):
    """
    Transform from stationary frame back to the rotating (simulation) frame:
    ρ_rot(t) = U_0†(t) ρ_stat(t) U_0(t).
    """
    U = _U0(params, t)
    return U.dag() * rho_stat * U


# ──────────────────────────────────────────────────────────────────────────────
#  Initial state preparation
# ──────────────────────────────────────────────────────────────────────────────
def prepare_initial_state(params=None):
    """
    Prepare the initial 3-qubit density matrix used in the paper:
        |Ψ_init⟩ = |Ψ+⟩_{QA} ⊗ |0⟩_E

    aprox_deg = 0 : Bell state constructed directly (ideal).
    aprox_deg = 1 : Bell state prepared by evolving |10⟩_{QA}|0⟩_E under H_QA
                    for  t = π / (2 Ω_QA)  (one iSWAP quarter period), all in
                    the rotating frame.

    Returns
    -------
    rho0 : qt.Qobj  — 8×8 density matrix, dims = [[2,2,2],[2,2,2]]
    """
    if params is None:
        params = DEFAULT_PARAMS
    deg = params.get("aprox_deg", 0)

    if deg == 0:
        Psi_plus = qt.bell_state('10')          # (|10⟩ + |01⟩)/√2
        e_ground = qt.basis(2, 0)
        return qt.ket2dm(qt.tensor(Psi_plus, e_ground))
    else:
        # Start: Q excited |1⟩, A and E ground |0⟩
        psi_Q = sx * qt.basis(2, 0)             # X|0⟩ = |1⟩
        psi_A = qt.basis(2, 0)
        psi_E = qt.basis(2, 0)
        psi0  = qt.tensor(psi_Q, psi_A, psi_E)
        rho0  = qt.ket2dm(psi0)

        # Evolve under H_QA for one sqrt(iSWAP) half-period:
        #   t = π/(4Ω) → (|10⟩ − i|01⟩)/√2  Bell state (C=1)
        #   t = π/(2Ω) → full iSWAP: |10⟩ → −i|01⟩  (separable, C=0)
        Om_QA  = params.get("Om_QA", DEFAULT_PARAMS["Om_QA"])
        t_iswap = np.pi / (4.0 * Om_QA)
        H_prep  = H_QA_int(params)
        result  = qt.mesolve(H_prep, rho0, [0.0, t_iswap], [], [])
        return result.states[-1]


def prepare_bell_state_pennylane():
    """
    Prepare Bell state |Ψ+⟩ using PennyLane's sqrt(iSWAP) gate.
    Returns a 2-qubit QuTiP density matrix (Q-A only).
    """
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def _circuit():
        qml.PauliX(wires=0)                    # |00⟩ → |10⟩
        qml.IsingXY(np.pi / 2, wires=[0, 1])  # sqrt(iSWAP)
        return qml.state()

    sv  = np.array(_circuit())
    rho = np.outer(sv, sv.conj())
    return qt.Qobj(rho, dims=[[2, 2], [2, 2]])


# ──────────────────────────────────────────────────────────────────────────────
#  Single-qubit gate utilities  (for aprox_deg = 1 / lab-frame experiments)
# ──────────────────────────────────────────────────────────────────────────────
def rotation_gate(axis, angle):
    """
    R_axis(θ) = exp(−i θ/2  σ_axis)   single-qubit rotation gate.
    axis : 'x', 'y', or 'z'
    angle : rotation angle (rad)
    Returns QuTiP Qobj.
    """
    pauli = {'x': sx, 'y': sy, 'z': sz}
    if axis not in pauli:
        raise ValueError(f"axis must be 'x','y','z', got '{axis}'")
    mat = expm(-1j * angle / 2.0 * pauli[axis].full())
    return qt.Qobj(mat)


def gate_time(gate_angle, params, qubit='Q'):
    """
    Time required to rotate a qubit by `gate_angle` radians under a resonant drive
    whose Rabi rate equals the iSWAP coupling of that qubit.

    gate_time = gate_angle / Ω_coupling   (μs)
    """
    rate_map = {
        'Q': params.get("Om_QA", DEFAULT_PARAMS["Om_QA"]),
        'A': params.get("Om_QA", DEFAULT_PARAMS["Om_QA"]),
        'E': params.get("Om_QE", DEFAULT_PARAMS["Om_QE"]),
    }
    if qubit not in rate_map:
        raise ValueError(f"qubit must be 'Q','A','E', got '{qubit}'")
    return gate_angle / rate_map[qubit]


def apply_single_qubit_gate(rho_3q, gate_qobj, qubit_idx):
    """
    Apply a 2×2 single-qubit gate (QuTiP Qobj) to one qubit of a 3-qubit state.
    qubit_idx : 0 = Q, 1 = A, 2 = E
    """
    ops = [I, I, I]
    ops[qubit_idx] = gate_qobj
    U = qt.tensor(ops)
    return U * rho_3q * U.dag()


def apply_gate_in_lab_frame(rho_rot, params, t, gate_qobj, qubit_idx):
    """
    Apply a lab-frame single-qubit gate to a state that is currently in the
    rotating frame (only meaningful for aprox_deg = 1):
      1. Transform ρ_rot → ρ_stat
      2. Apply gate
      3. Transform back ρ_stat → ρ_rot
    """
    rho_stat  = to_stationary_frame(rho_rot, params, t)
    rho_gated = apply_single_qubit_gate(rho_stat, gate_qobj, qubit_idx)
    return to_rotating_frame(rho_gated, params, t)


# ──────────────────────────────────────────────────────────────────────────────
#  Standard 2-qubit gates
# ──────────────────────────────────────────────────────────────────────────────
def iSWAP_gate():
    """Full iSWAP gate (2-qubit)."""
    return qt.Qobj(np.array([
        [1,   0,   0, 0],
        [0,   0, 1j, 0],
        [0, 1j,   0, 0],
        [0,   0,   0, 1],
    ], dtype=complex), dims=[[2, 2], [2, 2]])


def sqrt_iSWAP_gate():
    """sqrt(iSWAP) gate (2-qubit)."""
    s = 1.0 / np.sqrt(2)
    return qt.Qobj(np.array([
        [1,   0,    0, 0],
        [0,   s, 1j*s, 0],
        [0, 1j*s,  s, 0],
        [0,   0,    0, 1],
    ], dtype=complex), dims=[[2, 2], [2, 2]])


# ──────────────────────────────────────────────────────────────────────────────
#  Non-Markovianity measure  (Rivas-Huelga-Plenio)
# ──────────────────────────────────────────────────────────────────────────────
def nm_measure(conc_list, tlist):
    """
    RHP non-Markovianity: N = ∫ max(dC/dt, 0) dt
    Returns N > 0 iff dynamics cannot be CPTP (= proof of non-Markovianity).
    """
    C  = np.array(conc_list)
    dt = tlist[1] - tlist[0]
    dC = np.gradient(C, dt)
    return float(trapezoid(np.where(dC > 0, dC, 0), tlist))


# ──────────────────────────────────────────────────────────────────────────────
#  Wootters concurrence (manual implementation, for reference)
# ──────────────────────────────────────────────────────────────────────────────
def concurrence_manual(rho_qobj):
    """
    C(ρ) = max(0, √λ₁ − √λ₂ − √λ₃ − √λ₄)
    where λᵢ are eigenvalues of R = ρ (σ_y⊗σ_y) ρ* (σ_y⊗σ_y) in descending order.
    Returns (C, sqrt_eigenvalues).
    """
    rho   = rho_qobj.full()
    sy_np = np.array([[0, -1j], [1j, 0]])
    sysy  = np.kron(sy_np, sy_np)
    R     = rho @ (sysy @ rho.conj() @ sysy)
    eigs  = np.sort(np.maximum(np.real(np.linalg.eigvals(R)), 0))[::-1]
    seigs = np.sqrt(eigs)
    C     = max(0.0, seigs[0] - seigs[1] - seigs[2] - seigs[3])
    return C, seigs


# ──────────────────────────────────────────────────────────────────────────────
#  Self-test (run when executed directly)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=== state_prep.py self-test ===")
    import qutip as qt

    # Verify Bell state preparation
    for deg in [0, 1]:
        p = DEFAULT_PARAMS.copy()
        p["aprox_deg"] = deg
        rho0 = prepare_initial_state(p)
        rho_QA = rho0.ptrace([0, 1])
        C = qt.concurrence(rho_QA)
        print(f"  aprox_deg={deg}: C(rho_QA_init) = {C:.4f}  (expect ~1.0)")

    # Verify PennyLane Bell state
    rho_pl = prepare_bell_state_pennylane()
    print(f"  PennyLane Bell: C = {qt.concurrence(rho_pl):.4f}")

    # Verify frame transforms round-trip
    p = DEFAULT_PARAMS.copy(); p["aprox_deg"] = 1
    rho0 = prepare_initial_state(p)
    t = 1.0
    rho_stat  = to_stationary_frame(rho0, p, t)
    rho_back  = to_rotating_frame(rho_stat, p, t)
    err = (rho0 - rho_back).norm()
    print(f"  Frame round-trip error: {err:.2e}  (expect ~0)")

    print("=== OK ===")


if __name__ == "__main__":
    main()
