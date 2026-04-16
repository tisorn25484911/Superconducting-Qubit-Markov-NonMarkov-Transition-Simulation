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
from scipy.special import jn as bessel_jn

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
    # Qubit transition frequencies for aprox_deg = 1.
    #
    # *** IMPORTANT — frequency scale constraint for H_lab_parametric ***
    # The parametric-modulation (Jacobi-Anger) scheme requires:
    #
    #   g_bare  <<  ω_m  <<  Δ
    #
    # where g_bare = Ω_QA / J_2(β) ≈ 0.98 MHz and ω_m = Δ_QA / 2.
    # This means Δ_QA must be >> 0.98 MHz.
    #
    # Current values (below) give Δ_QA = 0.35 MHz < g_bare — the RWA is
    # completely invalid and H_lab_parametric produces unphysical rapid
    # Rabi oscillations (~1 µs period) with N ≈ 15.
    #
    # These values ARE used for H_detuning (AC Stark shift corrections),
    # which only requires Δ to be non-zero, not large.
    #
    # To enable H_lab_parametric, increase both detunings so Δ/g_bare > 50:
    #   "delta_QA": 2*np.pi*50,  "om_Q": 2*np.pi*200, "om_A": 2*np.pi*150
    # (requires nsteps ≥ 100 000 and longer run times).
    "delta_QA" : 2 * np.pi * (4.55 - 4.2),   # 0.35 MHz — too small for H_lab_parametric
    "delta_QE" : 2 * np.pi * (4.55 - 4.08),  # 0.47 MHz — too small for H_lab_parametric
    "om_Q"  : 2 * np.pi * 4.55,   # rad/μs  (NOTE: GHz numeral used as MHz value)
    "om_A"  : 2 * np.pi * 4.200,
    "om_E"  : 2 * np.pi * 4.08,
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

    Both A and E sit below Q:
      ω_A = ω_Q − Δ_QA,   ω_E = ω_Q − Δ_QE

    H_0 = ω_Q σ_z^Q/2 + ω_A σ_z^A/2 + ω_E σ_z^E/2
    """
    om_Q  = params.get("om_Q",  DEFAULT_PARAMS["om_Q"])
    delta_QA = params.get("delta_QA", DEFAULT_PARAMS["delta_QA"])
    delta_QE = params.get("delta_QE", DEFAULT_PARAMS["delta_QE"])
    om_A  = om_Q - delta_QA    # A below Q: Δ_QA = om_Q - om_A
    om_E  = om_Q - delta_QE    # E below Q: Δ_QE = om_Q - om_E
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
    Leading-order correction to the RWA for aprox_deg = 1 (doubly-rotating frame).

    After the Jacobi-Anger expansion and RWA, the dominant residual correction is
    the AC Stark (dispersive) shift from the off-resonant Bessel sidebands:

        χ_QA = Ω_QA² / Δ_QA   (Q-A dispersive shift, rad/μs)
        χ_QE = Ω_QE² / Δ_QE   (Q-E dispersive shift)

    These are of order Ω²/Δ and vanish in the Δ → ∞ limit (recovering aprox_deg=0).
    The sign convention is:
      - Q acquires a negative shift from both pairs (it is the "upper" qubit)
      - A and E each acquire a positive shift from their respective pair

    NOTE: The earlier implementation used ±Δ/2, which is the residual in a plain
    single-rotating frame — NOT the correct correction in the doubly-rotating
    parametric-modulation frame.  That expression is valid only if Δ >> Ω, in
    which case it describes the qubit precession relative to the modulation drive,
    but it is not a small correction and wrongly distorts the iSWAP dynamics when
    Δ ~ Ω (the current parameter regime).
    """
    delta_QA = params.get("delta_QA", DEFAULT_PARAMS["delta_QA"])
    delta_QE = params.get("delta_QE", DEFAULT_PARAMS["delta_QE"])
    Om_QA    = params.get("Om_QA",    DEFAULT_PARAMS["Om_QA"])
    Om_QE    = params.get("Om_QE",    DEFAULT_PARAMS["Om_QE"])

    # AC Stark shifts (dispersive correction, order Ω²/Δ)
    chi_QA = Om_QA**2 / delta_QA if abs(delta_QA) > 1e-12 else 0.0
    chi_QE = Om_QE**2 / delta_QE if abs(delta_QE) > 1e-12 else 0.0

    dQ = -(chi_QA + chi_QE) / 4.0   # Q shifts down from both pairs
    dA = +chi_QA / 4.0               # A shifts up from Q-A pair
    dE = +chi_QE / 4.0               # E shifts up from Q-E pair

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
#  Parametric modulation — lab-frame time-dependent Hamiltonian (aprox_deg = 1)
# ──────────────────────────────────────────────────────────────────────────────
def _bessel_scale(params):
    """
    Compute modulation parameters so the effective parametric coupling matches
    the paper values exactly.

    Physics (Gaikwad et al., Supplemental):
      Q-A: Q flux driven at ω_m = Δ_QA/2.
           Phase accumulated: Δ_QA·t + (ε/ω_m)·sin(ω_m·t).
           Jacobi-Anger: effective coupling = g_bare · J_2(ε/ω_m).
           → Choose β_QA = ε_QA/ω_m at the J_2 first maximum (β≈3.054).
           → Rescale g_bare = Ω_QA / J_2(β_QA) so g_bare·J_2 = Ω_QA exactly.

      Q-E: Q and E both driven anti-phase at ω_m = Δ_QE/4.
           Differential modulation amplitude is 2ε, so β' = 2ε/ω_m.
           Effective coupling = g_bare · J_4(β').
           → Choose β' at J_4 first maximum (β'≈5.318).
           → Rescale g_bare = Ω_QE / J_4(β').

    Returns dict with wm_QA, wm_QE, eps_QA, eps_QE, g_QA_bare, g_QE_bare.
    """
    delta_QA = params.get("delta_QA", DEFAULT_PARAMS["delta_QA"])
    delta_QE = params.get("delta_QE", DEFAULT_PARAMS["delta_QE"])
    Om_QA    = params.get("Om_QA",    DEFAULT_PARAMS["Om_QA"])
    Om_QE    = params.get("Om_QE",    DEFAULT_PARAMS["Om_QE"])

    wm_QA = abs(delta_QA) / 2.0   # modulation freq for Q-A
    wm_QE = abs(delta_QE) / 4.0   # modulation freq for Q-E

    # Optimal modulation indices at first maximum of J_n
    beta_QA = 3.0542    # argmax J_2;  J_2(3.054) ≈ 0.4865
    beta_QE = 5.3175    # argmax J_4 in terms of β'=2ε/ω_m;  J_4(5.318) ≈ 0.3912

    eps_QA  = beta_QA * wm_QA           # ε_QA = β · ω_m
    eps_QE  = beta_QE * wm_QE / 2.0    # β' = 2ε/ω_m → ε = β'·ω_m/2

    j2 = float(bessel_jn(2, beta_QA))  # ≈ 0.4865
    j4 = float(bessel_jn(4, beta_QE))  # ≈ 0.3912

    return dict(
        wm_QA     = wm_QA,
        wm_QE     = wm_QE,
        eps_QA    = eps_QA,
        eps_QE    = eps_QE,
        g_QA_bare = Om_QA / j2,
        g_QE_bare = Om_QE / j4,
        j2=j2, j4=j4,
    )


def H_lab_parametric(params):
    """
    Full lab-frame time-dependent Hamiltonian with parametric modulation.
    Used by evolve_lindblad when aprox_deg = 1.

    Returns (H_list, args) ready for qt.mesolve(H_list, ..., args=args).

    Structure:
      H(t) = H_free
            + g_QA_bare (σ+_Q σ-_A + h.c.)          [bare Q-A exchange]
            + g_QE_bare (σ+_Q σ-_E + h.c.)          [bare Q-E exchange]
            + [ε_QA cos(ω_m_QA t) + ε_QE cos(ω_m_QE t)] · σ_z^Q/2   [Q flux drive]
            + [−ε_QE cos(ω_m_QE t)]                 · σ_z^E/2        [E flux drive, anti-phase]

    The two drives on Q create:
      - J_2 resonance at Δ_QA  (from ε_QA at ω_m_QA = Δ_QA/2)
      - J_4 resonance at Δ_QE  (from ε_QE at ω_m_QE = Δ_QE/4, anti-phase with E)
    """
    bp = _bessel_scale(params)

    H_static = (
        H_free(params)
        + bp['g_QA_bare'] * (qt.tensor(s_plus, s_minus, I) + qt.tensor(s_minus, s_plus, I))
        + bp['g_QE_bare'] * (qt.tensor(s_plus, I, s_minus) + qt.tensor(s_minus, I, s_plus))
    )

    H_Q_mod = Q(sz) / 2.0   # Q flux operator
    H_E_mod = E(sz) / 2.0   # E flux operator (anti-phase, Q-E drive only)

    def _c_Q(t, args):
        """Q modulation: both drives act on Q's σ_z."""
        return (args['eps_QA'] * np.cos(args['wm_QA'] * t)
              + args['eps_QE'] * np.cos(args['wm_QE'] * t))

    def _c_E(t, args):
        """E modulation: anti-phase with Q-E drive."""
        return -args['eps_QE'] * np.cos(args['wm_QE'] * t)

    H_list = [H_static, [H_Q_mod, _c_Q], [H_E_mod, _c_E]]
    args   = {
        'eps_QA': bp['eps_QA'],
        'eps_QE': bp['eps_QE'],
        'wm_QA' : bp['wm_QA'],
        'wm_QE' : bp['wm_QE'],
    }
    return H_list, args


# ──────────────────────────────────────────────────────────────────────────────
#  Frame transformations  (rotating ↔ stationary lab frame)
# ──────────────────────────────────────────────────────────────────────────────
def _U0(params, t):
    """
    Free-evolution unitary  U_0(t) = exp(−i H_detuning · t)  (3-qubit).
    Used to transform between the doubly-rotating frame and a reference frame
    in which the qubit operators are stationary.
    """
    delta_QA = params.get("delta_QA", DEFAULT_PARAMS["delta_QA"])
    delta_QE = params.get("delta_QE", DEFAULT_PARAMS["delta_QE"])
    dQ = -delta_QA / 2.0 - delta_QE / 4.0
    dA = +delta_QA / 2.0
    dE = +delta_QE / 4.0
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
