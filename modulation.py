import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qutip as qt
import pennylane as qml
import pandas as pd
import warnings
from scipy.signal import find_peaks


from state_prep import DEFAULT_PARAMS, Q, A, E,H_free, H_QA_int, H_QE_int, H_interaction
from dynamical import (
    evolve_lindblad, evolve_mcsolve, evolve_born_markov,
    build_c_ops, gamma_c_zeno, fit_decay_rate,
    full_tomography_shots, qst_log_likelihood,)

# ──────────────────────────────────────────────────────────────────────────────
#  Shared time axis (μs)
# ──────────────────────────────────────────────────────────────────────────────
TLIST = np.linspace(0, 10, 400)
I       = qt.qeye(2)
sx      = qt.sigmax()
sy      = qt.sigmay()
sz      = qt.sigmaz()
sp  = qt.sigmap()    # |0⟩⟨1|  raising
sm = qt.sigmam()    # |1⟩⟨0|  lowering

def get_detuning(params=DEFAULT_PARAMS, mod_type="QA"):
    om_Q = params["om_Q"]
    om_A = params["om_A"]
    om_E = params["om_E"]

    if mod_type == "QA":
        return om_Q - om_A
    elif mod_type == "QE":
        return om_Q - om_E
    else:
        raise ValueError(f"Invalid mod_type: {mod_type}")

# ============================================================
# Time-dependent modulation function
# epsilon enters here as the modulation amplitude
# wm is the modulation frequency
# ============================================================

def cosine_modulation(t, args):
    """
    Direct cosine modulation at ω_m.

    The drive is applied at ω_m = Δ/2 (QA) or Δ/4 (QE) directly.
    The Jacobi-Anger expansion exp(iβ sin(ω_m t)) = Σ J_n(β) exp(inω_m t)
    then selects the n=−2 (QA) or n=−4 (QE) resonant term via RWA —
    no frequency doubling of the drive is needed or correct.
    """
    wm = args["wm"]
    return np.cos(wm * t)

def H_modulation_lab(params=DEFAULT_PARAMS, mod_type="QA", epsilon=0.0, wm=None):
    """
    Lab-frame time-dependent Hamiltonian for parametric modulation demo.

    Q-A:  Q flux modulated at ω_m = Δ_QA/2.  Effective coupling = g·J_2(ε/ω_m).
    Q-E:  Q and E flux modulated anti-phase at ω_m = Δ_QE/4.
          Effective coupling = g·J_4(2ε/ω_m).

    Note: pass epsilon calibrated so that g·J_n(β_opt) = Ω_paper.
    Use bessel_epsilon() to compute the correct epsilon.
    """
    detuning = get_detuning(params, mod_type=mod_type)

    if wm is None and mod_type == "QA":
        wm = abs(detuning) / 2       # drive at Δ_QA/2
    elif wm is None and mod_type == "QE":
        wm = abs(detuning) / 4       # drive at Δ_QE/4

    H0 = H_free(params)

    if mod_type == "QA":
        H_static  = H0 + H_QA_int(params)
        H_mod_op  = Q(sz) / 2.0      # only Q has FFL for QA; coeff = ε(t)

    elif mod_type == "QE":
        H_static  = H0 + H_QE_int(params)
        # Anti-phase: Q up, E down — doubles the differential frequency modulation
        # Effective β' = 2ε/ω_m → J_4 resonance
        H_mod_op  = Q(sz) / 2.0 - E(sz) / 2.0

    def _coeff(t, args):
        return epsilon * cosine_modulation(t, args)

    H = [H_static, [H_mod_op, _coeff]]
    return H, wm

# ============================================================
# Initial states
# ============================================================

g = qt.basis(2, 0)
e = qt.basis(2, 1)

# For QA: start in |100> = Q excited, A ground, E ground
psi0_QA = qt.tensor(e, g, g)

# For QE: also start in |100>
psi0_QE = qt.tensor(e, g, g)

# ============================================================
# Observables
# ============================================================

P_Q_exc = Q(e * e.dag())
P_A_exc = A(e * e.dag())
P_E_exc = E(e * e.dag())

# ============================================================
# Single simulation
# ============================================================

def run_modulation(params=DEFAULT_PARAMS, mod_type="QA", epsilon=2*np.pi*0.1, n=2.0, wm=None,
                   psi0=None, tlist=TLIST, c_ops=None):

    if psi0 is None:
        psi0 = psi0_QA if mod_type == "QA" else psi0_QE

    H, wm_used = H_modulation_lab(
        params=params,
        mod_type=mod_type,
        epsilon=epsilon
    )

    if mod_type == "QA":
        e_ops = [P_Q_exc, P_A_exc]
    else:
        e_ops = [P_Q_exc, P_E_exc]

    result = qt.mesolve(
        H,
        psi0,
        tlist,
        c_ops=[] if c_ops is None else c_ops,
        e_ops=e_ops,
        args={"wm": wm_used}
    )

    return result, wm_used

# ============================================================
# Plot a single run
# ============================================================

def plot_single_run(params=DEFAULT_PARAMS, mod_type="QA", epsilon=2*np.pi*0.1, n=2.0, wm=None,
                    psi0=None, tlist=TLIST):

    result, wm_used = run_modulation(
        params=params,
        mod_type=mod_type,
        epsilon=epsilon,
        n=n,
        wm=wm,
        psi0=psi0,
        tlist=tlist,
        c_ops = build_c_ops(params)
    )

    plt.figure(figsize=(8, 4))

    plt.plot(tlist, result.expect[0], label="Qubit excited population")

    if mod_type == "QA":
        plt.plot(tlist, result.expect[1], label="Ancilla excited population")
        title_target = "Qubit-Ancilla"
    else:
        plt.plot(tlist, result.expect[1], label="Environment excited population")
        title_target = "Qubit-Environment"

    plt.xlabel("Time (μs)")
    plt.ylabel("Population")
    plt.title(f"{title_target} modulation, ε = {epsilon:.4f}, ωm = {wm_used:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Scan modulation amplitude
#
# This is the practical way to choose epsilon:
# sweep epsilon and measure how much population transfer happens
# ============================================================

def scan_amplitude(params=DEFAULT_PARAMS, mod_type="QA", n=2.0, wm=None,
                   eps_list=None, psi0=None, tlist=TLIST):
    
    if mod_type == "QA":
        Delta_QA = abs(params["om_Q"] - params["om_A"])
        eps_list = np.linspace(0, 0.5 * Delta_QA, 30)   # sweep up to 0.5 * Δ_QA
    else:
        # QE J_4 peak at β'=2ε/ω_m ≈ 5.32, i.e. ε_opt ≈ 0.66 Δ_QE.
        # Scan to 1.0 × Δ_QE so the peak and first zero are both visible.
        Delta_QE = abs(params["om_Q"] - params["om_E"])
        eps_list = np.linspace(0, 1.0 * Delta_QE, 30)

    max_transfer = []
    estimated_swap_rate = []

    for eps in eps_list:
        result, wm_used = run_modulation(
            params=params,
            mod_type=mod_type,
            epsilon=eps,
            n=n,
            wm=wm,
            psi0=psi0,
            tlist=tlist,
            c_ops = build_c_ops(params)
        )

        target_pop = result.expect[1]
        max_transfer.append(np.max(target_pop))

        peaks, _ = find_peaks(target_pop, height=0.05)
        if len(peaks) >= 2:
            period = tlist[peaks[1]] - tlist[peaks[0]]
            estimated_swap_rate.append(2 * np.pi / period)
        elif len(peaks) == 1:
            estimated_swap_rate.append(np.nan)
        else:
            estimated_swap_rate.append(np.nan)

    return eps_list, np.array(max_transfer), np.array(estimated_swap_rate), wm_used

# ============================================================
# Plot amplitude scan
# ============================================================

def plot_amplitude_scan(params=DEFAULT_PARAMS, mod_type="QA", n=2.0, wm=None,
                        eps_list=None, psi0=None, tlist=TLIST):

    eps_list, max_transfer, swap_rate, wm_used = scan_amplitude(
        params=params,
        mod_type=mod_type,
        n=n,
        wm=wm,
        eps_list=eps_list,
        psi0=psi0,
        tlist=tlist
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(eps_list, max_transfer, marker="o")
    ax[0].set_xlabel("Modulation amplitude ε")
    ax[0].set_ylabel("Max target excitation")
    ax[0].set_title(f"Population transfer vs amplitude ({mod_type})")

    ax[1].plot(eps_list, swap_rate, marker="o")
    ax[1].set_xlabel("Modulation amplitude ε")
    ax[1].set_ylabel("Estimated swap rate")
    ax[1].set_title(f"Estimated oscillation rate vs amplitude ({mod_type})")

    plt.tight_layout()
    plt.show()

    best_idx = np.argmax(max_transfer)
    print(f"Chosen modulation type: {mod_type}")
    print(f"Used modulation frequency ωm = {wm_used:.6f}")
    print(f"Best ε by max transfer = {eps_list[best_idx]:.6f}")
    print(f"Max transfer = {max_transfer[best_idx]:.6f}")

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example 1: Qubit-Ancilla modulation at about half the detuning
    plot_single_run(
        params=DEFAULT_PARAMS,
        mod_type="QA",
        epsilon=2*np.pi*0.15,
        n=2.0
    )

    plot_amplitude_scan(
        params=DEFAULT_PARAMS,
        mod_type="QA",
        n=2.0,
        eps_list=np.linspace(0, 2*np.pi*0.4, 25)
    )

    # Example 2: Qubit-Environment modulation at about quarter detuning
    plot_single_run(
        params=DEFAULT_PARAMS,
        mod_type="QE",
        epsilon=2*np.pi*0.15,
        n=4.0
    )

    plot_amplitude_scan(
        params=DEFAULT_PARAMS,
        mod_type="QE",
        n=4.0,
        eps_list=np.linspace(0, 2*np.pi*0.4, 25)
    )