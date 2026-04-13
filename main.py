"""
main.py
=======
Reproduces all four experiments from:

  Gaikwad et al.
  "Entanglement Assisted Probe of the Non-Markovian to Markovian Transition
   in Open Quantum System Dynamics"
  Phys. Rev. Lett. 132, 200401 (2024)

Run:
    python main.py

What this script does
─────────────────────
  Exp.1  Markovian baseline — Bell state free decay, no Q-E coupling.
         Verifies exponential concurrence decay C(t) = C₀ exp(−t/T2Q − t/T2A).

  Exp.2  Non-Markovian dynamics — Q-E coupling active, gamma_E = 0.
         Demonstrates concurrence revival and computes N > 0.

  Exp.3  NM→Markovian transition — scans gamma_E from 0 to 10 rad/μs.
         Shows continuous suppression of non-Markovianity N as gamma_E grows.

  Exp.4  Quantum Zeno regime — gamma_E >> Ω_QE.
         Fitted Γ_c vs gamma_E follows the Zeno scaling Ω_QE²/(4γ) + Γ₀.

After all experiments, a four-panel dashboard is saved to simulation_dashboard.pdf.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import qutip as qt
import pennylane as qml

from state_prep import DEFAULT_PARAMS
from experiment import exp1, exp2, exp3, exp4, plot_full_dashboard


# ──────────────────────────────────────────────────────────────────────────────
#  Parameters
# ──────────────────────────────────────────────────────────────────────────────
PARAMS = DEFAULT_PARAMS.copy()
PARAMS["aprox_deg"] = 0   # interaction picture — exact paper reproduction

# Shared time axis for Experiments 1-3 (μs)
TLIST = np.linspace(0, 10, 400)

# gamma_E scan for Exp.3
GAMMA_SCAN     = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# gamma_E values for Exp.4 (Zeno regime)
GAMMA_ZENO_SCAN = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0,  20.0, 25.0, 30.0, 35.0, 45.5]


# ──────────────────────────────────────────────────────────────────────────────
def _banner(n, title):
    print()
    print("=" * 60)
    print(f"  [{n}/4] {title}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Gaikwad et al.  PRL 132, 200401 (2024)             ║")
    print("║   Non-Markovian → Markovian Transition Simulation    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  T2Q   = {PARAMS['T2Q']:.1f} μs")
    print(f"  T2A   = {PARAMS['T2A']:.1f} μs")
    print(f"  Ω_QA  = 2π × {PARAMS['Om_QA']/(2*np.pi):.3f} MHz")
    print(f"  Ω_QE  = 2π × {PARAMS['Om_QE']/(2*np.pi):.3f} MHz")
    print(f"  Approx degree: {PARAMS['aprox_deg']}  (0 = interaction picture)")

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    _banner(1, "Markovian baseline  (no Q-E coupling, gamma_E = 0)")
    res1 = exp1(params=PARAMS, tlist=TLIST, plot=True)

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    _banner(2, "Non-Markovian dynamics  (gamma_E = 0, Q-E coupled)")
    res2 = exp2(params=PARAMS, tlist=TLIST, plot=True, baseline=res1['conc'])

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    _banner(3, "NM → Markovian transition  (gamma_E scan)")
    res3 = exp3(params=PARAMS, tlist=TLIST, gamma_scan=GAMMA_SCAN, plot=True)

    # ── Experiment 4 ──────────────────────────────────────────────────────────
    _banner(4, "Quantum Zeno regime  (high gamma_E)")
    res4 = exp4(params=PARAMS, tlist=TLIST,
                gamma_zeno_scan=GAMMA_ZENO_SCAN, plot=True)

    # ── Summary dashboard ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Generating four-panel summary dashboard…")
    print("=" * 60)
    plot_full_dashboard(res1, res2, res3, res4, params=PARAMS, save_pdf=True)

    # ── Print summary ─────────────────────────────────────────────────────────
    T2Q    = PARAMS["T2Q"]; T2A = PARAMS["T2A"]
    Gamma0 = 1.0 / T2Q + 1.0 / T2A
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                 RESULTS SUMMARY                                    ║")
    print("╠════════════════════════════════════════════════════════════════════╣")
    print(f"║  Exp.1  N = {res1['NM']:.4f}   (expect: 0,   Markovian)                      ║")
    print(f"║  Exp.2  N = {res2['NM']:.4f}   (paper: ~1.4, non-Markovian)                  ║")
    print(f"║  Exp.3  N(γ=0)  = {res3['all_NM'][0.0]:.3f}                                            ║")
    print(f"║         N(γ=10) = {res3['all_NM'][10.0]:.4f}  (→ 0, Markovian)                         ║")
    print(f"║  Exp.4  Γ₀ = {Gamma0:.5f} μs⁻¹  (Zeno asymptote)                        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print("  simulation_dashboard.pdf saved.")
    print()


if __name__ == "__main__":
    main()
