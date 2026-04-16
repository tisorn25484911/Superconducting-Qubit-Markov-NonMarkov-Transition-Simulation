"""
experiment.py
=============
Experiment definitions reproducing Gaikwad et al. (PRL 132, 200401, 2024),
plus a custom PennyLane circuit runner for user-designed experiments.

Paper experiments
─────────────────
  exp1()  — Markovian baseline: Bell state free decay (no Q-E coupling)
  exp2()  — Non-Markovian dynamics: concurrence revival with Q-E coupling
  exp3()  — NM→Markovian transition: gamma_E scan
  exp4()  — Quantum Zeno regime: Γ_c vs gamma_E scaling

Dashboard
─────────
  plot_full_dashboard()  — four-panel summary figure (saves PDF)

Custom experiments
──────────────────
  run_custom_circuit()   — user-defined PennyLane circuit + QuTiP open dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qutip as qt
import pennylane as qml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from state_prep import DEFAULT_PARAMS, nm_measure, prepare_initial_state
from dynamical import (
    evolve_lindblad, evolve_mcsolve, evolve_born_markov,
    build_c_ops, gamma_c_zeno, fit_decay_rate,
    full_tomography_shots, qst_log_likelihood,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Shared time axis (μs)
# ──────────────────────────────────────────────────────────────────────────────
TLIST = np.linspace(0, 12, 400)

C_RECORD = 0.8   # concurrence threshold: recording / N-counting starts here


def _trim_to_c_start(conc, tlist, c_start=C_RECORD, nth=3):
    """
    Trim a concurrence time series so that the analysis window begins at the
    nth downward crossing of c_start (C transitions from above to at/below).

    The returned time axis is shifted so that trimmed point = t 0.
    If there are fewer than nth crossings the last crossing is used.
    If C never reaches c_start the original arrays are returned unchanged.

    Parameters
    ----------
    conc    : array-like   — concurrence values
    tlist   : array-like   — corresponding time points (μs)
    c_start : float        — threshold (default C_RECORD = 0.8)
    nth     : int          — which downward crossing to trim from (default 3)

    Returns
    -------
    t_trim   : np.ndarray
    conc_trim: np.ndarray
    """
    conc_arr = np.asarray(conc, dtype=float)
    t_arr    = np.asarray(tlist, dtype=float)

    # Collect indices of downward crossings: C[i-1] > c_start and C[i] <= c_start
    crossings = [i for i in range(1, len(conc_arr))
                 if conc_arr[i - 1] > c_start and conc_arr[i] <= c_start]

    if len(crossings) == 0:
        return t_arr, conc_arr

    # Use nth crossing (1-indexed); fall back to the last one if not enough
    idx = crossings[nth - 1] if len(crossings) >= nth else crossings[-1]
    return t_arr[idx:] - t_arr[idx], conc_arr[idx:]

# ──────────────────────────────────────────────────────────────────────────────
#  Experiment 1 — Markovian baseline
# ──────────────────────────────────────────────────────────────────────────────
def exp1(params=None, tlist=None, plot=True):
    """
    Experiment 1: Bell state free decay with no Q-E coupling.

    Uses a 2-qubit (Q-A only) Lindblad simulation so there is no E memory.
    Expected result: monotonic exponential decay C(t) = C₀ exp(−t/T2Q − t/T2A).

    Returns
    -------
    dict with keys:
      'tlist', 'conc', 'conc_analytic', 'NM',
      'T2Q', 'T2A', 'Gamma0'
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if tlist is None:
        tlist = TLIST

    T2Q    = params.get("T2Q", DEFAULT_PARAMS["T2Q"])
    T2A    = params.get("T2A", DEFAULT_PARAMS["T2A"])
    #T1Q = params.get("T1Q", DEFAULT_PARAMS["T1Q"])
    #T1A = params.get("T1A", DEFAULT_PARAMS["T1A"])
    Om_QA  = params.get("Om_QA", DEFAULT_PARAMS["Om_QA"])
    Gamma0 = 1.0 / T2Q + 1.0 / T2A

    # 2-qubit system: Q and A only, no E
    I2  = qt.qeye(2)
    sp2 = qt.sigmap(); sm2 = qt.sigmam(); sz2 = qt.sigmaz()
    H_2q  = Om_QA/2 * (qt.tensor(sp2, sm2) + qt.tensor(sm2, sp2))
    c_Q2  = np.sqrt(1.0 / (2*T2Q)) * qt.tensor(sz2, I2)
    c_A2  = np.sqrt(1.0 / (2*T2A)) * qt.tensor(I2, sz2)
    #c_Q1 = np.sqrt(1.0 / T1Q) * qt.tensor(sm2, I2)
    #c_A1 = np.sqrt(1.0 / T1A) * qt.tensor(I2, sm2)
    rho0  = qt.ket2dm(qt.bell_state('10'))

    result = qt.mesolve(
        H=H_2q, rho0=rho0, tlist=tlist,
        c_ops=[c_Q2, c_A2], e_ops=[],
        options={'nsteps': 5000},
    )
    conc          = [qt.concurrence(s) for s in result.states]
    C0            = conc[0]
    conc_analytic = C0 * np.exp(-tlist / T2Q - tlist / T2A)

    # Trim both series using the simulation's crossing index so lengths match
    conc_arr  = np.asarray(conc, dtype=float)
    below     = np.where(conc_arr <= C_RECORD)[0]
    trim_idx  = below[0] if len(below) > 0 else 0
    t_trim        = np.asarray(tlist)[trim_idx:] - tlist[trim_idx]
    conc_trim     = conc_arr[trim_idx:]
    analytic_trim = np.asarray(conc_analytic)[trim_idx:]
    NM = nm_measure(conc_trim, t_trim)

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_trim, conc_trim,     lw=2,   color='navy',
                label='mesolve (numerical)')
        ax.plot(t_trim, analytic_trim, lw=1.5, color='red', ls='--',
                label=r'$C_0\,e^{-t/T_{2Q} - t/T_{2A}}$ (analytic)')
        ax.set(xlabel=f'Time since C={C_RECORD} (μs)', ylabel='Concurrence  C',
               title='Exp.1 — Markovian baseline: monotonic exponential decay',
               ylim=[-0.02, 1.05])
        ax.legend()
        ax.grid(True)
        ax.text(0.62, 0.72, f'N = {NM:.4f}',
                transform=ax.transAxes, color='darkgreen', fontsize=12)
        plt.tight_layout(); plt.show()

    print(f"[Exp.1]  C₀={C0:.4f}  C(10μs)={conc[-1]:.4f}  "
          f"N={NM:.4f}  Γ₀={Gamma0:.5f} μs⁻¹")
    return {
        'tlist': t_trim, 'conc': conc_trim,
        'conc_analytic': analytic_trim,
        'NM': NM, 'T2Q': T2Q, 'T2A': T2A, 'Gamma0': Gamma0,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Experiment 2 — Non-Markovian dynamics
# ──────────────────────────────────────────────────────────────────────────────
def exp2(params=None, tlist=None, plot=True, baseline=None):
    """
    Experiment 2: Q-E coupling active, gamma_E = 0 (pure non-Markovian).

    Expected: concurrence revival at t ≈ π/Ω_QE; N > 0.

    Parameters
    ----------
    baseline : list or None  — Exp.1 concurrence to overlay on plot

    Returns
    -------
    dict with keys: 'tlist', 'conc', 'NM', 'states', 'dC_dt'
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if tlist is None:
        tlist = TLIST

    t, states, conc = evolve_lindblad(params, gamma_E=0.0, tlist=tlist)

    # Trim to start when C first reaches C_RECORD (t relabeled 0)
    t_trim, conc_trim = _trim_to_c_start(conc, t)
    NM  = nm_measure(conc_trim, t_trim)
    dC  = np.gradient(np.array(conc_trim), t_trim[1] - t_trim[0])

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        if baseline is not None:
            _n  = min(len(t_trim), len(baseline))
            _tb = t_trim[:_n]
            _b  = np.array(baseline)[:_n]
            _c  = np.array(conc_trim)[:_n]
            ax.plot(_tb, _b, color='gray', lw=1.5,
                    ls='--', alpha=0.7, label='Exp.1 (Markovian baseline)')
            ax.fill_between(_tb, _c, _b,
                            where=_c > _b,
                            alpha=0.25, color='limegreen', label='Revival region')
        ax.plot(t_trim, conc_trim, color='steelblue', lw=2,
                label=f'Exp.2 (γ_E = 0,  N = {NM:.3f})')
        ax.set(xlabel=f'Time since C={C_RECORD} (μs)', ylabel='Concurrence  C',
               title='Exp.2 — Non-Markovian: concurrence revival',
               ylim=[-0.02, 1.05])
        ax.legend(fontsize=9)
        ax.text(0.55, 0.35, f'N = {NM:.3f}',
                transform=ax.transAxes, fontsize=14,
                color='darkgreen', fontweight='bold')

        ax = axes[1]
        ax.plot(t_trim, dC, color='darkred', lw=1.5)
        ax.fill_between(t_trim, dC, 0, where=dC > 0,
                        alpha=0.4, color='limegreen')
        ax.fill_between(t_trim, dC, 0, where=dC < 0,
                        alpha=0.15, color='red')
        ax.axhline(0, color='black', lw=0.8)
        ax.set(xlabel=f'Time since C={C_RECORD} (μs)', ylabel='dC/dt',
               title='Rate of change = info backflow from E')
        ax.legend(fontsize=9)
        axes[0].grid(True)
        ax.grid(True)
        plt.tight_layout(); plt.show()

    print(f"[Exp.2]  N = {NM:.4f}  (paper reports ~1.4)")
    return {'tlist': t_trim, 'conc': conc_trim,
            'NM': NM, 'states': states, 'dC_dt': dC}


# ──────────────────────────────────────────────────────────────────────────────
#  Experiment 3 — NM→Markovian transition
# ──────────────────────────────────────────────────────────────────────────────
def exp3(params=None, tlist=None, gamma_scan=None, plot=True):
    """
    Experiment 3: scan gamma_E to drive the transition from non-Markovian
    to Markovian behaviour.

    Parameters
    ----------
    gamma_scan : list[float]  — E dephasing rates (rad/μs)

    Returns
    -------
    dict with keys: 'gamma_scan', 'all_conc', 'all_NM', 'tlist'
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if tlist is None:
        tlist = TLIST
    if gamma_scan is None:
        gamma_scan = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]

    all_conc = {}
    all_NM   = {}

    for g in gamma_scan:
        print(f"  Scanning γ_E = {g:.2f} rad/μs ...", end=' ', flush=True)
        t, _, conc        = evolve_lindblad(params, gamma_E=g, tlist=tlist)
        t_tr, conc_tr     = _trim_to_c_start(conc, t)
        all_conc[g]       = (t_tr, conc_tr)   # store trimmed pair
        all_NM[g]         = nm_measure(conc_tr, t_tr)
        print(f"N = {all_NM[g]:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        cmap   = plt.cm.plasma(np.linspace(0.1, 0.85, len(gamma_scan)))

        ax = axes[0]
        for g, col in zip(gamma_scan, cmap):
            t_g, c_g = all_conc[g]
            ax.plot(t_g, c_g, color=col, lw=1.8, label=f'γ={g:.1f}')
        ax.set(xlabel=f'Time since C={C_RECORD} (μs)', ylabel='Concurrence  C',
               title='Exp.3 — NM → Markovian transition (γ_E scan)',
               ylim=[-0.02, 1.02])
        ax.legend(fontsize=7, ncol=2)

        ax = axes[1]
        N_list = [all_NM[g] for g in gamma_scan]
        ax.plot(gamma_scan, N_list, 'o-', color='steelblue', lw=2, ms=7)
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.axvline(1.0, color='green',  ls=':', lw=1.5, label='γ≈1 (N→0 onset)')
        ax.axvline(3.0, color='orange', ls=':', lw=1.5, label='γ≈3 (Markovian)')
        ax.set(xlabel='γ_E (rad/μs)', ylabel='N (non-Markovianity)',
               title='Non-Markovianity vs dephasing rate')
        ax.legend(fontsize=9)
        axes[0].grid(True)
        ax.grid(True)
        plt.tight_layout(); plt.show()

    g_max = max(all_NM.keys())
    print(f"[Exp.3]  N(γ=0)={all_NM[0.0]:.3f}  "
          f"N(γ={g_max:.1f})={all_NM[g_max]:.4f}")
    return {'gamma_scan': gamma_scan, 'all_conc': all_conc,
            'all_NM': all_NM, 'tlist': tlist}


# ──────────────────────────────────────────────────────────────────────────────
#  Experiment 4 — Quantum Zeno regime
# ──────────────────────────────────────────────────────────────────────────────
def exp4(params=None, tlist=None, gamma_zeno_scan=None, plot=True):
    """
    Experiment 4: Quantum Zeno effect — strong dephasing on E freezes E in an
    eigenstate, effectively decoupling Q from E and slowing Q-A decoherence.

    Scaling prediction: Γ_c = Ω_QE² / (4γ) + Γ_0

    Parameters
    ----------
    gamma_zeno_scan : list[float]  — high gamma_E values (rad/μs)

    Returns
    -------
    dict with keys: 'gamma_scan', 'Gamma_c_sim', 'Gamma_c_zeno', 'Gamma0'
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if tlist is None:
        tlist = TLIST
    if gamma_zeno_scan is None:
        gamma_zeno_scan = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 45.5]

    T2Q    = params.get("T2Q", DEFAULT_PARAMS["T2Q"])
    T2A    = params.get("T2A", DEFAULT_PARAMS["T2A"])
    Gamma0 = 1.0 / T2Q + 1.0 / T2A

    Gamma_c_sim  = []
    Gamma_c_pred = []
    all_conc     = {}   # store time series for plotting (avoids re-running)

    for g in gamma_zeno_scan:
        print(f"  γ_E = {g:.2f} ...", end=' ', flush=True)
        t, _, conc        = evolve_lindblad(params, gamma_E=g, tlist=tlist)
        t_tr, conc_tr     = _trim_to_c_start(conc, t)
        all_conc[g]       = (t_tr, conc_tr)   # store trimmed pair
        Gc_sim            = fit_decay_rate(np.array(conc_tr), t_tr)
        Gc_pred     = gamma_c_zeno(g, params)
        Gamma_c_sim.append(Gc_sim)
        Gamma_c_pred.append(Gc_pred)
        print(f"Γ_c(sim)={Gc_sim:.4f}  Γ_c(Zeno)={Gc_pred:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # Left panel: show ~8 representative curves from the scan
        ax = axes[0]
        n_show = min(8, len(gamma_zeno_scan))
        idx_show = np.round(np.linspace(0, len(gamma_zeno_scan) - 1, n_show)).astype(int)
        subset_g = [gamma_zeno_scan[i] for i in idx_show]
        cols_z   = plt.cm.Blues(np.linspace(0.35, 0.95, n_show))
        for g, col in zip(subset_g, cols_z):
            t_g, c_g = all_conc[g]
            ax.plot(t_g, c_g, color=col, lw=1.8, label=f'γ={g:.1f}')
        ax.set(xlabel=f'Time since C={C_RECORD} (μs)', ylabel='Concurrence  C',
               title='Exp.4 — Zeno regime: more γ → slower decay',
               ylim=[-0.02, 1.02])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True)

        # Right panel: all scan points as small dots + post-peak data fit
        ax = axes[1]
        valid_pairs = [(g, Gc) for g, Gc in zip(gamma_zeno_scan, Gamma_c_sim)
                       if not np.isnan(Gc)]
        fit_params = None
        if valid_pairs:
            gv, Gcv = np.array(list(zip(*valid_pairs)))
            ax.plot(gv, Gcv, 'o', color='steelblue', ms=3, zorder=5,
                    label='Simulation (fitted)')

            # Fit A/γ + B to data points strictly after the peak
            peak_idx = int(np.argmax(Gcv))
            post_g  = gv[peak_idx + 4:]
            post_Gc = Gcv[peak_idx + 4:]
            if len(post_g) >= 2:
                coeffs = np.polyfit(1.0 / post_g, post_Gc, 1)
                A_fit, B_fit = float(coeffs[0]), float(coeffs[1])
                g_fit = np.linspace(0.4, max(gamma_zeno_scan) + 2, 400)
                ax.plot(g_fit, A_fit / g_fit + B_fit, 'r-', lw=2,
                        label=r'Post-peak fit: $A/\gamma + B$')
                fit_params = {'A': A_fit, 'B': B_fit,
                              'g_min': float(post_g[0]),
                              'g_max': float(max(gamma_zeno_scan))}
        ax.axhline(Gamma0, color='green', ls='--', lw=1.5,
                   label=f'Γ₀ = {Gamma0:.4f} μs⁻¹')
        ax.set(xlabel='γ_E (rad/μs)', ylabel='Γ_c (μs⁻¹)',
               title='Zeno scaling of concurrence decay rate')
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.grid(True)
        plt.tight_layout(); plt.show()

    # Compute fit_params outside the plot block so they are always returned
    if not plot:
        valid_pairs = [(g, Gc) for g, Gc in zip(gamma_zeno_scan, Gamma_c_sim)
                       if not np.isnan(Gc)]
        fit_params = None
        if valid_pairs:
            gv, Gcv = np.array(list(zip(*valid_pairs)))
            peak_idx = int(np.argmax(Gcv))
            post_g  = gv[peak_idx + 1:]
            post_Gc = Gcv[peak_idx + 1:]
            if len(post_g) >= 2:
                coeffs  = np.polyfit(1.0 / post_g, post_Gc, 1)
                fit_params = {'A': float(coeffs[0]), 'B': float(coeffs[1]),
                              'g_min': float(post_g[0]),
                              'g_max': float(max(gamma_zeno_scan))}

    print(f"[Exp.4]  Γ₀ = {Gamma0:.5f} μs⁻¹  (asymptote at large γ)")
    return {
        'gamma_scan'  : gamma_zeno_scan,
        'Gamma_c_sim' : Gamma_c_sim,
        'Gamma_c_zeno': Gamma_c_pred,
        'Gamma0'      : Gamma0,
        'fit_params'  : fit_params,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Four-panel summary dashboard
# ──────────────────────────────────────────────────────────────────────────────
def plot_full_dashboard(res1, res2, res3, res4, params=None, save_pdf=True):
    """
    Reproduce the four-panel summary figure from the paper.

    Parameters
    ----------
    res1, res2, res3, res4 : dicts returned by exp1()–exp4()
    save_pdf : bool  — save to 'simulation_dashboard.pdf'
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38)

    # ── Panel A: Exp.1 Markovian baseline ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(res1['tlist'], res1['conc'],
             lw=2, color='navy', label='mesolve')
    ax1.plot(res1['tlist'], res1['conc_analytic'],
             lw=1.5, color='red', ls='--', label='Analytic')
    ax1.set(title='(A) Exp.1 — Markovian baseline',
            xlabel='Time (μs)', ylabel='Concurrence  C', ylim=[0, 1.05])
    ax1.legend(fontsize=8)
    ax1.grid(True)
    ax1.text(0.5, 0.12,
             rf"$\Gamma_0$={res1['Gamma0']:.4f} μs⁻¹",
             transform=ax1.transAxes, fontsize=8, color='darkblue')

    # ── Panel B: Exp.2 non-Markovian revival ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # Align trimmed arrays for overlay (take the shorter length)
    _n = min(len(res1['conc']), len(res2['conc']))
    _t_b  = res2['tlist'][:_n]
    _c1   = np.array(res1['conc'][:_n])
    _c2   = np.array(res2['conc'][:_n])
    ax2.plot(_t_b, _c1,
             color='gray', lw=1.2, ls='--', alpha=0.7, label='Baseline')
    ax2.plot(_t_b, _c2,
             color='steelblue', lw=2,
             label=f"γ=0  (N = {res2['NM']:.2f})")
    ax2.fill_between(_t_b, _c2, _c1,
                     where=_c2 > _c1,
                     alpha=0.3, color='limegreen', label='Revival region')
    ax2.set(title='(B) Exp.2 — Non-Markovian revival',
            xlabel=f'Time since C={C_RECORD} (μs)', ylabel='Concurrence  C',
            ylim=[0, 1.05])
    ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.text(0.5, 0.25, f"N = {res2['NM']:.2f}",
             transform=ax2.transAxes, fontsize=14,
             color='darkgreen', fontweight='bold')

    # ── Panel C: Exp.3 gamma scan ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    gamma_plot = [g for g in res3['gamma_scan'] if g <= 5.0]
    cols3      = plt.cm.viridis(np.linspace(0.1, 0.9, len(gamma_plot)))
    for g, col in zip(gamma_plot, cols3):
        t_g, c_g = res3['all_conc'][g]
        ax3.plot(t_g, c_g, color=col, lw=1.8, label=f'γ={g:.1f}')
    ax3.set(title='(C) Exp.3 — NM → Markovian transition',
            xlabel=f'Time since C={C_RECORD} (μs)', ylabel='Concurrence  C',
            ylim=[0, 1.05])
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True)

    # ── Panel D: Exp.4 Zeno scaling ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    valid_pairs = [(g, Gc) for g, Gc in
                   zip(res4['gamma_scan'], res4['Gamma_c_sim'])
                   if not np.isnan(Gc)]
    if valid_pairs:
        gv, Gcv = zip(*valid_pairs)
        ax4.plot(gv, Gcv, 'o', color='steelblue', ms=3, label='Simulation')
    fp = res4.get('fit_params')
    if fp is not None:
        g_fit = np.linspace(fp['g_min'], fp['g_max'] + 2, 300)
        ax4.plot(g_fit, fp['A'] / g_fit + fp['B'], 'r-', lw=2,
                 label=r'Post-peak fit: $A/\gamma + B$')
    ax4.axhline(res4['Gamma0'], color='green', ls='--', lw=1.5,
                label=f"Γ₀={res4['Gamma0']:.4f}")
    ax4.set(title='(D) Exp.4 — Zeno scaling',
            xlabel='γ_E (rad/μs)', ylabel='Γ_c (μs⁻¹)')
    ax4.legend(fontsize=8)
    ax4.set_xscale('log')
    ax4.grid(True)

    plt.suptitle(
        'Gaikwad et al. PRL 132, 200401 (2024)\n'
        'Non-Markovian to Markovian Transition in Open Quantum System Dynamics',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    if save_pdf:
        fig.savefig('simulation_dashboard.pdf', bbox_inches='tight', dpi=150)
        print("Dashboard saved → simulation_dashboard.pdf")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  Custom PennyLane circuit runner
# ──────────────────────────────────────────────────────────────────────────────
def run_custom_circuit(circuit_fn, gamma_E=0.0, params=None, tlist=None,
                       n_shots=1024, plot=True, label="Custom circuit"):
    """
    Execute a user-defined PennyLane circuit as the initial state, then simulate
    its open-system time evolution under the Lindblad master equation.

    Parameters
    ----------
    circuit_fn : callable  — a bare Python function (no decorator).
                 Must contain PennyLane gate calls for 2 wires and return
                 `qml.state()`.  Example:
                     def my_circuit():
                         qml.Hadamard(wires=0)
                         qml.CNOT(wires=[0, 1])
                         return qml.state()
    gamma_E : float  — E dephasing rate (rad/μs)
    params  : dict
    tlist   : array
    n_shots : int    — shots for optional QST at t=0
    plot    : bool
    label   : str    — legend label in the plot

    Returns
    -------
    dict with keys: 'tlist', 'conc', 'NM', 'states', 'initial_rho_QA',
                    'qst_rho' (MLE-reconstructed initial state)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if tlist is None:
        tlist = TLIST

    # ── 1. Execute the PennyLane circuit ─────────────────────────────────────
    dev   = qml.device('default.qubit', wires=2)
    qnode = qml.QNode(circuit_fn, dev)
    sv    = np.array(qnode())
    rho_QA_np = np.outer(sv, sv.conj())
    rho_QA    = qt.Qobj(rho_QA_np, dims=[[2, 2], [2, 2]])

    C_init = qt.concurrence(rho_QA)
    print(f"[{label}]  Initial state C₀ = {C_init:.4f}")
    print(f"  Circuit:\n{qml.draw(qnode)()}")

    # ── 2. Embed into 3-qubit space (E starts in ground state) ───────────────
    rho_E   = qt.ket2dm(qt.basis(2, 0))
    rho0_3q = qt.tensor(rho_QA, rho_E)

    # ── 3. Open-system evolution ──────────────────────────────────────────────
    t, states, conc = evolve_lindblad(params, gamma_E=gamma_E,
                                       tlist=tlist, rho0=rho0_3q)
    NM = nm_measure(conc, t)

    # ── 4. Optional QST at t = 0 ─────────────────────────────────────────────
    df_tomo  = full_tomography_shots(rho_QA, n_shots=n_shots,
                                      exp_label="custom", t_label="t0", seed=0)
    rho_mle, _, _ = qst_log_likelihood(df_tomo)
    C_mle    = qt.concurrence(rho_mle)
    print(f"  QST (MLE, n_shots={n_shots}): C_MLE = {C_mle:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.plot(t, conc, lw=2, color='darkorange',
                label=f'{label}  γ_E={gamma_E:.2f}  N={NM:.3f}')
        ax.set(xlabel='Time (μs)', ylabel='Concurrence  C',
               title='Custom circuit — open system time evolution',
               ylim=[-0.02, max(1.05, max(conc) + 0.05)])
        ax.legend(fontsize=9)
        ax.grid(True)

        ax = axes[1]
        dC = np.gradient(np.array(conc), t[1] - t[0])
        ax.plot(t, dC, color='darkred', lw=1.5)
        ax.fill_between(t, dC, 0, where=dC > 0,
                        alpha=0.4, color='limegreen', label=f'N={NM:.3f}')
        ax.fill_between(t, dC, 0, where=dC < 0, alpha=0.15, color='red')
        ax.axhline(0, color='black', lw=0.8)
        ax.set(xlabel='Time (μs)', ylabel='dC/dt',
               title='Rate of concurrence change')
        ax.legend(fontsize=9)
        ax.grid(True)
        plt.suptitle(label, fontsize=12)
        plt.tight_layout(); plt.show()

    print(f"[{label}]  N = {NM:.4f}")
    return {
        'tlist': t, 'conc': conc, 'NM': NM,
        'states': states, 'initial_rho_QA': rho_QA, 'qst_rho': rho_mle,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=== experiment.py self-test (short tlist) ===")
    params = DEFAULT_PARAMS.copy()
    tlist  = np.linspace(0, 5, 80)

    r1 = exp1(params=params, tlist=tlist, plot=False)
    print(f"  Exp.1 OK  N={r1['NM']:.4f}")

    r2 = exp2(params=params, tlist=tlist, plot=False, baseline=r1['conc'])
    print(f"  Exp.2 OK  N={r2['NM']:.4f}")

    r3 = exp3(params=params, tlist=tlist, plot=False,
              gamma_scan=[0.0, 1.0, 5.0])
    g_max = max(r3['all_NM'].keys())
    print(f"  Exp.3 OK  N(γ=0)={r3['all_NM'][0.0]:.3f}  N(γ={g_max})={r3['all_NM'][g_max]:.4f}")

    r4 = exp4(params=params, tlist=tlist, plot=False,
              gamma_zeno_scan=[2.0, 10.0])
    print(f"  Exp.4 OK  Γ₀={r4['Gamma0']:.4f}")

    # Custom circuit
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    rc = run_custom_circuit(bell_circuit, gamma_E=0.0, params=params,
                             tlist=tlist, n_shots=256, plot=False,
                             label="CNOT Bell state")
    print(f"  Custom circuit OK  N={rc['NM']:.4f}")
    print("=== OK ===")


if __name__ == "__main__":
    main()
