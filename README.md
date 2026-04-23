# Open Quantum System Simulation Platform

Numerical reproduction of:

> A. Gaikwad, C. Regmi, M. Khurana, N. Dixit, A. Rao, T. Mahesh,
> *"Entanglement Assisted Probe of the Non-Markovian to Markovian
> Transition in Open Quantum System Dynamics"*,
> **Phys. Rev. Lett. 132, 200401 (2024)**

---

## Table of Contents

1. [Overview](#overview)
2. [Physical Background](#physical-background)
3. [System Architecture](#system-architecture)
4. [File Structure](#file-structure)
5. [Module Reference](#module-reference)
   - [state_prep.py](#state_preppy)
   - [dynamical.py](#dynamicalpy)
   - [experiment.py](#experimentpy)
   - [modulation.py](#modulationpy)
   - [main.py](#mainpy)
6. [Physics Parameters](#physics-parameters)
7. [Approximation Degrees](#approximation-degrees)
8. [Solvers](#solvers)
9. [Tomography](#tomography)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Expected Results](#expected-results)
13. [Reference](#reference)

---

## Overview

This platform simulates a **three-qubit open quantum system** — **Q** (system qubit), **A** (ancilla), and **E** (environment) — in which the E qubit acts as a tunable memory channel. By sweeping the dephasing rate γ_E on E, the dynamics of the Q-A entanglement (measured by Wootters concurrence C) can be tuned **continuously from non-Markovian to Markovian**:

- **γ_E = 0:** E retains its quantum state, acts as a memory, and information can flow back from E to Q-A → concurrence revival, N > 0
- **γ_E large:** Strong dephasing on E destroys its memory — Q-A decoherence becomes monotonic and Markovian (N → 0)
- **γ_E ≫ Ω_QE (Zeno regime):** Paradoxically, very strong dephasing *freezes* E via the quantum Zeno effect, decoupling Q from E and *slowing* Q-A decoherence → Γ_c = Ω_QE²/(4γ_E) + Γ_0

The four paper experiments are reproduced numerically:

| # | Experiment | Key observable | Expected result |
|---|------------|---------------|-----------------|
| 1 | Markovian baseline — Bell state free decay, no Q-E coupling | Concurrence C(t), NM measure N | N ≈ 0, C(t) = C₀ exp(−t/T2Q − t/T2A) |
| 2 | Non-Markovian dynamics — Q-E coupling, γ_E = 0 | C(t) revival, dC/dt > 0 | N ≈ 1.4, clear concurrence revival |
| 3 | NM → Markovian transition — γ_E scan 0 → 20 rad/μs | N(γ_E) | N decreases continuously to 0 |
| 4 | Quantum Zeno regime — large γ_E | Γ_c vs γ_E | Γ_c = Ω_QE²/(4γ_E) + Γ_0 scaling |

---

## Physical Background

### Hilbert Space

The full system lives in **H_Q ⊗ H_A ⊗ H_E** (dimension 8). Qubit ordering convention throughout all code:

```
index 0 = Q (system qubit)
index 1 = A (ancilla qubit)
index 2 = E (environment qubit)
```

### Hamiltonian

In the interaction picture (approximation degree 0, default), the time-independent Hamiltonian is:

```
H = (Ω_QA/2)(σ+^Q σ-^A + σ-^Q σ+^A)   [Q-A iSWAP exchange]
  + (Ω_QE/2)(σ+^Q σ-^E + σ-^Q σ+^E)   [Q-E iSWAP exchange]
```

In the doubly-rotating frame (approximation degree 1), AC Stark shift corrections are added:

```
H += (χ_QA/4) σ_z^A  −  ((χ_QA + χ_QE)/4) σ_z^Q  +  (χ_QE/4) σ_z^E
```
where χ_QA = Ω_QA² / Δ_QA, χ_QE = Ω_QE² / Δ_QE.

### Lindblad Dissipators

The open-system master equation (GKSL) is:

```
dρ/dt = −i[H, ρ] + Σ_k (L_k ρ L_k† − ½ {L_k† L_k, ρ})
```

Collapse operators included:

| Collapse operator | Rate | Effect |
|------------------|------|--------|
| Q(σ_z) | √(1/2T2Q) | Pure dephasing on Q |
| A(σ_z) | √(1/2T2A) | Pure dephasing on A |
| E(σ_z) | √(1/2T2E) | Intrinsic dephasing on E |
| Q(σ_−) | √(1/T1Q) | Amplitude decay on Q |
| A(σ_−) | √(1/T1A) | Amplitude decay on A |
| E(σ_−) | √(1/T1E) | Amplitude decay on E |
| E(σ_z) | √(γ_E) | Tunable extra dephasing on E (memory control) |

### Non-Markovianity Measure

Uses the Rivas-Huelga-Plenio (RHP) measure based on entanglement backflow:

```
N = ∫ max(dC/dt, 0) dt ≡ (integral of dC/dt) + ΔC
```

Computationally implemented as:

```python
N = trapezoid(|dC/dt|, t) + (C_final − C_initial)
```

N > 0 if and only if there exist time intervals where dC/dt > 0 (information flows back from E to Q-A).

### Quantum Zeno Scaling

When γ_E ≫ Ω_QE, E is rapidly projected onto a dephasing eigenstate, which Zeno-freezes the Q-E interaction. The effective Q-A concurrence decay rate follows:

```
Γ_c = Ω_QE² / (4 γ_E) + Γ_0
```

where `Γ_0 = 1/T2Q + 1/T2A ≈ 0.0499 μs⁻¹` is the bare Q-A decoherence rate and `Ω_QE²/(4γ_E)` is the residual Zeno-suppressed Q-E contribution.

### Parametric Modulation (Lab Frame)

The iSWAP couplings are realised in the lab frame via **flux modulation** of the transmon qubits:

- **Q-A:** Q flux driven at ω_m = Δ_QA/2 with amplitude ε_QA. After Jacobi-Anger expansion, the resonant n=2 Bessel sideband gives an effective coupling g·J₂(β) where β = ε/ω_m. Optimal: β_QA ≈ 3.054 (first maximum of J₂), so J₂(3.054) ≈ 0.487.
- **Q-E:** Q and E driven anti-phase at ω_m = Δ_QE/4. The differential β' = 2ε/ω_m selects J₄. Optimal: β' ≈ 5.318, J₄(5.318) ≈ 0.391.

---

## System Architecture

```
                ┌─────────────────────────────────────────────┐
                │          interactive_simulation.ipynb        │
                │              (12 sections, widgets)          │
                └──────────────┬──────────────────────────────┘
                               │  imports
              ┌────────────────┼────────────────────────┐
              ▼                ▼                        ▼
        experiment.py      modulation.py            main.py
         exp1–exp4()      H_modulation_lab()       runs all 4
         dashboard         scan_amplitude()         + dashboard
              │
              ▼
        dynamical.py  ◄──── state_prep.py
         3 solvers           operators
         QST / QPT           Hamiltonians
         measurement         frame transforms
                             Bell state prep
```

**Data flow:**
1. `state_prep.py` defines all operators, Hamiltonians, and the initial state.
2. `dynamical.py` uses those to run time evolution, collect shots, and do tomography.
3. `experiment.py` orchestrates the four paper experiments and plotting.
4. `modulation.py` provides the independent lab-frame parametric modulation analysis.
5. `main.py` chains all experiments and generates the PDF dashboard.

---

## File Structure

```
.
├── state_prep.py              # Operators, Hamiltonians, frame transforms, Bell prep, NM measure
├── dynamical.py               # Solvers (Lindblad/Born-Markov/MC), measurement, QST, QPT
├── experiment.py              # exp1()–exp4(), dashboard, custom PennyLane circuit runner
├── modulation.py              # Parametric flux-modulation analysis (lab-frame, Bessel sideband)
├── main.py                    # Runs all four experiments sequentially, saves dashboard PDF
├── interactive_simulation.ipynb  # 12-section interactive notebook (imports .py modules)
├── measurement_results/       # Auto-generated: multi-shot tomography .txt files
│   ├── qst_demo_t-1_xx.txt   # shots in XX basis (Q measured in X, A in X)
│   ├── qst_demo_t-1_xy.txt   # … and 8 other Pauli-basis combinations
│   └── …
└── simulation_dashboard.pdf   # Auto-generated: four-panel summary figure
```

---

## Module Reference

### `state_prep.py`

State preparation, Hamiltonian definitions, and frame-transformation utilities.

#### Operators

Single-qubit Paulis (`I`, `sx`, `sy`, `sz`, `s_plus`, `s_minus`) and embedding functions:

```python
Q(op)  # embeds op into Q slot: tensor(op, I, I)
A(op)  # embeds op into A slot: tensor(I, op, I)
E(op)  # embeds op into E slot: tensor(I, I, op)
```

#### Standard 2-qubit Gates

```python
iSWAP_gate()       # full iSWAP: |10⟩↔i|01⟩
sqrt_iSWAP_gate()  # √iSWAP (half period)
```

#### Hamiltonians

| Function | Description |
|----------|-------------|
| `H_free(params)` | Lab-frame free Hamiltonian H₀ = ω_Q σ_z^Q/2 + ω_A σ_z^A/2 + ω_E σ_z^E/2 |
| `H_QA_int(params)` | Q-A iSWAP: (Ω_QA/2)(σ+^Q σ-^A + h.c.) |
| `H_QE_int(params)` | Q-E iSWAP: (Ω_QE/2)(σ+^Q σ-^E + h.c.) |
| `H_interaction(params)` | Pure interaction Hamiltonian (RWA, aprox_deg=0) |
| `H_detuning(params)` | AC Stark shift corrections (aprox_deg=1 only) |
| `H_for_simulation(params)` | Selects correct H based on `params["aprox_deg"]` |
| `H_lab_parametric(params)` | Full lab-frame time-dependent H with parametric drives |

#### State Preparation

```python
prepare_initial_state(params)
# Returns the paper's initial 8×8 density matrix rho0:
#   aprox_deg=0: exact Bell state |Ψ+>_{QA} ⊗ |0>_E (C=1)
#   aprox_deg=1: Bell state prepared by evolving |10,0>_{QAE}
#                under H_QA for t = π/(4 Ω_QA) (sqrt(iSWAP) gate)

prepare_bell_state_pennylane()
# Returns the Q-A Bell state via PennyLane's IsingXY gate (C=1)
```

#### Frame Transforms (aprox_deg=1 only)

```python
to_stationary_frame(rho_rot, params, t)
# ρ_stat(t) = U₀(t) ρ_rot(t) U₀†(t)

to_rotating_frame(rho_stat, params, t)
# ρ_rot(t) = U₀†(t) ρ_stat(t) U₀(t)
```

#### Single-Qubit Gate Utilities

```python
rotation_gate(axis, angle)
# R_axis(θ) = exp(−i θ/2 σ_axis), axis in {'x','y','z'}

gate_time(gate_angle, params, qubit='Q')
# Time to perform gate_angle rotation using the qubit's iSWAP coupling rate

apply_single_qubit_gate(rho_3q, gate_qobj, qubit_idx)
# Apply a 2×2 gate to one qubit of a 3-qubit state

apply_gate_in_lab_frame(rho_rot, params, t, gate_qobj, qubit_idx)
# Apply gate in lab frame: transform to stationary → apply → transform back
```

#### Non-Markovianity and Concurrence

```python
nm_measure(conc_list, tlist)
# N = trapezoid(|dC/dt|, t) + ΔC  (RHP measure)
# Returns float; N>0 ↔ non-Markovian

concurrence_manual(rho_qobj)
# Wootters concurrence C = max(0, √λ₁ − √λ₂ − √λ₃ − √λ₄)
# where λᵢ are eigenvalues of R = ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
# Returns (C, sqrt_eigenvalues)
```

---

### `dynamical.py`

Dynamical evolution, multi-shot measurement simulation, quantum state tomography (MLE), and quantum process tomography.

#### Collapse Operators

```python
build_c_ops(params, gamma_E=0.0)
# Returns Lindblad collapse operators:
#   Q pure dephasing: sqrt(1/(2T2Q)) * Q(σ_z)
#   A pure dephasing: sqrt(1/(2T2A)) * A(σ_z)
#   E intrinsic deph: sqrt(1/(2T2E)) * E(σ_z)
#   Q amplitude decay: sqrt(1/T1Q) * Q(σ_−)
#   A amplitude decay: sqrt(1/T1A) * A(σ_−)
#   E amplitude decay: sqrt(1/T1E) * E(σ_−)
#   E tunable deph:   sqrt(gamma_E) * E(σ_z)  [only if gamma_E > 0]
```

#### Solver 1: Lindblad Master Equation

```python
t, states, conc = evolve_lindblad(params, gamma_E=0.0, tlist=None, rho0=None)
```

Solves the GKSL master equation using QuTiP `mesolve`. Returns:
- `tlist` — time points (μs)
- `states` — list of 3-qubit density matrices (QuTiP Qobj)
- `conc` — list of Q-A Wootters concurrences (via `ptrace([0,1])`)

Solver options: `nsteps=8000`, `atol=1e-9`, `rtol=1e-7`.

#### Solver 2: Bloch-Redfield (Born-Markov, structured bath)

```python
t, states, conc = evolve_born_markov(params, spectral_density=None, tlist=None, rho0=None)
```

Uses QuTiP `brmesolve`. Goes beyond flat-spectrum Lindblad by retaining the frequency-dependent bath coupling — appropriate for transmon 1/f noise. Default spectral function: Ohmic bath with cutoff ω_c = 2π × 5 GHz calibrated to match T2Q, at temperature kBT = 0.04 (≈ 20 mK). Falls back to `evolve_lindblad` on failure.

#### Solver 3: Monte Carlo Quantum Trajectories

```python
t, states, conc = evolve_mcsolve(params, gamma_E=0.0, tlist=None, rho0=None, ntraj=200)
```

Uses QuTiP `mcsolve`. Each trajectory represents a single experimental shot with stochastic quantum jumps. Ensemble average over `ntraj` trajectories reconstructs the density matrix. Requires a pure-state ket as input; falls back to `evolve_lindblad` for mixed initial states. The Q-A density matrix is reconstructed from nine 2-qubit Pauli expectation values (trace over E).

#### Measurement Simulation

```python
outcomes, counts = measure_state(rho_QA, basis='z', n_shots=1024, save_file=None, seed=None)
```

Simulates projective multi-shot measurement of a 2-qubit Q-A state:
- `basis`: per-qubit measurement basis — `'x'`, `'y'`, `'z'`, or list `['x','z']`
- Outcomes are sampled from the exact probability distribution (multinomial sampling)
- Saves ±1 outcomes per qubit to `measurement_results/<save_file>` if specified
- Returns `outcomes` array of shape (n_shots, 2) and `counts` dict with N_pp/N_pm/N_mp/N_mm

```python
df = full_tomography_shots(rho_QA, n_shots=1024, exp_label="exp", t_label="t0", seed=None)
```

Runs `measure_state` for all 9 Pauli-basis combinations (xx, xy, xz, yx, yy, yz, zx, zy, zz). Saves each to its own `.txt` file and returns a combined DataFrame.

```python
df = load_shots(filepath)
```

Reloads a saved `.txt` shot file and reconstructs the DataFrame with columns `[q_out, a_out, basis_Q, basis_A]`.

#### Quantum State Tomography (MLE)

```python
rho_QA, bloch_Q, bloch_A = qst_log_likelihood(df_tomo)
```

Maximum-likelihood quantum state tomography for a 2-qubit Q-A state:

1. For each qubit independently: maximise log-likelihood over the Bloch vector (r_x, r_y, r_z) given shot counts in each basis (Nelder-Mead optimisation, `maxiter=20000`)
2. Reconstruct the 2-qubit density matrix from all 9 Pauli correlators:

```
ρ = (I⊗I + Σᵢ rᵢ^Q σᵢ⊗I + Σⱼ rⱼ^A I⊗σⱼ + Σᵢⱼ cᵢⱼ σᵢ⊗σⱼ) / 4
```

Returns `rho_QA` (QuTiP Qobj), `bloch_Q` (np.ndarray shape 3), `bloch_A` (np.ndarray shape 3).

#### Quantum Process Tomography

```python
chi = qpt_chi_matrix(channel_fn, n_input_states=4)
```

Standard QPT protocol for a single-qubit channel:
- Input basis states: |0⟩, |1⟩, |+⟩, |+i⟩
- Operator basis: {I, X, iY, Z}
- Solves the linear system β · vec(χ) = vec(λ) via least-squares
- Returns the real 4×4 χ-matrix in the Pauli basis

For the identity channel, χ[0,0] = 1 and all other entries ≈ 0.

#### Zeno Utilities

```python
Gamma_c = gamma_c_zeno(gamma_E, params=None)
# Theoretical Zeno scaling: Ω_QE²/(4γ_E) + Γ_0

Gamma_c = fit_decay_rate(conc, tlist, skip_frac=0.05)
# Fit log(C(t)) ~ −Γt to extract effective decay rate
# skip_frac: fraction of initial points to skip (removes transient)
# Returns np.nan if fewer than 5 valid data points
```

---

### `experiment.py`

Experiment definitions, plotting, and the custom PennyLane circuit runner.

#### `exp1(params, tlist, plot=True)` — Markovian Baseline

Simulates Bell state free decay in a **2-qubit Q-A system only** (E not included), so there is no memory channel. Uses a pure dephasing Lindblad for Q and A.

Returns dict with: `tlist`, `conc`, `conc_analytic`, `NM`, `T2Q`, `T2A`, `Gamma0`.

The analysis window starts from the first time C drops to `C_RECORD = 0.8` to match the paper's convention.

#### `exp2(params, tlist, plot=True, baseline=None)` — Non-Markovian Dynamics

Full 3-qubit simulation with γ_E = 0. Q-E coupling is active and E acts as a coherent quantum memory.

Returns dict with: `tlist`, `conc`, `NM`, `states`, `dC_dt`.

Plots both C(t) and dC/dt (shaded green where dC/dt > 0, i.e., information backflow).

#### `exp3(params, tlist, gamma_scan, plot=True)` — NM→Markovian Transition

Scans γ_E over a list of values (default: 0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0 rad/μs). Computes N(γ_E) for each.

Returns dict with: `gamma_scan`, `all_conc` (dict γ→(t_trim, conc_trim)), `all_NM` (dict γ→N), `tlist`.

#### `exp4(params, tlist, gamma_zeno_scan, plot=True)` — Quantum Zeno Regime

Scans the high-γ_E Zeno regime (default: 0.1 → 45.5 rad/μs, 49 points). Extracts the effective decay rate Γ_c by fitting log(C) vs t and compares to the Zeno scaling prediction.

Returns dict with: `gamma_scan`, `Gamma_c_sim`, `Gamma_c_zeno`, `Gamma0`, `fit_params`.

The plot uses a log-scale x-axis and fits A/γ + B to the post-peak data.

#### `plot_full_dashboard(res1, res2, res3, res4, params, save_pdf=True)` — Summary Figure

Generates a 2×2 GridSpec figure:
- **Panel A:** Exp.1 mesolve + analytic overlay
- **Panel B:** Exp.2 revival + Exp.1 baseline + shaded revival region
- **Panel C:** Exp.3 γ_E scan (γ ≤ 5 shown)
- **Panel D:** Exp.4 Zeno Γ_c scaling on log-x axis

Saved to `simulation_dashboard.pdf`.

#### `run_custom_circuit(circuit_fn, gamma_E, params, tlist, n_shots, plot, label)` — Custom PennyLane Circuit

Execute any user-defined PennyLane 2-qubit circuit as the initial state, then simulate its open-system dynamics:

1. Run the PennyLane circuit on `default.qubit` (2 wires)
2. Convert the output state vector to a QuTiP density matrix
3. Embed into 3-qubit space: ρ_3q = ρ_QA ⊗ |0⟩⟨0|_E
4. Evolve under `evolve_lindblad` with the given γ_E
5. Run QST (1024 shots, MLE) on the initial state and report C_MLE

Returns dict with: `tlist`, `conc`, `NM`, `states`, `initial_rho_QA`, `qst_rho`.

---

### `modulation.py`

Independent lab-frame parametric modulation analysis. Demonstrates how iSWAP couplings are physically realised via flux modulation and Jacobi-Anger expansion.

#### Functions

```python
get_detuning(params, mod_type="QA")
# Returns Δ_QA = ω_Q − ω_A  or  Δ_QE = ω_Q − ω_E

cosine_modulation(t, args)
# Returns cos(ω_m t)  (time-dependent coefficient for H_mod_op)

H, wm = H_modulation_lab(params, mod_type="QA", epsilon=0.0, wm=None)
# Full lab-frame time-dependent Hamiltonian for parametric modulation:
#   Q-A: Q flux modulated at ω_m = Δ_QA/2 → J₂ sideband coupling
#   Q-E: Q and E driven anti-phase at ω_m = Δ_QE/4 → J₄ sideband coupling
# Returns (H_list, wm_used) ready for qt.mesolve

result, wm = run_modulation(params, mod_type, epsilon, n, wm, psi0, tlist, c_ops)
# Single simulation run, returns mesolve result with Q and A/E excitation observables

plot_single_run(params, mod_type, epsilon, n, wm, psi0, tlist)
# Visualise population transfer for one (epsilon, ω_m) setting

eps_list, max_transfer, swap_rate, wm = scan_amplitude(params, mod_type, n, wm, eps_list, psi0, tlist)
# Sweep modulation amplitude epsilon and record:
#   max_transfer    — peak population transferred to target qubit
#   swap_rate       — effective iSWAP rate extracted from oscillation period
#   wm_used         — modulation frequency used

plot_amplitude_scan(params, mod_type, n, wm, eps_list, psi0, tlist)
# Plot max_transfer and swap_rate vs epsilon, report optimal epsilon
```

The Q-A amplitude scan sweeps ε from 0 to 0.5 Δ_QA (30 points).
The Q-E scan sweeps ε from 0 to 1.0 Δ_QE (30 points), covering the J₄ peak and first zero.

---

### `main.py`

Orchestrates all four experiments in sequence and generates the summary PDF.

```
PARAMS["aprox_deg"] = 0   (interaction picture, exact paper reproduction)
TLIST = linspace(0, 10, 400)  μs
GAMMA_SCAN = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
GAMMA_ZENO_SCAN = [0.1, 0.15, …, 45.5]  (49 points, dense at low γ)
```

Runs Exp.1 → Exp.2 → Exp.3 → Exp.4 → dashboard → prints results summary box.

---

## Physics Parameters

All parameters are stored in `DEFAULT_PARAMS` (state_prep.py) and can be overridden by passing a modified copy.

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| `T2Q` | T₂^Q | 39 μs | Q pure dephasing time |
| `T2A` | T₂^A | 41 μs | A pure dephasing time |
| `T2E` | T₂^E | 38 μs | E intrinsic dephasing time |
| `T1Q` | T₁^Q | 31 μs | Q amplitude decay time |
| `T1A` | T₁^A | 32 μs | A amplitude decay time |
| `T1E` | T₁^E | 28 μs | E amplitude decay time |
| `Om_QA` | Ω_QA | 2π × 0.477 rad/μs | Q-A iSWAP coupling rate |
| `Om_QE` | Ω_QE | 2π × 0.473 rad/μs | Q-E iSWAP coupling rate |
| `chi` | χ | 2π × 0.200 rad/μs | ZZ cross-Kerr (dispersive, optional) |
| `delta_QA` | Δ_QA | 2π × 350 rad/μs | Q-A detuning (aprox_deg=1) |
| `delta_QE` | Δ_QE | 2π × 470 rad/μs | Q-E detuning (aprox_deg=1) |
| `om_Q` | ω_Q | 2π × 4650 rad/μs | Q transition frequency |
| `om_A` | ω_A | 2π × 4200 rad/μs | A transition frequency |
| `om_E` | ω_E | 2π × 5370 rad/μs | E transition frequency |
| `aprox_deg` | — | 0 | Approximation degree (0 or 1) |

**Time units:** microseconds (μs)
**Frequency units:** rad/μs = 2π × MHz

**Derived quantities:**

```
Γ_0 = 1/T2Q + 1/T2A ≈ 0.0499 μs⁻¹   (bare Q-A dephasing rate)
t_Bell = π / (4 Ω_QA) ≈ 0.166 μs     (sqrt(iSWAP) gate time)
t_revival ≈ π / Ω_QE ≈ 0.333 μs      (concurrence revival period, aprox_deg=0)
```

---

## Approximation Degrees

Controlled by `params["aprox_deg"]`. Selects both the Hamiltonian and the initial state preparation method.

| `aprox_deg` | Frame | Hamiltonian | Bell prep | When to use |
|-------------|-------|-------------|-----------|-------------|
| `0` (default) | Interaction picture (RWA) | H_QA + H_QE only (time-independent) | Exact Bell state via `qt.bell_state` | Reproduces paper figures; fastest |
| `1` | Doubly-rotating frame | H_interaction + H_detuning (AC Stark corrections) | Time-evolution under H_QA for t = π/(4Ω_QA) | Lab-frame gate definitions, frame transforms |

> **Note on H_lab_parametric:** The full parametric-modulation Hamiltonian (`H_lab_parametric`) requires the hierarchy g_bare ≪ ω_m ≪ Δ. With the current default parameters (Δ_QA/(2π) = 0.35 MHz ≈ g_bare/(2π) = 0.98 MHz), this hierarchy is violated and the RWA fails. To enable `H_lab_parametric`, set `delta_QA = 2π × 50 MHz` and increase `nsteps ≥ 100 000`.

---

## Solvers

| Solver | Function | Method | Best for |
|--------|----------|--------|----------|
| Lindblad (GKSL) | `evolve_lindblad` | QuTiP `mesolve` | Standard open-system simulation; default |
| Bloch-Redfield | `evolve_born_markov` | QuTiP `brmesolve` | Frequency-dependent bath (1/f noise, transmon) |
| Monte Carlo | `evolve_mcsolve` | QuTiP `mcsolve` | Single-trajectory simulation; requires pure state |

All solvers return `(tlist, states, conc)` with the same signature. The Lindblad solver works for both aprox_deg = 0 and 1. The Born-Markov solver uses an Ohmic spectral density by default.

---

## Tomography

### Quantum State Tomography (QST)

**Protocol:** Full 2-qubit QST using 9 Pauli-basis combinations × n_shots measurements each.

```
Bases: {xx, xy, xz, yx, yy, yz, zx, zy, zz}
Total shots: 9 × n_shots
```

**Reconstruction:** MLE on individual qubit Bloch vectors (unconstrained Nelder-Mead), combined with classical Pauli correlators for the 2-qubit density matrix.

**Scaling:** With n_shots = 1024, typical QST fidelity to the exact state is > 99% for C ≈ 1.

### Quantum Process Tomography (QPT)

**Protocol:** 4 input states × single-qubit channel → χ-matrix in {I, X, iY, Z} basis.

For the identity channel: χ[0,0] = 1, all others ≈ 0.
For a dephasing channel: χ[0,0] > 0 and χ[3,3] > 0 (diagonal in Z).

### Measurement Files

Shot files are saved to `measurement_results/` with the naming convention:

```
{exp_label}_{t_label}_{basis_Q}{basis_A}.txt
```

Each file has a header specifying the bases and n_shots, followed by one `q_out a_out` line per shot (±1 values).

---

## Installation

```bash
pip install qutip pennylane numpy scipy matplotlib pandas sympy ipywidgets
```

**Requirements:**
- Python 3.9+
- QuTiP 4.7 or 5.x
- PennyLane 0.36+
- NumPy, SciPy, Matplotlib, Pandas (standard scientific stack)
- ipywidgets (for interactive notebook sliders)

**Tested on:** macOS (Darwin 25.x), Python 3.11, QuTiP 5.0, PennyLane 0.39.

---

## Usage

### Run all four paper experiments

```bash
python main.py
```

Runs Exp.1 through Exp.4 sequentially with the default parameters, displays inline plots, and saves `simulation_dashboard.pdf`. Runtime: approximately 5–15 minutes depending on hardware (Exp.4 Zeno scan is the most expensive with 49 γ values).

### Interactive exploration

Open `interactive_simulation.ipynb` in Jupyter Lab or VS Code.

The notebook has 12 sections with inline plots and interactive widgets:

| Section | Content | Key widget/output |
|---------|---------|-------------------|
| 0 | Imports and setup | — |
| 1 | Parameter configuration | Edit `DEFAULT_PARAMS` |
| 2 | Exp.1 — Markovian baseline | C(t) + analytic overlay, N value |
| 3 | Exp.2 — Non-Markovian revival | C(t) + dC/dt panel, N value |
| 4 | Exp.3 — γ_E scan | Interactive slider over γ_E |
| 5 | Exp.4 — Zeno regime | Results DataFrame, Γ_c vs γ plot |
| 6 | Four-panel dashboard | Calls `plot_full_dashboard()` |
| 7 | aprox_deg=1 demo + gate application | AC Stark shifts, frame transforms |
| 8 | Quantum State Tomography (shot-count scaling) | QST fidelity vs n_shots |
| 9 | Quantum Process Tomography (χ-matrix) | χ heatmap for dephasing channel |
| 10 | Monte Carlo vs mesolve comparison | Trajectory ensemble vs master eq |
| 11 | Bloch-Redfield (Born-Markov) comparison | brmesolve vs mesolve overlay |
| 12 | Custom PennyLane circuit + γ_E sweep slider | User-defined circuit + NM analysis |

### Run a custom PennyLane circuit

```python
from experiment import run_custom_circuit
import pennylane as qml

def my_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

result = run_custom_circuit(my_circuit, gamma_E=1.0)
print(f"N = {result['NM']:.4f}")
print(f"Concurrence at t=0: {result['conc'][0]:.4f}")
```

The circuit must target 2 wires (Q and A) and return `qml.state()`. The environment E is automatically initialised to |0⟩.

### Use individual modules

```python
from state_prep import DEFAULT_PARAMS, prepare_initial_state
from dynamical import evolve_lindblad, full_tomography_shots, qst_log_likelihood
import numpy as np

params = DEFAULT_PARAMS.copy()
tlist  = np.linspace(0, 10, 400)

# Run Lindblad evolution
t, states, conc = evolve_lindblad(params, gamma_E=0.0, tlist=tlist)
print(f"Concurrence at t=0: {conc[0]:.4f}")
print(f"Concurrence at t=10μs: {conc[-1]:.4f}")

# Run QST on the initial state
rho_QA = states[0].ptrace([0, 1])
df = full_tomography_shots(rho_QA, n_shots=1024, exp_label="my_exp", t_label="t0")
rho_mle, bloch_Q, bloch_A = qst_log_likelihood(df)
import qutip as qt
print(f"QST concurrence: {qt.concurrence(rho_mle):.4f}  (exact: {qt.concurrence(rho_QA):.4f})")
```

### Parametric modulation analysis

```python
from modulation import plot_single_run, plot_amplitude_scan
from state_prep import DEFAULT_PARAMS

# Visualise a single Q-A modulation run
plot_single_run(params=DEFAULT_PARAMS, mod_type="QA", epsilon=2*3.14159*0.15)

# Scan amplitude to find the optimal epsilon for Q-E coupling
plot_amplitude_scan(params=DEFAULT_PARAMS, mod_type="QE")
```

### Change approximation degree

```python
from state_prep import DEFAULT_PARAMS, prepare_initial_state, to_stationary_frame
from dynamical import evolve_lindblad
import numpy as np

params = DEFAULT_PARAMS.copy()
params["aprox_deg"] = 1  # doubly-rotating frame with AC Stark corrections

tlist = np.linspace(0, 10, 400)
t, states, conc = evolve_lindblad(params, gamma_E=0.0, tlist=tlist)

# Transform final state to lab (stationary) frame
rho_lab = to_stationary_frame(states[-1], params, t[-1])
```

---

## Expected Results

Running `python main.py` should produce output close to:

```
╔══════════════════════════════════════════════════════╗
║   Gaikwad et al.  PRL 132, 200401 (2024)             ║
║   Non-Markovian → Markovian Transition Simulation    ║
╚══════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════╗
║                 RESULTS SUMMARY                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  Exp.1  N = 0.0000   (expect: 0,   Markovian)                      ║
║  Exp.2  N = 1.4xxx   (paper: ~1.4, non-Markovian)                  ║
║  Exp.3  N(γ=0)  = 1.4xx                                            ║
║         N(γ=10) = 0.0000  (→ 0, Markovian)                         ║
║  Exp.4  Γ₀ = 0.04997 μs⁻¹  (Zeno asymptote)                        ║
╚════════════════════════════════════════════════════════════════════╝
```

Key numerical benchmarks:

| Quantity | Expected value | Paper value |
|----------|---------------|-------------|
| Exp.1 N | ≈ 0.000 | 0 (Markovian) |
| Exp.2 N | ≈ 1.4 | ~1.4 |
| Exp.3 N(γ=0) | ≈ 1.4 | > 0 |
| Exp.3 N(γ=10) | ≈ 0.000 | ≈ 0 |
| Exp.4 Γ₀ = 1/T2Q + 1/T2A | 0.04997 μs⁻¹ | Zeno asymptote |
| Exp.4 Zeno fit A/γ + B | B ≈ Γ₀ | confirms scaling |

---

## Reference

A. Gaikwad, C. Regmi, M. Khurana, N. Dixit, A. Rao, T. Mahesh,
*Phys. Rev. Lett.* **132**, 200401 (2024).
DOI: [10.1103/PhysRevLett.132.200401](https://doi.org/10.1103/PhysRevLett.132.200401)
