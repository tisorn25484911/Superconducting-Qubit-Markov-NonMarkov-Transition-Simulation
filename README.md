# Open Quantum System Simulation Platform

Numerical reproduction of:

> Gaikwad et al., *"Entanglement Assisted Probe of the Non-Markovian to Markovian
> Transition in Open Quantum System Dynamics"*,
> **Phys. Rev. Lett. 132, 200401 (2024)**

---

## Overview

This platform simulates a three-qubit open quantum system — **Q** (system qubit),
**A** (ancilla), and **E** (environment) — in which the E qubit acts as a tunable
memory channel. By sweeping the dephasing rate γ_E on E, the dynamics of the Q-A
entanglement (measured by Wootters concurrence) can be tuned continuously from
non-Markovian (information backflow, concurrence revival) to Markovian (monotonic
exponential decay).

The four paper experiments are reproduced numerically:

| # | Experiment | Key result |
|---|-----------|------------|
| 1 | Markovian baseline — Bell state free decay, no Q-E coupling | N = 0 (monotonic) |
| 2 | Non-Markovian dynamics — Q-E coupling, γ_E = 0 | N > 0, concurrence revival |
| 3 | NM → Markovian transition — γ_E scan | N decreases continuously to 0 |
| 4 | Quantum Zeno regime — large γ_E | Γ_c = Ω²_QE/(4γ_E) + Γ_0 scaling |

---

## File Structure

```
.
├── state_prep.py              # Operators, Hamiltonians, frame transforms, Bell state prep
├── dynamical.py               # Solvers (Lindblad, Born-Markov, MC), QST, QPT, measurement
├── experiment.py              # exp1()–exp4(), dashboard, custom PennyLane circuit runner
├── main.py                    # Runs all four paper experiments sequentially
├── interactive_simulation.ipynb  # 12-section interactive notebook (imports .py modules)
├── measurement_results/       # Auto-generated: multi-shot tomography .txt files
└── simulation_dashboard.pdf   # Auto-generated: four-panel summary figure
```

### Module responsibilities

**`state_prep.py`**
- Single- and multi-qubit operators (σ±, σz, identity) embedded in the 8-dimensional Q⊗A⊗E space
- Hamiltonians: `H_free`, `H_QA_int`, `H_QE_int`, `H_interaction`, `H_detuning`
- `H_for_simulation(params)` — selects approximation degree
- `prepare_initial_state(params)` — Bell state |Ψ⁺⟩ via sqrt(iSWAP) gate on |10,0⟩_QAE
- Frame transforms: `to_stationary_frame`, `to_rotating_frame`
- `nm_measure(conc, tlist)` — Rivas-Huelga-Plenio non-Markovianity N
- `concurrence_manual(rho)` — Wootters concurrence without QuTiP dependency

**`dynamical.py`**
- `evolve_lindblad` — GKSL master equation via `mesolve`
- `evolve_born_markov` — Bloch-Redfield equation via `brmesolve`
- `evolve_mcsolve` — Monte Carlo quantum trajectories via `mcsolve`
- `measure_state` — projective multi-shot measurement with basis rotation, saves to `.txt`
- `full_tomography_shots` — 9 Pauli-basis combinations for full QST
- `qst_log_likelihood` — MLE quantum state tomography (single- and two-qubit)
- `qpt_chi_matrix` — χ-matrix reconstruction via quantum process tomography
- `gamma_c_zeno`, `fit_decay_rate` — Zeno scaling utilities

**`experiment.py`**
- `exp1()` through `exp4()` — individual paper experiments with inline plots
- `plot_full_dashboard()` — four-panel GridSpec figure, saved as `simulation_dashboard.pdf`
- `run_custom_circuit(circuit_fn, ...)` — execute any PennyLane circuit as initial state, then run open-system evolution

---

## Physics Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| T2Q | 39 μs | Q dephasing time |
| T2A | 41 μs | A dephasing time |
| Ω_QA | 2π × 0.477 rad/μs | Q-A iSWAP coupling |
| Ω_QE | 2π × 0.473 rad/μs | Q-E iSWAP coupling |
| χ | 2π × 0.200 rad/μs | ZZ cross-Kerr (optional) |
| γ_E | 0 … 45.5 rad/μs | Tunable E dephasing rate |

**Qubit ordering:** 0 = Q, 1 = A, 2 = E (Hilbert space Q⊗A⊗E, dimension 8)

**Time units:** microseconds (μs); frequency units: rad/μs

**Bell state preparation:** evolve |10,0⟩_QAE under H_QA for t = π/(4Ω_QA) — the
sqrt(iSWAP) half-period — to obtain (|10⟩ − i|01⟩)/√2 with C = 1.

**Non-Markovianity measure:**
```
N = ∫ max(dC/dt, 0) dt
```
N > 0 signals information backflow from E back to Q-A.

**Quantum Zeno scaling:**
```
Γ_c = Ω²_QE / (4 γ_E) + Γ_0,   Γ_0 = 1/T2Q + 1/T2A
```

---

## Approximation Degrees

Controlled by `params["aprox_deg"]`:

| `aprox_deg` | Frame | Hamiltonian |
|-------------|-------|-------------|
| `0` (default) | Interaction picture | H_QA + H_QE only (exact paper reproduction) |
| `1` | Doubly-rotating frame | + detuning terms at Δ_QA/2 and Δ_QE/4 (realistic lab-frame demo) |

---

## Installation

```bash
pip install qutip pennylane numpy scipy matplotlib pandas sympy ipywidgets
```

Python 3.9+ recommended. Tested with QuTiP 4.7 / 5.x and PennyLane 0.36+.

---

## Usage

### Run all four paper experiments

```bash
python main.py
```

Outputs a printed results summary and saves `simulation_dashboard.pdf`.

### Interactive exploration

Open `interactive_simulation.ipynb` in Jupyter Lab or VS Code.

The notebook has 12 sections:

| Section | Content |
|---------|---------|
| 0 | Imports and setup |
| 1 | Parameter configuration |
| 2 | Exp.1 — Markovian baseline |
| 3 | Exp.2 — Non-Markovian revival |
| 4 | Exp.3 — γ_E scan with interactive slider |
| 5 | Exp.4 — Zeno regime with results DataFrame |
| 6 | Four-panel dashboard |
| 7 | aprox_deg=1 demo + gate application |
| 8 | Quantum State Tomography (shot-count scaling) |
| 9 | Quantum Process Tomography (χ-matrix) |
| 10 | Monte Carlo vs mesolve comparison |
| 11 | Bloch-Redfield (Born-Markov) comparison |
| 12 | Custom PennyLane circuit + γ_E sweep slider |

### Run a custom circuit

```python
from experiment import run_custom_circuit
import pennylane as qml

def my_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

result = run_custom_circuit(my_circuit, gamma_E=1.0)
print(f"N = {result['NM']:.4f}")
```

### Use individual modules

```python
from state_prep import DEFAULT_PARAMS, prepare_initial_state
from dynamical import evolve_lindblad, qst_log_likelihood
import numpy as np

params = DEFAULT_PARAMS.copy()
tlist  = np.linspace(0, 10, 400)

t, states, conc = evolve_lindblad(params, gamma_E=0.0, tlist=tlist)
print(f"Concurrence at t=0: {conc[0]:.4f}")
```

---

## Expected Results

```
Exp.1  N ≈ 0.000   (Markovian, no revival)
Exp.2  N ≈ 1.4     (paper value; non-Markovian revival)
Exp.3  N(γ=0) > 0, N(γ=10) ≈ 0  (transition)
Exp.4  Γ_c → Γ_0 = 1/T2Q + 1/T2A ≈ 0.0499 μs⁻¹  (Zeno asymptote)
```

---

## Reference

A. Gaikwad, C. Regmi, M. Khurana, N. Dixit, A. Rao, T. Mahesh,
*Phys. Rev. Lett.* **132**, 200401 (2024).
