# System Architecture: Physics-Constrained Multi-head Network (PCMN)
## Digital Twin for Turbojet Engine Health Monitoring

---

## 1. Problem Understanding & Motivation

### 1.1 The Aerospace Digital Twin Challenge
Aerospace propulsion systems represent one of the most safety-critical, high-consequence engineering domains in existence. A single turbofan engine undergoes millions of thermodynamic micro-cycles per flight hour, and the gradual accumulation of material fatigue, thermal creep, compressor blade fouling, and combustor coking leads to measurable — and predictable — performance degradation long before mechanical failure. The core engineering challenge is: **can we detect, quantify, and predict this degradation trajectory in real-time, using only data collected from installed sensors?**

Traditional approaches to this problem bifurcate into two fundamentally incompatible paradigms:

1. **Pure Physics-Based Approaches**: High-fidelity thermodynamic simulation models (e.g., numerical continuation of the Brayton cycle equations) can, in principle, precisely model engine behavior. However, they are computationally prohibitive for real-time inference — each evaluation requires iterative solvers that take orders of magnitude longer than the operational cycle itself. They are also sensitive to poorly-characterized degradation modes (e.g., how does a 2% compressor blade tip clearance increase affect isentropic efficiency?), and cannot generalize beyond their parametric assumptions.

2. **Pure Data-Driven Approaches**: Modern deep learning has demonstrated remarkable pattern recognition capability across industrial sensor arrays. However, unconstrained neural networks are epistemically unreliable: they will confidently extrapolate in physically impossible directions, predict negative thrust, or simultaneously output a compressor efficiency of 110% and a turbine health of 0.1 — contradictions that violate basic conservation laws. An engineer cannot deploy, certify, or even trust such a model for safety-critical advisory.

### 1.2 The Grey-Box Synthesis
The fundamental motivation of this project is to build a **grey-box Digital Twin** that refuses to sacrifice the strengths of either paradigm. We propose a system that:

- **Learns from data** like a neural network — requiring no hand-crafted, parametric degradation model
- **Obeys physics** like an analytical model — embedding thermodynamic laws directly into the network's architectural constraints and loss formulation
- **Quantifies its own uncertainty** — critical for decision-support in safety-critical systems where an overconfident wrong answer is worse than an uncertain correct one
- **Operates in real-time** — inference must occur at avionics edge-compute timescales (sub-millisecond)

### 1.3 Why Turbojet Engines Specifically?
A turbojet engine operates on the Brayton thermodynamic cycle, consisting of four thermodynamically coupled stages: **Intake → Compression → Combustion → Expansion (Turbine)**. This coupling is a deliberate architectural asset for our approach: the inlet conditions to each stage are the outlet conditions of the previous, creating a web of physical invariants (conservation of mass, energy, and momentum) that a neural network's outputs must respect. Any violation of these invariants is not merely a statistical anomaly — it is a physically impossible prediction that must be structurally prevented.

The turbojet also provides a rich multi-sensor stream: temperatures (T2, T3, T4), pressures (P2, P3, P4), rotational speed (RPM), and fuel flow — all of which have direct physical interpretations in the Brayton cycle. This makes it an ideal testbed for physics-constrained machine learning.

### 1.4 Competitive Context & Academic Rigor
A key methodological requirement is that performance claims must be **statistically defensible**. With a small dataset (300 total cycles across 10 engines), a single train/test split is insufficient to draw reliable conclusions — a fortuitous or unfortunate hold-out split can swing RMSE by factors of 2×. Our entire evaluation framework is therefore built around **Leave-One-Engine-Out Cross-Validation (LOEO-CV)**, and all performance comparisons are subjected to non-parametric statistical significance testing via the **Wilcoxon Signed-Rank Test** with effect size reporting.

---

## 2. Proposed Methodology & Technical Approach

### 2.1 Architectural Overview: A Three-Phase Iterative Design
The final architecture did not emerge from a single design decision. It was developed through three sequential, empirically-validated phases, each motivated by specific observed failures in the prior iteration. Understanding these phases is essential to understanding *why* the final architecture is structured as it is.

```
Phase 1: Hard Physics Constraints (0% TSFC Violation by Design)
    └─> Resolved: Soft penalty loss still allowed ~9% thermodynamic violation
Phase 2: Information Bottleneck Resolution (Combined Feature Space)
    └─> Resolved: Baseline-PhysFeat underperformed Baseline-Raw
Phase 3: Temporal Sequence Modeling (GRU Backbone)
    └─> Resolved: Static point-in-time inference cannot capture degradation trajectories
```

### 2.2 The Thermodynamics Engine (Grey-Box Feature Engineering)

Before any machine learning occurs, raw sensor telemetry passes through a deterministic physics calculation layer — the `ThermodynamicsEngine`. This is not preprocessing in the traditional sense; it is a **grey-box transformation** that converts raw measurements into physical invariants that carry far more semantic information per dimension than the raw sensor values alone.

The `ThermodynamicsEngine` computes the following features from the raw sensor stream `{Tamb, Pamb, T2, P2, T3, P3, T4, P4, RPM, Fuel_Flow, Altitude_m, Mach}`:

#### 2.2.1 Compressor Pressure Ratio (PR_comp)
$$PR_{comp} = \frac{P_2}{P_{amb}}$$
The ratio of compressor exit pressure to ambient inlet pressure. A dimensionless indicator of how hard the compressor is working. Degradation in compressor blades manifests directly in this ratio's relationship to RPM: a fouled compressor produces less pressure per unit of rotational work.

#### 2.2.2 Turbine Pressure Ratio (PR_turb)
$$PR_{turb} = \frac{P_4}{P_3}$$
The pressure ratio across the turbine stage. Since the turbine expands hot combustion gases to extract mechanical work, this ratio characterizes the expansion process. Turbine blade degradation (erosion, coating loss) causes the expansion to become less efficient, detectable as an anomalous relationship between this ratio and the temperature drop across the turbine.

#### 2.2.3 Compressor Isentropic Efficiency ($\eta_c$)
$$\eta_c = \frac{\left(\frac{P_2}{P_{amb}}\right)^{\frac{\gamma-1}{\gamma}} - 1}{\frac{T_2}{T_{amb}} - 1}$$
This is the ratio of **ideal isentropic work** to **actual work** done by the compressor. In an ideal, lossless compressor, this equals 1.0. Compressor degradation (blade fouling, tip clearance increase, surge damage) irreversibly reduces this value. This is the single most informative feature for `CompressorHealth` as it captures the fundamental thermodynamic cost of compression losses.

The constant specific heat ratio $\gamma = 1.4$ is used (air as a calorically perfect gas). A real-gas variant with temperature-dependent $c_p(T)$ and $\gamma(T)$ was also developed and explored during Phase 2 analysis.

#### 2.2.4 Turbine Isentropic Efficiency ($\eta_t$)
$$\eta_t = \frac{1 - \frac{T_4}{T_3}}{1 - \left(\frac{P_4}{P_3}\right)^{0.2857}}$$
Where $0.2857 \approx \frac{\gamma-1}{\gamma}$. This is the turbine-side analogue of $\eta_c$, measuring how effectively the turbine converts enthalpy drop into mechanical shaft work. Turbine health degradation (thermal barrier coating spallation, blade creep) causes this to fall below its design point value.

#### 2.2.5 Net Specific Work ($W_{net}$)
$$W_{net} = c_p \cdot (T_3 - T_4) - c_p \cdot (T_2 - T_{amb}) = W_{turbine} - W_{compressor}$$
Where $c_p = 1005 \, \text{J/(kg·K)}$ is the specific heat at constant pressure for air. This is the net thermodynamic work extracted per unit mass of working fluid. It forms the fundamental driver of thrust production via the momentum theorem: $F_{thrust} = \dot{m}_{air} \cdot \Delta V_{jet}$.

#### 2.2.6 Estimated Air Mass Flow Rate ($\dot{m}_{air}$)
Using the complete energy balance of the Brayton combustor, coupled with the Lower Heating Value of Jet-A fuel ($\text{LHV} = 42.8 \, \text{MJ/kg}$):
$$Q_{combustor} = \dot{m}_{fuel} \cdot \text{LHV} = \dot{m}_{air} \cdot c_p \cdot (T_3 - T_2)$$
$$\therefore \dot{m}_{air} = \frac{\dot{m}_{fuel} \cdot \text{LHV}}{c_p \cdot (T_3 - T_2)}$$
This is a critical derived quantity: air mass flow rate is **not directly measured** by any sensor in our dataset (it requires expensive, intrusive flow measurement equipment). Yet it is fundamental to thrust production. Deriving it from the energy balance enables estimation of a physically meaningful quantity without additional instrumentation.

#### 2.2.7 Estimated Thermal Efficiency ($\eta_{thermal}$)
$$\eta_{thermal} = \frac{\dot{m}_{air} \cdot W_{net}}{\dot{m}_{fuel} \cdot \text{LHV}}$$
This is the complete Brayton cycle thermal efficiency — the fraction of fuel chemical energy that is converted into net shaft/kinetic work. This is the ultimate figure of merit for engine health: a perfectly healthy engine approaches its design-point thermal efficiency; a degraded engine wastes more fuel energy as heat loss.

#### 2.2.8 Collinearity Analysis & Feature Pruning
A critical preprocessing step was **systematic collinearity removal**. Several intuitively useful features were found to be algebraically collinear with the retained features:

- `Combustion_Temp_Rise` ($T_3 - T_2$) is perfectly collinear with `Combustor_Heat_Addition` ($c_p \cdot (T_3 - T_2)$) — same information, different scale
- `Overall_Pressure_Ratio` ($P_3 / P_{amb}$) is perfectly collinear with `PR_comp` ($P_2 / P_{amb}$) given the cascade structure
- `Normalized_RPM` ($RPM / \sqrt{T_{amb}}$) is a corrected speed parameter that, while physically valid, introduced collinearity with `PR_comp` in this dataset (both track engine operating point)

These were dropped from the final feature set to prevent rank deficiency in the feature matrix, which causes $(\mathbf{X}^T \mathbf{X})$ to become near-singular and produces pathological gradient updates during optimization.

**Final ThermodynamicsEngine output features**: `{PR_comp, PR_turb, Comp_Isentropic_Efficiency, Turb_Isentropic_Efficiency, Net_Specific_Work, Estimated_Air_Mass_Flow, Estimated_Thermal_Efficiency, Altitude_m, Mach}` — 9 features.

### 2.3 Phase 1: Hard Physics Constraints & Loss Function Design

#### 2.3.1 The Target Scale Disparity Problem
Raw target values span radically different numeric ranges:
- `CompressorHealth`, `CombustorHealth`, `TurbineHealth`, `OverallHealth` ∈ [0.0, 1.0]
- `Thrust_N` ∈ [10,000–90,000] N
- `TSFC_g_N_s` ∈ [0.01–0.03] g/N/s

Without normalization, the squared error from `Thrust_N` (~$10^9$ per unit error) completely dominates the gradient signal. The optimizer exclusively minimizes Thrust error and completely ignores the health targets, resulting in near-constant predictions for all health metrics.

**Solution**: All 6 targets are independently Z-score normalized using `sklearn.preprocessing.StandardScaler` fitted exclusively on the training split:
$$\hat{y}_{normalized} = \frac{y - \mu_{train}}{\sigma_{train}}$$
The network outputs abstract Z-scores (which can be negative), and all physics constraints are applied after **internal denormalization** to real-world units.

#### 2.3.2 PhysicsConstrainedLoss Architecture
The `PhysicsConstrainedLoss` (`loss.py`) implements a composite loss that enforces two simultaneous physics constraints:

**Loss Formulation:**
$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{MSE} + \beta_{health} \cdot \mathcal{L}_{Health\_Consistency}$$

**Component 1 — Multi-Head MSE:**
$$\mathcal{L}_{MSE} = \frac{1}{5} \sum_{k \in \{c, b, t, o, thrust\}} \text{MSE}(\hat{y}_k, y_k)$$
This balances learning across all five prediction heads in normalized space, ensuring no single head dominates.

**Component 2 — Thermodynamic Health Consistency:**
$$\mathcal{L}_{Health} = \text{MSE}\left(\hat{h}_{overall,\, norm}, \; \frac{0.40 \cdot \hat{h}_{comp, real} + 0.30 \cdot \hat{h}_{turb, real} + 0.30 \cdot \hat{h}_{comb, real} - \mu_{overall}}{\sigma_{overall}}\right)$$
This constraint enforces a weighted aggregation relationship between subsystem and overall health. The weights (0.40 Compressor, 0.30 Turbine, 0.30 Combustor) reflect aerospace engineering importance rankings. The constraint operates on **internally denormalized predictions** (real-world health values), then **re-normalizes the expected overall** before computing the penalty — ensuring the gradient remains in the same scale as the MSE term.

**Component 3 — Hard TSFC Constraint (0% Violation by Design):**
Rather than including TSFC as a network output and penalizing violations, TSFC is removed from the network's output heads entirely. The network predicts `Thrust_N`, and TSFC is computed **deterministically post-inference**:
$$TSFC_{real} = \frac{\dot{m}_{fuel} \; [\text{g/s}]}{Thrust_{real} \; [\text{N}]}$$
This transforms the TSFC constraint from a *learned* behavior (which a soft penalty can violate) into a **structural tautology** — mathematically guaranteed to hold for every single prediction, at every single inference call, with 0% violation by construction.

**Hyperparameters:** `alpha=1.0`, `beta_health=1.0`

### 2.4 Phase 2: The Information Bottleneck & Combined Feature Space

#### 2.4.1 The Anomalous Inversion Finding
During initial benchmarking, a counter-intuitive result emerged: `Baseline-Raw` (fed 12 raw sensor columns) consistently outperformed `Baseline-PhysFeat` (fed the 9 thermodynamic invariants computed from those same sensors) in terms of RMSE. If physics features encode more semantically meaningful information, this should not be possible.

#### 2.4.2 The Information Bottleneck Hypothesis
The resolution was the **Information Bottleneck Hypothesis**: the `ThermodynamicsEngine`'s feature extraction is a **lossy, many-to-one transformation**. 

Consider `Comp_Isentropic_Efficiency = f(Tamb, T2, Pamb, P2)`. A single scalar encodes 4 input variables. Any nuanced interaction between, say, the absolute level of P2 and the ambient humidity correction is irreversibly compressed into one number. The neural network fed only the physics features has no way to recover the information that was discarded by this compression — and the training data (only 300 cycles) does not provide enough samples to learn to compensate.

Concretely: 8 raw sensor columns → 9 physics features **discards information** in the compression. The MLP fed raw sensors retains this information and can exploit it directly.

#### 2.4.3 The Combined Feature Space Resolution
The fix is **not** to abandon physics features (they provide critical interpretability and constrain the solution space). Instead, we **concatenate** both representations:

$$\mathbf{x}_{combined} = [\mathbf{x}_{raw,\; 12} \; \| \; \mathbf{x}_{phys,\; 7}]^T \in \mathbb{R}^{19}$$

Where `||` denotes concatenation. The 7 physics features exclude `Altitude_m` and `Mach` since those are already present in the raw sensor set.

This gives the network:
- All 12 raw sensor readings (preserving the original information)
- 7 pre-computed physics invariants (providing gradient shortcuts to semantically meaningful combinations)

The network can now learn to exploit both: using the raw values for fine-grained regression and the physics features as direct proxies for subsystem health indicators.

#### 2.4.4 Statistical Validation of the Resolution
The Combined model was evaluated under LOEO-CV. The Wilcoxon Signed-Rank Test between `Combined` and `Baseline-Raw` RMSE vectors produced **p = 0.8457**, establishing that the `Combined` approach achieves statistical parity with the raw baseline — eliminating the inversion anomaly. More importantly, it retains full thermodynamic interpretability while adding no performance penalty.

### 2.5 Phase 3: Gated Recurrent Unit (GRU) Backbone for Temporal Degradation

#### 2.5.1 The Markovian Nature of Degradation
Engine degradation is inherently a **Markovian stochastic process** — the health state at cycle $t+1$ is a function of the health state at cycle $t$ plus any new damage accumulated during that cycle. A static MLP that maps $\mathbf{x}_t \rightarrow \hat{y}_t$ treats each cycle as an independent, identically distributed sample. It discards all temporal context.

This fundamentally misrepresents the physics of degradation:
- A single anomalous cycle could be sensor noise or could be the onset of accelerated deterioration — indistinguishable without temporal context
- The *rate of change* of efficiency (e.g., $\Delta\eta_c$ over the last 5 cycles) is at least as informative as the absolute current value
- Degradation trends compound: a slowly-declining compressor efficiency accelerates turbine loading, which in turn accelerates turbine degradation — a temporal cascade

#### 2.5.2 GRU Architecture
The `DigitalTwinModel` (`pinn.py`) was upgraded from a static 3-layer MLP to a `model_type='gru'` variant using PyTorch's `nn.GRU` with `batch_first=True`.

**Input**: 3D tensor of shape `(batch_size, seq_len, n_features)` where `seq_len = 5` and `n_features = 19` (Combined space).

**GRU Recurrent Layer**: `nn.GRU(input_size=19, hidden_size=32, num_layers=1, batch_first=True)`

The GRU maintains a **hidden state** $h_t \in \mathbb{R}^{32}$ that is propagated across all $N=5$ timesteps:
$$h_t = \text{GRU}(x_t, h_{t-1})$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \quad \text{(candidate state)}$$

The two learnable gates control information flow: the **reset gate** $r_t$ determines how much past hidden state to forget; the **update gate** $z_t$ determines how much of the candidate state to write into the hidden state. This gives the GRU the ability to learn both short-range (cycle-to-cycle) and longer-range (multi-cycle) dependencies without the vanishing gradient problems of vanilla RNNs.

**Extraction**: Only the **final hidden state** $h_N$ is used — the summary of the entire $N$-length sequence context:
```python
gru_out, h_n = self.gru(x)  # x: (batch, 5, 19)
x = h_n[-1]                  # h_n[-1]: (batch, 32)
```

**Post-GRU Processing**: A single dense layer followed by MC Dropout maps the hidden state into the shared representation space used by all output heads.

**Output Heads**: 5 independent linear layers (`compressor_head`, `combustor_head`, `turbine_head`, `overall_health_head`, `thrust_head`), each mapping `(batch, 32) → (batch, 1)`.

#### 2.5.3 MLP Flattening Baseline for Ablation
The ablation comparison is not GRU vs. a point-in-time MLP, but GRU vs. an MLP fed **flattened temporal context**. The MLP baseline receives the same $N$-length sequence as a single flattened vector of dimension $N \times 19 = 5 \times 19 = 95$. This is the canonical "hand-engineered rolling features" approach — the MLP can theoretically learn the same temporal patterns if sufficient parameters are available. The ablation tests whether the inductive bias of GRU's recurrent structure provides a measurable advantage over flat concatenation on this dataset.

#### 2.5.4 Sliding Window Sequence Extraction
The `extract_sequences()` function (`dataset.py`) generates the sequence tensors required for the GRU with the following algorithm:

**For each engine:**
1. If `num_cycles >= N`: Apply a sliding window of length $N$ with stride 1, yielding `num_cycles - N + 1` overlapping windows per engine. Each window's target label is the ground truth at the final timestep $t+N-1$.
2. If `num_cycles < N` (cold-start edge case): Apply zero-padding with $N - \text{num\_cycles}$ leading zero vectors, and generate a **binary mask** tensor $M \in \{0, 1\}^N$ indicating which timesteps are real vs. padded.

```
Engine with 3 cycles, N=5:
  Input:  [0, 0, x_1, x_2, x_3]   (zero-padded)
  Mask:   [0, 0,  1,   1,   1]   (1 = real data)
  Target: y_3                     (final real cycle's ground truth)
```

This explicit masking prevents cold-start crashes (where an engine with 1 or 2 cycles would fail to form any window), and allows the loss function to optionally weight padded timesteps differently.

#### 2.5.5 Monte Carlo Dropout Uncertainty Quantification
Both MLP and GRU variants employ **MC Dropout** (`nn.Dropout(p=0.1)`) as a Bayesian approximation to a Gaussian Process posterior. The key insight is that dropout at *inference time*, rather than only at training time, effectively samples from an ensemble of different network architectures — each corresponding to a different subset of dropped neurons.

The `predict_with_uncertainty()` method runs $K=10$ stochastic forward passes (with dropout active via `model.train()` mode) and computes:
$$\mu(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^K \hat{y}^{(k)}(\mathbf{x}), \quad \sigma(\mathbf{x}) = \sqrt{\frac{1}{K}\sum_{k=1}^K \left(\hat{y}^{(k)}(\mathbf{x}) - \mu\right)^2}$$

The denormalized standard deviation $\sigma \cdot \sigma_{train}$ represents the **1-sigma confidence bound** on each prediction. A well-calibrated model should have approximately 68% of true ground-truth values fall within $[\mu - \sigma, \mu + \sigma]$.

### 2.6 Leave-One-Engine-Out Cross-Validation (LOEO-CV) Framework

The evaluation framework (`benchmark.py`) implements a strict **LOEO-CV** harness to eliminate any single-split statistical noise:

```
For engine_i in {Engine 1, ..., Engine 10}:
    Train on all cycles from Engines \{1,...,10} \ {i}
    Test on all cycles from Engine i
    Record: RMSE_health(i), TSFC_RMSE(i)

Final metrics: mean ± std across 10 held-out engines
Significance: Wilcoxon Signed-Rank Test (paired by engine fold)
```

**Why LOEO, not k-Fold?** With only 300 total samples split across 10 engines (30 cycles each), each engine's degradation trajectory is a correlated time series. Random k-fold would inadvertently place some cycles from Engine 3 in train and others in test — training on the future to predict the past, an egregious form of data leakage. LOEO guarantees that at test time, the model has never seen *any* cycle from the held-out engine.

**Statistical Testing Protocol**: The Wilcoxon Signed-Rank Test is preferred over a paired t-test because the distribution of per-fold RMSE values is not guaranteed to be Gaussian with only 10 folds. The Wilcoxon test is a non-parametric rank-based test that makes no distributional assumptions. Effect size is computed as the matched-pairs rank-biserial correlation.

---

## 3. Data Sources & Preprocessing

### 3.1 Dataset Overview
The primary dataset consists of simulated turbojet engine telemetry from **10 distinct engines**, each operated for **30 operational cycles**, yielding **300 total annotated samples**. The dataset is structured across three CSV files:

| File | Description | Rows |
|:---|:---|:---|
| `train.csv` | Engine cycles 1–8, raw sensor readings | 240 |
| `test.csv` | Engine cycles 9–10, raw sensor readings | 60 |
| `ground_truth.csv` | Health & performance labels for all cycles | 300 |

The `load_and_merge_data()` function (`dataset.py`) performs an `inner join` on `{EngineID, Cycle}` to ensure perfect alignment between sensor readings and ground truth labels. Duplicate rows (if any overlap between train and test) are removed by de-duplicating on `{EngineID, Cycle}`.

### 3.2 Raw Sensor Schema
Each row represents one operational cycle of one engine. The **12 raw sensor columns** are:

| Column | Physical Meaning | Unit | Range (approx.) |
|:---|:---|:---|:---|
| `Tamb` | Ambient temperature at inlet | K | 220–300 K |
| `Pamb` | Ambient static pressure | Pa | 26,500–101,325 Pa |
| `T2` | Compressor exit temperature | K | 320–550 K |
| `P2` | Compressor exit pressure | Pa | 100,000–1,200,000 Pa |
| `T3` | Combustor exit temperature | K | 1,000–3,500+ K |
| `P3` | Combustor exit pressure | Pa | 90,000–1,100,000 Pa |
| `T4` | Turbine exit temperature | K | 700–2,500 K |
| `P4` | Turbine exit pressure | Pa | 50,000–600,000 Pa |
| `RPM` | Shaft rotational speed | rev/min | 5,000–15,000 |
| `Fuel_Flow` | Fuel mass flow rate | kg/s | 0.5–3.0 |
| `Altitude_m` | Operational altitude | m | 0–12,000 m |
| `Mach` | Flight Mach number | dimensionless | 0.0–0.85 |

### 3.3 Target / Label Schema
The **6 target variables** provided in `ground_truth.csv`:

| Column | Physical Meaning | Unit | Scale |
|:---|:---|:---|:---|
| `CompressorHealth` | Compressor component health index | dimensionless | [0, 1] |
| `CombustorHealth` | Combustor component health index | dimensionless | [0, 1] |
| `TurbineHealth` | Turbine component health index | dimensionless | [0, 1] |
| `OverallHealth` | Engine-level health index | dimensionless | [0, 1] |
| `Thrust_N` | Net thrust produced | N | ~10,000–90,000 N |
| `TSFC_g_N_s` | Thrust-Specific Fuel Consumption | g/N/s | ~0.010–0.030 |

**Note on TSFC**: As a result of Phase 1's hard physics constraint, `TSFC_g_N_s` is **no longer a network output**. It is computed post-inference as `TSFC = Fuel_Flow_g_s / Thrust_N`. The label is still retained in the dataset for evaluation scoring only.

### 3.4 Column Renaming & Standardisation
The raw CSV uses verbose column names from the simulation framework. `load_and_merge_data()` applies the following mapping to match the `ThermodynamicsEngine`'s expected interface:

```python
rename_map = {
    'Tamb_K':       'Tamb',
    'Pamb_Pa':      'Pamb',
    'T2_K':         'T2',
    'P2_Pa':        'P2',
    'T3_K':         'T3',
    'P3_Pa':        'P3',
    'T4_K':         'T4',
    'P4_Pa':        'P4',
    'RPM_rev_min':  'RPM',
    'FuelFlow_kg_s':'Fuel_Flow'
}
```

### 3.5 Leakage-Free Train/Validation/Test Split
The dataset split is **engine-grouped**, not row-randomized. All cycles from an engine are assigned together to a single partition. This is mandatory because the cycles within one engine form a correlated time series — a random row split would leak future degradation information from one engine into training.

**Canonical Evaluation Split** (used in `train.py` for final model training):
- **Train**: Engines 1–6 → 180 cycles
- **Validation**: Engines 7–8 → 60 cycles (used exclusively for early stopping)
- **Test** (blind): Engines 9–10 → 60 cycles (never touched during training or tuning)

**LOEO-CV Split** (used in `benchmark.py` for evaluation):
- Each of the 10 engines serves as the test fold in turn
- Training uses all remaining 9 engines (270 cycles)
- A fresh `StandardScaler` is fitted on the 270-cycle training split each fold

### 3.6 Normalization Protocol

#### 3.6.1 Feature Normalization
All input features (both raw and physics-derived) are independently Z-score normalized using a `StandardScaler` fitted **exclusively on the training partition** of each fold. The same scaler is then applied `.transform()` (not `.fit_transform()`) to the validation and test partitions, ensuring no test statistics contaminate the normalization.

$$x_{scaled} = \frac{x - \mu_{train}}{\sigma_{train}}$$

#### 3.6.2 Target Normalization
All 6 targets are independently Z-score normalized using a separate `target_scaler`, again fitted only on training data. The network outputs Z-scored predictions. Denormalization at inference time:

$$\hat{y}_{real} = \hat{y}_{scaled} \cdot \sigma_{train} + \mu_{train}$$

This is performed inside the `PhysicsConstrainedLoss` during training (for the health consistency constraint) and explicitly in the evaluation code and telemetry streamer at inference time.

#### 3.6.3 Why Z-Score over Min-Max?
Z-score normalization is preferred over min-max [0,1] normalization because:
1. Min-max is sensitive to outliers in the training split (one anomalous cycle sets the global scale)
2. The targets (particularly `Thrust_N`) do not have theoretically bounded extremes — a degraded engine could potentially produce thrust below any observed minimum
3. Z-score preserves the full dynamic range of the distribution and naturally handles out-of-distribution values during inference (they simply produce large-magnitude Z-scores, which the network can recognize as anomalous)

### 3.7 Temporal Sequence Assembly

For the GRU, raw rows are transformed into 3D sequence tensors. The `extract_sequences()` function (`dataset.py`) enforces **strict engine boundary respect** — sequences never span two different engines:

```python
for engine_id, group in df.groupby('EngineID', sort=False):
    # Only form windows within this engine's lifecycle
    for i in range(num_cycles - seq_length + 1):
        X_seq.append(X_group[i : i + seq_length])    # (5, 19)
        y_seq.append(y_group[i + seq_length - 1])    # label at t+N-1
```

**Edge case handling for cold-start engines** (`num_cycles < seq_length`):
```python
pad_len = seq_length - num_cycles
X_pad = np.zeros((pad_len, n_features), dtype=np.float32)
X_padded = np.vstack([X_pad, X_group])              # (5, 19) with leading zeros
mask = np.concatenate([np.zeros(pad_len), np.ones(num_cycles)])  # [0,0,1,1,1]
```

### 3.8 DataLoader & Batch Collation
PyTorch `TensorDataset` bundles `(X_seq, y_scaled, fuel_flow, mask)` into a batched format. The `DataLoader` uses `batch_size=64` with `shuffle=True` for training (to break temporal autocorrelation between successive batches) and `shuffle=False` for validation and test.

---

## 4. Expected Outcomes & Evaluation Metrics

### 4.1 Primary Evaluation Metric: LOEO-CV Health RMSE
The primary competition metric is **Root Mean Square Error on Overall Health** across the LOEO-CV folds:

$$\text{RMSE}_{health} = \sqrt{\frac{1}{|fold|} \sum_{i \in fold} \left(\hat{h}_{overall,i} - h_{overall,i}\right)^2}$$

Reported as: **mean ± std** across 10 LOEO folds. This is the most honest representation of expected performance on an unseen engine.

### 4.2 Phase 3 GRU Ablation Results (LOEO-CV)

| Model | Backbone | Seq Length N | Health RMSE (mean ± std) | TSFC RMSE | p-value (vs MLP) |
|:---|:---:|:---:|:---:|:---:|:---:|
| MLP | Dense 3×32 | N=3 | 0.0350 ± 0.0101 | 0.0027 | — |
| **GRU** | Recurrent | N=3 | **0.0290 ± 0.0075** | **0.0019** | **0.0039** |
| MLP | Dense 3×32 | N=5 | 0.0420 ± 0.0065 | 0.0026 | — |
| **GRU** | Recurrent | N=5 | **0.0307 ± 0.0078** | **0.0019** | **0.0020** |

Both GRU variants outperform their respective MLP counterparts with statistical significance (p < 0.05 under two-tailed Wilcoxon Signed-Rank Test). The GRU's recurrent inductive bias — explicitly sharing hidden state across the temporal sequence — provides a measurable, statistically robust advantage over dense concatenation of rolling context.

### 4.3 Thermodynamic Compliance Metric: TSFC Violation
By construction of the hard physics constraint (Phase 1), the TSFC Violation Rate is **0.000% for all models** that use the deterministic TSFC computation. This is not an empirical result — it is a mathematical guarantee. It is reported as a design property, not a benchmark number, to distinguish our approach from unconstrained models that could exhibit any violation rate.

### 4.4 Surrogate Speedup Benchmark
The network is benchmarked against the `ThermodynamicsEngine.extract_physics_features()` method for computational efficiency:

- **Slow path**: Row-wise physics calculation (sequential DataFrame operations) for all N=300 samples
- **Fast path**: Batched GRU inference on the entire dataset

Expected speedup: **>2000×**, operating at sub-microsecond per-sample latency. This demonstrates **edge deployment viability** for avionics-grade embedded systems where real-time sensor fusion must occur at flight computer cycle rates.

### 4.5 Uncertainty Calibration
The MC Dropout uncertainty is reported as **empirical coverage rate**: the fraction of true ground-truth values that fall within the predicted 1-sigma (±1 std) bounds across all test samples. A perfectly calibrated Gaussian predictor should achieve **68% coverage** at 1-sigma. The model's coverage rate represents the degree to which its expressed uncertainty is reliable — low coverage means overconfidence (bounds too tight), high coverage means underconfidence (bounds too wide).

---

## 5. Final Architecture Summary

```
INPUT: Raw Sensor Stream {Tamb, Pamb, T2, P2, T3, P3, T4, P4, RPM, Fuel_Flow, Altitude, Mach}
           |
           v
    ThermodynamicsEngine (deterministic)
    -> {PR_comp, PR_turb, eta_c, eta_t, W_net, m_dot_air, eta_thermal, Altitude, Mach}
           |
           v
    Feature Concatenation [raw_12 || phys_7] = x_combined ∈ R^19
           |
           v
    StandardScaler (Z-score, fitted on train split only)
           |
           v
    Sliding Window Buffer [x_t-4, x_t-3, x_t-2, x_t-1, x_t] ∈ R^{5x19}
    (Zero-padded for cold-start cycles; "Insufficient History" flag if < N cycles)
           |
           v
    GRU (input=19, hidden=32, layers=1, batch_first=True)
    -> final hidden state h_N ∈ R^32
           |
           v
    MC Dropout (p=0.1) + Dense 32→32 + ReLU + MC Dropout
           |
      _____|_____
     |           |
     v           v
  [comp_h]  [comb_h]  [turb_h]  [overall_h]  [thrust]   <- 5 independent linear heads
     |           |         |          |            |
     v           v         v          v            v
  Z-score outputs -> denormalize to real units
                                                   |
                                                   v
                                           TSFC = Fuel_Flow_g/Thrust  (deterministic)
           |
           v
  PhysicsConstrainedLoss:
     L_total = alpha * L_MSE + beta_health * L_Health_Consistency
```

**Parameter count**: ~3,200 trainable parameters (GRU: 3×32×(19+32+1)×4 = ~5,888; plus output heads 5×33 = 165; plus post-GRU dense = 32×32+32 = 1,056). A deliberately minimal footprint for edge deployment.

**Training**: Adam optimizer, lr=0.001, weight_decay=1e-4, early stopping with patience=20 on validation MSE, max 300 epochs.

## 6. Next Steps: Cross-Architecture Representation Transfer (Phase 1)

While the baseline Physics-Constrained Digital Twin achieves strong performance, it is constrained by a small supervised dataset (300 rows). To address this, Phase 1 introduces a rigorous methodology to transfer thermodynamic representations from the massive N-CMAPSS turbofan dataset.

### 6.1 Research Hypothesis & Success Criteria

**H1 (Alternative):** The encoder learns representations that are transferable across gas-turbine architectures because fundamental thermodynamic relationships are shared.
**H0 (Null):** Representations are architecture-specific and do not transfer.

**Success Levels:**
- **Level 0 (Failure):** No RMSE improvement, no convergence improvement, no latent alignment -> Conclude transfer failed.
- **Level 1:** RMSE unchanged, BUT training converges much faster -> Positive result.
- **Level 2:** RMSE improves AND convergence improves -> Excellent.
- **Level 3:** RMSE, uncertainty calibration, and generalization all improve -> Outstanding.

**Abort Criterion:** If a Linear Probe (frozen pretrained encoder + trained linear head) performs no better than random initialization after N epochs, further transfer-learning experiments will be discontinued to prevent sunk-cost fallacy.

### 6.2 The Transfer Architecture

To prevent the network from learning false physics (e.g., equating a turbofan fan speed to a turbojet core speed), we introduce domain-specific adapters projecting into an **Engine State Representation Space**.

1. **Multi-Regime Sampling:** Extract 128-timestep sequences across Takeoff, Climb, Cruise, and Descent from N-CMAPSS to capture all transient degradation signatures.
2. **N-CMAPSS Adapter:** Linear(14, 64) -> ReLU -> LayerNorm -> Linear(64, 32)
3. **Turbojet Adapter:** Linear(19, 64) -> ReLU -> LayerNorm -> Linear(64, 32)
4. **Shared Backbone:** A shared GRU operating exclusively on the 32-dimensional Engine State Representation Space.

### 6.3 Contrastive Learning Objective

To prevent 'shortcut learning' common in masked reconstruction (e.g., predicting constant fuel flow during cruise), we utilize Contrastive Time-Series Representation Learning.
- **Positive Pairs:** Overlapping windows, nearby windows from the same flight, temporal neighbors.
- **Negative Pairs:** Different engines, distant flights, different operating regimes.

### 6.4 Experimental Ablations & Diagnostics

The evaluation will isolate variables to prove *why* performance improves:

- **Baseline Hierarchy:**
  - *Baseline A:* Random initialization -> Fine-tune.
  - *Baseline B:* N-CMAPSS transfer -> Fine-tune.
- **Control Experiments:** 
  - Pretrain on randomly shuffled N-CMAPSS (destroying temporal order). If performance matches Baseline B, temporal representations were not driving the transfer.
  - Freeze adapter vs. freeze encoder to locate where transfer happens.
- **Ablations:** Sequence lengths (32, 64, 128) and Adapter depths (64->32 vs. 32->16).
- **Representation Diagnostics:** Latent spaces will be evaluated quantitatively using Cosine Similarity, Silhouette Score, and Cluster Purity, alongside qualitative visualization via UMAP and PCA.
- **Representation Drift:** Track the distance between the pretrained latent space and the fine-tuned latent space to quantify forgetting during layer-wise fine-tuning.

### 6.5 Research Risks & Mitigations

| Risk | Mitigation |
|:---|:---|
| No transfer | Early Abort via Linear Probe, pivot to TurboJetSim |
| Negative transfer | Adapter tuning, learning rate adjustments |
| Domain mismatch | Layer-wise fine-tuning |
| Small downstream dataset | Physics-constrained loss + LOEO-CV |
| Overfitting | Early stopping + MC Dropout |

