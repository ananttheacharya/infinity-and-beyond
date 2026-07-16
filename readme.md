# Zero and Already Behind — Definitive Technical Documentation

## Physics-Constrained Multi-Head Network (PCMN) Digital Twin for Four-Stage Single-Spool Turbojet Engine Health Monitoring

**Project:** IIT Indore × HAL, Statement #2  
**Document Revision:** 1.0  
**Date:** July 15, 2026  
**Classification:** Complete Engineering & Research Reference  

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Complete System Overview](#2-complete-system-overview)
3. [Repository Architecture](#3-repository-architecture)
4. [Dataset Documentation](#4-dataset-documentation)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Complete Model Architecture](#7-complete-model-architecture)
8. [Physics-Informed Learning](#8-physics-informed-learning)
9. [Loss Functions](#9-loss-functions)
10. [Training Pipeline](#10-training-pipeline)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Uncertainty Estimation](#12-uncertainty-estimation)
13. [Benchmark Experiments](#13-benchmark-experiments)
14. [Hyperparameter Reference](#14-hyperparameter-reference)
15. [Current Results](#15-current-results)
16. [Current Limitations](#16-current-limitations)
17. [Future Roadmap](#17-future-roadmap)
18. [Architecture Diagram Specification](#18-architecture-diagram-specification)
19. [Presentation Assets](#19-presentation-assets)
20. [Research Contributions](#20-research-contributions)

---

# 1. Executive Summary

## 1.1 The Engineering Problem

Modern aerospace propulsion systems—specifically gas turbine engines—undergo continuous thermodynamic degradation throughout their operational lifecycle. Compressor blade fouling, combustor coking, turbine blade erosion, and thermal barrier coating spallation incrementally degrade engine health in ways that are measurable through installed sensor suites (thermocouples, pressure transducers, tachometers, flow meters) but difficult to interpret in isolation. The engineering challenge is: **given only the raw sensor telemetry from a four-stage single-spool turbojet engine, can we simultaneously estimate subsystem-level health degradation, predict net thrust, and compute fuel efficiency—all in real time, while guaranteeing that predictions never violate the laws of thermodynamics?**

## 1.2 The Scientific Problem

Pure physics-based approaches (e.g., numerical Brayton cycle solvers) are computationally prohibitive for real-time inference and require complete parametric characterization of degradation modes that may not be available. Pure data-driven approaches (unconstrained neural networks) can predict physically impossible states: negative thrust, compressor efficiencies exceeding 100%, or simultaneous health indicators that violate conservation laws. The scientific problem is the **grey-box synthesis**: constructing a machine learning system that learns from data like a neural network while obeying thermodynamic constraints like an analytical model, and quantifying its own epistemic uncertainty in a regime of severely limited training data (N=300 samples across 10 engines).

## 1.3 Why This Approach Was Chosen

The Physics-Constrained Multi-Head Network (PCMN) architecture was chosen over a canonical Physics-Informed Neural Network (PINN, Raissi et al. 2019) because the available physics at the sensor measurement resolution are *algebraic* (pressure ratios, isentropic efficiency relations, energy balances), not *differential* (PDE residuals). The PCMN embeds these algebraic constraints directly into the loss function and architectural structure—specifically, TSFC is computed deterministically rather than predicted, making thermodynamic violation mathematically impossible by construction. A GRU recurrent backbone captures temporal degradation trajectories that static point-in-time estimators miss. Combined raw + physics-engineered features resolve the information bottleneck that arises when raw sensor data is lossy-compressed into thermodynamic invariants alone.

## 1.4 Current Status

The project is operational with the following completed components:

- **Thermodynamics Engine**: Deterministic grey-box feature engineering pipeline producing 7 physics-derived features from 12 raw sensors
- **PCMN Model**: GRU-based multi-head network with 5 prediction heads, MC Dropout uncertainty, and physics-constrained loss
- **LOEO-CV Evaluation**: Statistically rigorous Leave-One-Engine-Out cross-validation with Wilcoxon significance testing
- **Phase 3 Ablation**: GRU vs MLP benchmark across sequence lengths N=3 and N=5, demonstrating statistically significant GRU superiority (p=0.0020 for N=5)
- **Transfer Learning (Phase 1)**: Contrastive pretraining on N-CMAPSS completed; linear probe ablation triggered abort criterion (no transferable representations)
- **Live Dashboard**: Socket.io + Express.js telemetry streaming with real-time inference visualization
- **Surrogate Speed Benchmark**: >2000× speedup over iterative physics computation, sub-microsecond per-sample inference

## 1.5 Future Roadmap

Three-pillar strategy:

- **Path A (Competition Dataset)**: Continue optimizing the PCMN on the 300-sample turbojet dataset with advanced regularization, real-gas thermodynamics, and calibrated uncertainty
- **Path B (N-CMAPSS)**: Formally concluded—transfer learning from turbofan to turbojet did not produce transferable representations (abort criterion triggered at p=0.8457)
- **Path C (TurboJetSim)**: Build a native physics-consistent synthetic data generator to create thousands of realistic turbojet degradation trajectories, removing the N=300 bottleneck entirely

---

# 2. Complete System Overview

## 2.1 End-to-End Pipeline

The complete data flow from raw sensor telemetry to deployed prediction is illustrated below:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RAW ENGINE TELEMETRY                                │
│  {Tamb, Pamb, T2, P2, T3, P3, T4, P4, RPM, Fuel_Flow, Altitude, Mach} │
│                        12 sensor channels                               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                                   │
│           ThermodynamicsEngine (deterministic, grey-box)                 │
│                                                                         │
│  Computes: PR_comp, PR_turb, η_c, η_t, W_net,                         │
│            ṁ_air (estimated), η_thermal (estimated)                     │
│                                                                         │
│  Output: 7 physics features (after collinearity pruning)               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                        │
│                                                                         │
│  1. Feature Concatenation: [raw_12 ‖ phys_7] → x_combined ∈ ℝ¹⁹       │
│  2. Z-Score Normalization: StandardScaler fit on TRAIN split ONLY       │
│  3. Target Normalization: Separate StandardScaler for 6 targets         │
│  4. Fuel Flow extraction for deterministic TSFC computation             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE GENERATION                                   │
│                                                                         │
│  Group by EngineID (strict engine boundary respect)                     │
│  Sliding window of length N (default N=5, stride=1)                     │
│  Zero-padding + binary mask for engines with < N cycles                 │
│  Target label: ground truth at final timestep t+N−1                     │
│                                                                         │
│  Output: X_seq ∈ ℝ^(B×5×19), y_seq ∈ ℝ^(B×6), masks ∈ ℝ^(B×5)       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL                                           │
│              DigitalTwinModel (model_type='gru')                        │
│                                                                         │
│  GRU(input=19, hidden=32, layers=1) → h_N ∈ ℝ³²                       │
│  MC Dropout(p=0.1) → Dense(32→32) + ReLU → MC Dropout(p=0.1)          │
│                                                                         │
│  5 Independent Linear Heads:                                            │
│    comp_head(32→1), comb_head(32→1), turb_head(32→1),                  │
│    overall_head(32→1), thrust_head(32→1)                               │
│                                                                         │
│  Output: 5 Z-scored predictions                                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PREDICTION                                        │
│                                                                         │
│  Denormalize: ŷ_real = ŷ_scaled × σ_train + μ_train                   │
│  Deterministic TSFC: TSFC = FuelFlow_g_s / Thrust_N                    │
│  MC Dropout Uncertainty: K=10..30 stochastic forward passes             │
│    → predictive mean μ(x) and standard deviation σ(x)                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                        │
│                                                                         │
│  LOEO-CV: 10-fold Leave-One-Engine-Out Cross-Validation                │
│  Metrics: RMSE, MAE, R², TSFC RMSE                                     │
│  Significance: Wilcoxon Signed-Rank Test (paired by engine fold)       │
│  Effect Size: Matched-pairs rank-biserial correlation                   │
│  Uncertainty: MC Dropout calibration coverage at 1σ                     │
│  Speed: Surrogate speedup factor, per-sample latency (μs)              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT                                        │
│                                                                         │
│  telemetry_streamer.py: Python → HTTP POST → Express.js server          │
│  server.js: Express.js + Socket.io WebSocket relay                      │
│  Dashboard: HTML/CSS/JS with Chart.js real-time visualization          │
│  Model Card: N=300, limitations, calibration displayed on-dashboard     │
│  Saved artifacts: .pth model, .joblib scalers                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Stage-by-Stage Explanation

### Stage 1: Raw Engine Telemetry
The system ingests 12 raw sensor channels from a four-stage single-spool turbojet engine simulation. Each row represents one operational cycle of one engine. The sensor suite follows standard aerospace station-numbering: Station 0/1 (Ambient/Inlet: Tamb, Pamb), Station 2 (Compressor Exit: T2, P2), Station 3 (Combustor Exit: T3, P3), Station 4 (Turbine Exit: T4, P4), plus cross-cutting measurements (RPM, Fuel_Flow, Altitude_m, Mach). The dataset comprises 10 distinct engines × 30 cycles = 300 total annotated samples.

### Stage 2: Feature Engineering
The `ThermodynamicsEngine` class transforms raw sensor columns into thermodynamically meaningful derived features using closed-form algebraic relationships from gas turbine theory. These features (pressure ratios, isentropic efficiencies, specific work, estimated air mass flow, estimated thermal efficiency) encode physical invariants that capture degradation signatures far more directly than raw sensor values. A collinearity analysis was performed to prune redundant features (e.g., `Combustion_Temp_Rise` is perfectly collinear with `Combustor_Heat_Addition`).

### Stage 3: Preprocessing
Raw and physics features are concatenated into a 19-dimensional combined feature vector. Z-score normalization is applied using scalers fitted exclusively on the training partition of each fold, preventing any test-time information leakage. Target variables (health indices, thrust, TSFC) are independently normalized using a separate target scaler.

### Stage 4: Sequence Generation
For temporal models (GRU), the `extract_sequences()` function constructs sliding windows of length N (default 5) grouped strictly by EngineID. Engines with fewer than N cycles receive zero-padded leading vectors and a binary mask tensor. The target for each window is the ground truth at the final timestep.

### Stage 5: Model
The `DigitalTwinModel` is a GRU-based recurrent network with a shared 32-dimensional hidden state and 5 independent linear output heads. MC Dropout layers remain active during inference for uncertainty quantification.

### Stage 6: Prediction
Z-scored network outputs are denormalized to real-world units using stored scaler parameters. TSFC is computed deterministically as the ratio of fuel flow (g/s) to predicted thrust (N), making thermodynamic violation structurally impossible. For uncertainty estimation, K stochastic forward passes produce a predictive mean and standard deviation.

### Stage 7: Evaluation
All evaluations use Leave-One-Engine-Out Cross-Validation (LOEO-CV) to eliminate single-split statistical noise. The Wilcoxon Signed-Rank Test provides non-parametric significance testing. Multiple metrics (RMSE, TSFC RMSE, coverage, speed) are reported simultaneously.

### Stage 8: Deployment
A Python telemetry streamer iterates through dataset rows (simulating a live sensor feed), performs real-time inference, and POSTs JSON payloads to an Express.js server. The server relays these via Socket.io WebSockets to a browser-based dashboard with Chart.js visualizations.

---

# 3. Repository Architecture

## 3.1 Top-Level Directory Structure

```
zero and already behind/
├── src/                          # Core Python source code
│   ├── data_pipeline/            # Data loading, preprocessing, physics features
│   ├── models/                   # Neural network architectures and loss functions
│   ├── training/                 # Training scripts for all model variants
│   └── evaluation/               # Benchmarking, metrics, and live inference
├── Dataset/                      # Raw and processed datasets
│   ├── train.csv                 # Training sensor data (240 rows, Engines 1-8)
│   ├── test.csv                  # Test sensor data (60 rows, Engines 9-10)
│   ├── ground_truth.csv          # Target labels for all 300 cycles
│   ├── turbojet_complete_dataset.csv  # Monolithic merged dataset (300 rows)
│   ├── N-CMAPSS_DS03-012.h5     # NASA N-CMAPSS turbofan dataset (3.7 GB)
│   ├── c-mapss-2-data-loading-and-exploration.ipynb  # N-CMAPSS exploration
│   └── processed/                # Preprocessed data artifacts
│       └── ncmapss_pairs_128.npz # Contrastive learning positive pairs (307 MB)
├── models/                       # Saved model weights and scalers
│   └── ncmapss_pretrained_encoder.pth  # Pretrained shared encoder (28 KB)
├── dist/                         # Distribution artifacts (trained models)
├── docs/                         # Project documentation
│   ├── system_architecture.md    # Detailed system architecture document
│   ├── context.txt               # Agent changelog and context log
│   ├── judge.md                  # Scientific adjudication report
│   └── references/               # Academic reference PDFs
├── public/                       # Dashboard web assets (served by Express)
│   ├── index.html                # Dashboard HTML entry point
│   ├── css/                      # Dashboard stylesheets
│   ├── js/                       # Dashboard JavaScript
│   └── data/                     # JSON data for dashboard charts
├── js/                           # Root-level JavaScript
│   └── app.js                    # Main dashboard application logic
├── css/                          # Root-level CSS
│   └── style.css                 # Main dashboard stylesheet
├── scripts/                      # Utility and automation scripts
│   ├── download_papers.py        # Academic paper downloader
│   ├── integrate_notebook.py     # Jupyter notebook integration
│   ├── read_pdfs.py              # PDF reader utility
│   └── remove_tf_imports.py      # TensorFlow import cleaner
├── orchestrator.py               # Master pipeline orchestrator
├── server.js                     # Express.js + Socket.io backend server
├── index.html                    # Root dashboard page (legacy)
├── package.json                  # Node.js dependencies and scripts
├── test.py                       # Quick dataset inspection script
├── test1.py                      # Additional test script
├── roast.md                      # Independent code audit (post-mortem)
├── Zero_and_Already_Behind_Scientific_Proposal 1.md  # Corrected methodology
├── DASHBOARD_BUILD_GUIDE.md      # Dashboard construction guide
└── .gitignore                    # Git ignore rules
```

## 3.2 Detailed File Documentation

### 3.2.1 `src/data_pipeline/dataset.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Central data loading, merging, normalization, sequence generation, and DataLoader creation |
| **Responsibilities** | (1) Load and merge train.csv, test.csv, ground_truth.csv via inner join on {EngineID, Cycle}. (2) Rename columns to match ThermodynamicsEngine API. (3) Compute combined feature sets (raw + physics). (4) Fit StandardScaler on training data only. (5) Generate sliding-window sequences with zero-padding for cold-start engines. (6) Create PyTorch DataLoaders with batched tensors. |
| **Dependencies** | `pandas`, `numpy`, `torch`, `sklearn.preprocessing.StandardScaler`, `sklearn.model_selection.GroupKFold`, `src.data_pipeline.thermodynamics.ThermodynamicsEngine` |
| **Inputs** | CSV files in `Dataset/` directory: `train.csv`, `test.csv`, `ground_truth.csv` |
| **Outputs** | `(train_loader, val_loader, scaler, target_scaler, feature_columns)` — PyTorch DataLoaders containing `(X_seq, y_scaled, fuel_flow, mask)` tensors |

**Key Functions:**
- `load_and_merge_data(dataset_dir)` → `DataFrame` — Loads, concatenates, deduplicates, and merges all CSV files
- `extract_sequences(df, features_array, targets_array, seq_length)` → `(X_seq, y_seq, masks, fuel_flows)` — Groups by EngineID, creates sliding windows or zero-padded sequences
- `prepare_dataloaders(df, train_idx, val_idx, batch_size, use_physics_features, seq_length)` → `(train_loader, val_loader, scaler, target_scaler, feature_cols)` — End-to-end data preparation
- `get_engine_split(df, test_engines)` → `(df_train_val, df_test)` — Engine-grouped train/test split

### 3.2.2 `src/data_pipeline/thermodynamics.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Deterministic grey-box physics feature extraction using Brayton cycle thermodynamics |
| **Responsibilities** | (1) Compute compressor and turbine pressure ratios. (2) Compute isentropic efficiencies for both compressor and turbine. (3) Compute specific work terms. (4) Estimate air mass flow rate via combustor energy balance. (5) Estimate thermal efficiency. (6) Perform collinearity pruning. (7) Provide real-gas variant with temperature-dependent cp(T) and γ(T). |
| **Dependencies** | `pandas`, `numpy` |
| **Inputs** | DataFrame with columns: `{Tamb, Pamb, T2, P2, T3, P3, T4, P4, RPM, Fuel_Flow, Altitude_m, Mach}` |
| **Outputs** | DataFrame with physics features: `{PR_comp, PR_turb, Comp_Isentropic_Efficiency, Turb_Isentropic_Efficiency, Net_Specific_Work, Estimated_Air_Mass_Flow, Estimated_Thermal_Efficiency, Altitude_m, Mach}` |

**Key Classes/Methods:**
- `ThermodynamicsEngine.__init__()` — Sets physical constants: γ=1.4, R=287.05 J/(kg·K), LHV=42.8 MJ/kg
- `compute_compressor_pressure_ratio(p_out, p_in)` — PR_comp = P2/Pamb
- `compute_turbine_pressure_ratio(p_out, p_in)` — PR_turb = P4/P3
- `compute_temperature_ratio(t_out, t_in)` — T_out/T_in
- `compute_isentropic_efficiency(t_in, t_out, p_in, p_out)` — Full isentropic efficiency formula with ε=1e-6 protection
- `extract_physics_features(df)` → DataFrame — Complete feature extraction pipeline (constant cp=1005)
- `extract_real_gas_features(df)` → DataFrame — Temperature-dependent cp(T) and γ(T) variant

### 3.2.3 `src/data_pipeline/ncmapss_extractor.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Extract positive pairs from NASA N-CMAPSS HDF5 dataset for contrastive self-supervised pretraining |
| **Responsibilities** | (1) Load N-CMAPSS development arrays (A_dev, X_s_dev). (2) Group data by unit and cycle (flight). (3) Extract non-overlapping blocks of 2×seq_len from each flight. (4) Split each block into two adjacent windows forming a positive pair. (5) Save as compressed .npz file. |
| **Dependencies** | `h5py`, `numpy`, `tqdm` |
| **Inputs** | `Dataset/N-CMAPSS_DS03-012.h5` (3.7 GB HDF5 file) |
| **Outputs** | `Dataset/processed/ncmapss_pairs_128.npz` containing arrays `X1` and `X2`, each of shape `(N_pairs, 128, 14)` |

### 3.2.4 `src/models/pinn.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Core neural network architecture for the Digital Twin model |
| **Responsibilities** | (1) Implement shared backbone (MLP or GRU). (2) Implement 5 independent prediction heads. (3) Implement MC Dropout uncertainty estimation. |
| **Dependencies** | `torch`, `torch.nn`, `torch.nn.functional` |
| **Inputs** | Tensor of shape `(batch, seq_len, n_features)` for GRU or `(batch, n_features)` for MLP |
| **Outputs** | Tuple of 5 tensors, each `(batch, 1)`: `(comp_h, comb_h, turb_h, overall_h, thrust)` |

### 3.2.5 `src/models/loss.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Physics-constrained composite loss function |
| **Responsibilities** | (1) Compute multi-head MSE across all 5 prediction targets. (2) Enforce health consistency constraint (overall health as weighted sum of subsystem healths). (3) Perform internal denormalization for physics constraints in real-world units. (4) Return decomposed loss components for monitoring. |
| **Dependencies** | `torch`, `torch.nn` |
| **Inputs** | `(preds, targets, fuel_flow_g, target_mean, target_scale)` |
| **Outputs** | `(total_loss, mse_total, tsfc_consistency=0.0, health_consistency)` |

### 3.2.6 `src/models/transfer.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Transfer learning architecture with domain-specific adapters and a shared encoder |
| **Responsibilities** | (1) Define DomainAdapter base class (Linear→ReLU→LayerNorm→Linear projection). (2) Define NCMAPSSAdapter (14→64→32). (3) Define TurbojetAdapter (19→64→32). (4) Define SharedEncoder (GRU operating in 32-dim latent space). (5) Define ContrastivePretrainingModel (N-CMAPSS adapter + shared encoder). (6) Define TransferredDigitalTwinModel (Turbojet adapter + pretrained/random encoder + 5 output heads). |
| **Dependencies** | `torch`, `torch.nn`, `torch.nn.functional` |
| **Inputs** | N-CMAPSS: `(batch, seq_len, 14)` / Turbojet: `(batch, seq_len, 19)` |
| **Outputs** | Pretraining: `h ∈ ℝ³²` latent vector. Downstream: 5-tuple of `(batch, 1)` predictions |

### 3.2.7 `src/training/train.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Main training script for the three canonical model variants |
| **Responsibilities** | (1) Load and split data (Train: Engines 1-6, Val: 7-8, Test: 9-10). (2) Train Baseline-Raw (MLP, N=1, no physics features, no physics loss). (3) Train PhysFeat-Combined (MLP, N=1, combined features, no physics loss). (4) Train Full Model (GRU, N=5, combined features, physics-constrained loss). (5) Implement early stopping with patience=20. (6) Save best model weights and scalers to disk. |
| **Dependencies** | `torch`, `torch.optim`, `pandas`, `numpy`, `joblib`, all `src.*` modules |
| **Inputs** | CSV dataset files |
| **Outputs** | `dist/models/{variant_name}.pth`, `dist/models/{variant_name}_scaler.joblib`, `dist/models/{variant_name}_target_scaler.joblib` |

### 3.2.8 `src/training/train_pinn.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Earlier PINN training script using `turbojet_complete_dataset.csv` directly (legacy, references removed classes) |
| **Responsibilities** | Load monolithic dataset, extract physics features, train a 6-headed PINN with physics-informed loss |
| **Dependencies** | References `PINNDigitalTwin` and `PhysicsInformedLoss` which are no longer in the current codebase |
| **Inputs** | `Dataset/turbojet_complete_dataset.csv` |
| **Outputs** | `dist/models/pinn_model.pth` |
| **Status** | **Legacy — will crash on execution** due to undefined imports (`PINNDigitalTwin`, `PhysicsInformedLoss`). Superseded by `train.py`. |

### 3.2.9 `src/training/pretrain.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Contrastive self-supervised pretraining of the SharedEncoder on N-CMAPSS data |
| **Responsibilities** | (1) Load preprocessed positive pairs from .npz. (2) Apply global Z-score normalization. (3) Implement InfoNCE contrastive loss with temperature τ=0.1. (4) Train ContrastivePretrainingModel for 30 epochs. (5) Save only the SharedEncoder weights (not the N-CMAPSS adapter). |
| **Dependencies** | `torch`, `torch.nn.functional`, `numpy`, `tqdm`, `src.models.transfer.ContrastivePretrainingModel` |
| **Inputs** | `Dataset/processed/ncmapss_pairs_128.npz` |
| **Outputs** | `models/ncmapss_pretrained_encoder.pth` (SharedEncoder state_dict, 28 KB) |

### 3.2.10 `src/training/linear_probe.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Linear probe ablation to determine if pretrained encoder learned transferable representations |
| **Responsibilities** | (1) Compare frozen pretrained encoder vs. frozen random encoder. (2) Train only the adapter and output heads. (3) Run full LOEO-CV across all engines. (4) Compute Wilcoxon significance test. (5) Evaluate abort criterion: if pretrained is no better than random, abort transfer learning. |
| **Dependencies** | `torch`, `torch.optim`, `numpy`, `pandas`, `scipy.stats.wilcoxon`, all `src.*` modules |
| **Inputs** | `models/ncmapss_pretrained_encoder.pth`, CSV dataset files |
| **Outputs** | Console report: per-variant RMSE ± std, Wilcoxon p-value, SUCCESS/FAILURE verdict |

### 3.2.11 `src/evaluation/benchmark.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Comprehensive LOEO-CV benchmark comparing GRU vs MLP across sequence lengths, plus surrogate speed benchmark |
| **Responsibilities** | (1) Run LOEO-CV for 4 model variants: MLP(N=3), GRU(N=3), MLP(N=5), GRU(N=5). (2) Record per-fold RMSE and TSFC RMSE. (3) Compute Wilcoxon significance for MLP vs GRU at each N. (4) Benchmark surrogate inference speed vs iterative physics computation. (5) Save results as JSON for dashboard. |
| **Dependencies** | `torch`, `torch.optim`, `numpy`, `pandas`, `scipy.stats.wilcoxon`, `time`, `json`, `joblib`, all `src.*` modules |
| **Inputs** | CSV dataset files |
| **Outputs** | `public/data/benchmark_results.json`, console report |

### 3.2.12 `src/evaluation/loeo_benchmark.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Earlier LOEO-CV benchmark comparing Baseline-Raw vs Baseline-PhysFeat (Phase 2 information bottleneck experiment) |
| **Responsibilities** | (1) Run LOEO-CV for Raw vs PhysFeat models. (2) Compute Wilcoxon test and effect size. (3) Report per-fold results. |
| **Dependencies** | `torch`, `pandas`, `numpy`, `scipy.stats.wilcoxon`, `sklearn.preprocessing.StandardScaler`, all `src.*` modules |
| **Inputs** | `Dataset/turbojet_complete_dataset.csv` |
| **Outputs** | Console report with RMSE, Wilcoxon p-value, rank-biserial effect size |

### 3.2.13 `src/evaluation/metrics.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Reusable evaluation metric functions |
| **Responsibilities** | Compute TSFC violation percentage for both PyTorch tensors and NumPy arrays |
| **Dependencies** | `torch`, `numpy` |
| **Inputs** | `(predicted_tsfc, theoretical_tsfc)` — scalar, tensor, or array |
| **Outputs** | Float — mean absolute percentage violation |

### 3.2.14 `src/evaluation/telemetry_streamer.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Live inference streamer that simulates real-time telemetry by iterating through the dataset and pushing predictions to the dashboard server |
| **Responsibilities** | (1) Load all 3 trained model variants and their scalers. (2) Iterate through dataset rows simulating live feed. (3) Maintain a sliding buffer of N=5 steps for the GRU model. (4) Perform real-time inference with MC Dropout uncertainty (K=10). (5) Compute deterministic TSFC. (6) POST JSON payloads to Express.js server via HTTP. (7) Handle cold-start (insufficient history) gracefully. |
| **Dependencies** | `torch`, `pandas`, `numpy`, `joblib`, `requests`, `time`, all `src.*` modules |
| **Inputs** | `dist/models/*.pth`, `dist/models/*_scaler.joblib`, `Dataset/turbojet_complete_dataset.csv` |
| **Outputs** | HTTP POST to `http://localhost:3000/api/telemetry` |

### 3.2.15 `orchestrator.py`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Master pipeline orchestrator that runs the complete ML pipeline in sequence |
| **Responsibilities** | Execute `train_pinn.py` via subprocess (legacy reference) |
| **Dependencies** | `subprocess`, `sys`, `time` |
| **Status** | **Partially outdated** — references `train_pinn.py` which uses removed classes |

### 3.2.16 `server.js`

| Property | Detail |
|:---------|:-------|
| **Purpose** | Express.js + Socket.io backend server for real-time dashboard |
| **Responsibilities** | (1) Serve static files from `public/`. (2) Accept telemetry POST at `/api/telemetry`. (3) Broadcast telemetry to connected WebSocket clients. (4) Detect offline state via 3-second watchdog timer. (5) Synchronize task checkbox state across multiple dashboard instances. |
| **Dependencies** | `express@5.2.1`, `socket.io@4.8.3` |
| **Inputs** | HTTP POST JSON from telemetry_streamer.py |
| **Outputs** | WebSocket emissions to connected dashboard clients |

---

# 4. Dataset Documentation

## 4.1 Competition Dataset (Primary)

### 4.1.1 Source
Synthetic turbojet engine telemetry generated from a physics-based four-stage single-spool turbojet engine model, provided as part of the IIT Indore × HAL aerospace hackathon (Statement #2). The data has been independently verified to be physically consistent through hand-computation of isentropic efficiencies (see Section 8.1).

### 4.1.2 Purpose
Primary training and evaluation dataset for the Digital Twin. Models all six competition deliverables: compressor health, combustor health, turbine health, overall engine health, thrust, and TSFC.

### 4.1.3 Size
- **Total samples**: 300 annotated cycles
- **Engines**: 10 distinct engines
- **Cycles per engine**: 30
- **Sensor channels**: 12 raw measurements
- **Target variables**: 6 labeled outputs

### 4.1.4 Files and Schema

#### `train.csv` (240 rows)

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `EngineID` | int | — | Engine identifier (Engines 1-8) |
| `Cycle` | int | — | Operational cycle number (1-30) |
| `Tamb_K` | float | K | Ambient temperature at intake |
| `Pamb_Pa` | float | Pa | Ambient static pressure |
| `T2_K` | float | K | Compressor exit temperature |
| `P2_Pa` | float | Pa | Compressor exit pressure |
| `T3_K` | float | K | Combustor exit temperature |
| `P3_Pa` | float | Pa | Combustor exit pressure |
| `T4_K` | float | K | Turbine exit temperature |
| `P4_Pa` | float | Pa | Turbine exit pressure |
| `RPM_rev_min` | float | rev/min | Shaft rotational speed |
| `FuelFlow_kg_s` | float | kg/s | Fuel mass flow rate |
| `Altitude_m` | float | m | Operational altitude |
| `Mach` | float | — | Flight Mach number |

#### `test.csv` (60 rows)
Same schema as `train.csv`. Contains Engines 9-10 (organizer-provided held-out engines).

#### `ground_truth.csv` (300 rows)

| Column | Type | Unit | Range | Description |
|:-------|:-----|:-----|:------|:------------|
| `EngineID` | int | — | 1-10 | Engine identifier |
| `Cycle` | int | — | 1-30 | Cycle number |
| `CompressorHealth` | float | — | [0, 1] | Compressor health index |
| `CombustorHealth` | float | — | [0, 1] | Combustor health index |
| `TurbineHealth` | float | — | [0, 1] | Turbine health index |
| `OverallHealth` | float | — | [0, 1] | Engine-level health index |
| `Thrust_N` | float | N | ~10,000–90,000 | Net thrust produced |
| `TSFC_g_N_s` | float | g/(N·s) | ~0.010–0.030 | Thrust-Specific Fuel Consumption |

#### `turbojet_complete_dataset.csv` (300 rows)
Pre-merged monolithic file containing all sensor columns and all target columns. Used by legacy scripts (`train_pinn.py`, `loeo_benchmark.py`). Structurally identical to the inner join of `train.csv + test.csv + ground_truth.csv`.

### 4.1.5 Train/Validation/Test Split

| Partition | Engines | Cycles | Purpose |
|:----------|:--------|:-------|:--------|
| **Train** | 1, 2, 3, 4, 5, 6 | 180 | Model parameter optimization |
| **Validation** | 7, 8 | 60 | Early stopping, hyperparameter selection |
| **Test** (blind) | 9, 10 | 60 | Final evaluation (touched once) |
| **LOEO-CV** | Each of 1-10 held out | 270 train / 30 test per fold | Statistically robust evaluation |

### 4.1.6 Preprocessing
Columns are renamed from verbose names (e.g., `Tamb_K` → `Tamb`) to match the `ThermodynamicsEngine` API. The complete rename mapping is documented in Section 3.2.1.

### 4.1.7 Why This Dataset Exists
The competition organizers provided this dataset to evaluate participants' ability to build a physics-informed digital twin for turbojet health monitoring. The data was generated from an actual thermodynamic engine model (verified by hand-computation of isentropic efficiencies landing in the realistic 78-90% band), not column-independent random noise. The organizer-provided train/test split by engine is a deliberate leave-engines-out generalization test.

## 4.2 NASA N-CMAPSS Dataset (Transfer Learning Source)

### 4.2.1 Source
NASA Prognostics Center of Excellence. Arias Chao, Kulkarni, Goebel & Fink (2021). "Aircraft Engine Run-to-Failure Dataset Under Real Flight Conditions for Prognostics and Diagnostics." Dataset DS03-012.

### 4.2.2 Purpose
Source domain for contrastive self-supervised pretraining of the SharedEncoder. The hypothesis was that fundamental thermodynamic relationships are shared across gas-turbine architectures (turbofan → turbojet), enabling representation transfer from data-rich N-CMAPSS to data-poor turbojet competition dataset.

### 4.2.3 Size
- **File**: `Dataset/N-CMAPSS_DS03-012.h5` — 3.7 GB HDF5
- **Processed pairs**: `Dataset/processed/ncmapss_pairs_128.npz` — 307 MB
- **Sensor channels**: 14 (X_s_dev array)
- **Auxiliary columns**: 4 (A_dev: unit, cycle, Fc, hs)

### 4.2.4 Schema (Used Columns)

| Array | Shape | Description |
|:------|:------|:------------|
| `A_dev` | (N, 4) | Auxiliary: [unit, cycle, flight_class, health_state] |
| `X_s_dev` | (N, 14) | Sensor measurements: 14 channels |

### 4.2.5 Preprocessing
The `ncmapss_extractor.py` script:
1. Loads `A_dev` and `X_s_dev` into memory
2. Groups data by unit and cycle (flight)
3. Extracts non-overlapping blocks of size 2×128=256 timesteps from each flight
4. Splits each block into two adjacent 128-timestep windows → positive pair
5. Saves as compressed NPZ with arrays X1 and X2

During pretraining, global Z-score normalization is applied:
- Mean computed over X1 along axes (0, 1) with keepdims=True
- Std computed similarly with ε=1e-6 for numerical stability
- Both X1 and X2 normalized using X1 statistics

### 4.2.6 Why This Dataset Exists
N-CMAPSS provides orders of magnitude more data than the 300-sample competition dataset. If cross-architecture representation transfer worked, the pretrained encoder would provide a better initialization than random weights, enabling better downstream performance on the turbojet task.

### 4.2.7 Current Status
**Transfer learning failed.** Linear probe ablation showed the frozen pretrained encoder (RMSE 0.0235) performed no better than a frozen random encoder (RMSE 0.0224) with Wilcoxon p=0.8457. The abort criterion was triggered. Path B is formally concluded.

## 4.3 TurboJetSim (Planned)

### 4.3.1 Source
**Planned** — A native physics-consistent synthetic data generator to be built in-house.

### 4.3.2 Purpose
Generate thousands of realistic turbojet degradation trajectories using parametric Brayton cycle models with stochastic degradation injection, removing the N=300 sample size bottleneck entirely.

### 4.3.3 Current Status
**Not yet implemented.** This is Path C of the three-pillar roadmap (Section 17).

---

# 5. Data Preprocessing Pipeline

## 5.1 Step 1: Data Loading and Merging

**Implementation**: `load_and_merge_data()` in `dataset.py`

1. Load `train.csv`, `test.csv`, `ground_truth.csv` from the `Dataset/` directory
2. Concatenate train and test DataFrames: `pd.concat([df_train, df_test], ignore_index=True)`
3. Remove duplicates: `.drop_duplicates(subset=['EngineID', 'Cycle'])`
4. Inner join with ground truth: `pd.merge(df_all_sensors, df_gt, on=['EngineID', 'Cycle'], how='inner')`
5. Rename columns to match ThermodynamicsEngine expectations (e.g., `Tamb_K` → `Tamb`)

**Rationale**: The organizers split the data across three files but provided overlapping EngineIDs. Merging reconstructs the complete 300-sample dataset with aligned sensor readings and target labels.

## 5.2 Step 2: Data Validation

**Implementation**: The dataset was validated by hand-computation (documented in the Scientific Proposal):

- **Compressor Isentropic Efficiency**: Engine 1 Cycle 1 → η_c = 78.6%, Cycle 2 → η_c = 90.0%
- **Turbine Isentropic Efficiency**: Engine 1 Cycle 1 → η_t = 83.4%
- **Conclusion**: Both land in the realistic 78-90% band for real turbomachinery, confirming the dataset is physics-based, not random noise

**Structural Discovery**: `train.csv` (240 rows) + `test.csv` (60 rows) = 300 rows = `turbojet_complete_dataset.csv`. The organizers already provided a leave-engines-out generalization split.

## 5.3 Step 3: Feature Engineering

**Implementation**: `ThermodynamicsEngine.extract_physics_features()` in `thermodynamics.py`

From the 12 raw sensor columns, 7 physics features are computed (after collinearity pruning):

1. PR_comp (compressor pressure ratio)
2. PR_turb (turbine pressure ratio)
3. Comp_Isentropic_Efficiency
4. Turb_Isentropic_Efficiency
5. Net_Specific_Work
6. Estimated_Air_Mass_Flow
7. Estimated_Thermal_Efficiency

**Dropped (collinear)**:
- `Combustion_Temp_Rise` — perfectly collinear with `Combustor_Heat_Addition`
- `Overall_Pressure_Ratio` — perfectly collinear with `PR_comp`
- `Normalized_RPM` — introduced collinearity with `PR_comp`
- `Compressor_Specific_Work`, `Turbine_Specific_Work` — intermediate quantities used to compute `Net_Specific_Work`
- `Combustor_Heat_Addition` — intermediate quantity

## 5.4 Step 4: Feature Concatenation

**Implementation**: `prepare_dataloaders()` in `dataset.py`

```
x_combined = [x_raw (12 dims) ‖ x_phys_no_atm (7 dims)] → ℝ¹⁹
```

The 7 physics features exclude `Altitude_m` and `Mach` since those are already present in the raw sensor set. This resolves the Information Bottleneck identified in Phase 2 (Section 8.4).

## 5.5 Step 5: Normalization

**Implementation**: `StandardScaler` from scikit-learn

### Feature Normalization
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)   # Fit on train ONLY
X_val_scaled = scaler.transform(X_val_flat)            # Transform-only
```

Each of the 19 features is independently Z-score normalized: x_scaled = (x − μ_train) / σ_train

### Target Normalization
```python
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train_raw)  # Fit on train ONLY
y_val_scaled = target_scaler.transform(y_val_raw)          # Transform-only
```

All 6 target variables are independently Z-score normalized. The network outputs Z-scores; denormalization occurs at inference time: ŷ_real = ŷ_scaled × σ_train + μ_train.

**Why Z-Score over Min-Max**: Z-score is preferred because (1) min-max is sensitive to outliers, (2) thrust has no theoretically bounded extremes, and (3) Z-score naturally handles out-of-distribution values as large-magnitude scores.

## 5.6 Step 6: Sequence Generation

**Implementation**: `extract_sequences()` in `dataset.py`

For each engine (grouped by EngineID, sort=False):

**Case 1: num_cycles ≥ N** (typical case, N=5):
- Sliding window of length N with stride 1
- Yields `num_cycles − N + 1` overlapping windows per engine
- Target: ground truth at final timestep t+N−1
- Mask: all ones (no padding)

**Case 2: num_cycles < N** (cold-start edge case):
- Zero-pad with `N − num_cycles` leading zero vectors
- Binary mask: `[0, 0, ..., 1, 1, ..., 1]`
- Single window per engine
- Target: ground truth at last available cycle

**Output tensors**:
- `X_seq`: shape `(num_windows, N, 19)` — float32
- `y_seq`: shape `(num_windows, 6)` — float32 (Z-scored targets)
- `masks`: shape `(num_windows, N)` — float32 binary mask
- `fuel_flows`: shape `(num_windows,)` — float32 (raw fuel flow at final timestep)

## 5.7 Step 7: Train/Test Splitting

**Implementation**: `get_engine_split()` in `dataset.py`

Engine-grouped splitting ensures no cycles from the same engine appear in both train and test, preventing temporal data leakage.

```python
def get_engine_split(df, test_engines=[9, 10]):
    test_mask = df['EngineID'].isin(test_engines)
    df_test = df[test_mask].reset_index(drop=True)
    df_train_val = df[~test_mask].reset_index(drop=True)
    return df_train_val, df_test
```

## 5.8 Step 8: DataLoader Creation

**Implementation**: `prepare_dataloaders()` in `dataset.py`

```python
train_dataset = TensorDataset(X_train_t, y_train_t, ff_train_t, masks_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t, ff_val_t, masks_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

Each batch contains 4 tensors: `(X_seq, y_scaled, fuel_flow, mask)`.

## 5.9 Step 9: Scaler Persistence

**Implementation**: `train.py`

```python
joblib.dump(scaler, f'dist/models/{model_name}_scaler.joblib')
joblib.dump(target_scaler, f'dist/models/{model_name}_target_scaler.joblib')
```

Both feature and target scalers are saved alongside model weights for reproducible inference. The telemetry streamer loads these at startup to ensure consistent normalization.

---

# 6. Feature Engineering

## 6.1 Complete Feature Catalog

### Raw Sensor Features (12 dimensions)

| # | Feature | Formula | Physical Meaning | Unit | Computed In |
|:--|:--------|:--------|:-----------------|:-----|:------------|
| 1 | `Tamb` | Direct measurement | Ambient temperature at engine intake (Station 0/1) | K | Raw data |
| 2 | `Pamb` | Direct measurement | Ambient static pressure | Pa | Raw data |
| 3 | `T2` | Direct measurement | Compressor exit stagnation temperature (Station 2) | K | Raw data |
| 4 | `P2` | Direct measurement | Compressor exit stagnation pressure (Station 2) | Pa | Raw data |
| 5 | `T3` | Direct measurement | Combustor exit / turbine inlet temperature (Station 3) | K | Raw data |
| 6 | `P3` | Direct measurement | Combustor exit / turbine inlet pressure (Station 3) | Pa | Raw data |
| 7 | `T4` | Direct measurement | Turbine exit temperature (Station 4) | K | Raw data |
| 8 | `P4` | Direct measurement | Turbine exit pressure (Station 4) | Pa | Raw data |
| 9 | `RPM` | Direct measurement | Shaft rotational speed | rev/min | Raw data |
| 10 | `Fuel_Flow` | Direct measurement | Fuel mass flow rate (ṁ_fuel) | kg/s | Raw data |
| 11 | `Altitude_m` | Direct measurement | Operational altitude | m | Raw data |
| 12 | `Mach` | Direct measurement | Flight Mach number | dimensionless | Raw data |

### Physics-Derived Features (7 dimensions, after pruning)

| # | Feature | Formula | Physical Meaning | Why Introduced | Computed In |
|:--|:--------|:--------|:-----------------|:---------------|:------------|
| 13 | `PR_comp` | P₂ / P_amb | Compressor pressure ratio — how hard the compressor is working. Degradation in compressor blades manifests directly in this ratio's relationship to RPM | Dimensionless indicator of compressor operating point; fouled compressor produces less pressure per unit rotational work | `thermodynamics.py:17-19` |
| 14 | `PR_turb` | P₄ / P₃ | Turbine pressure ratio — characterizes the expansion process. Always < 1.0 since pressure drops across the turbine | Turbine blade erosion causes less efficient expansion, detectable as anomalous relationship between this ratio and temperature drop | `thermodynamics.py:21-23` |
| 15 | `Comp_Isentropic_Efficiency` | ((P₂/P_amb)^((γ-1)/γ) − 1) / (T₂/T_amb − 1), γ=1.4 | Ratio of ideal isentropic work to actual work in compressor. Perfect compressor → 1.0. Degradation reduces this irreversibly | Single most informative feature for CompressorHealth; captures fundamental thermodynamic cost of compression losses | `thermodynamics.py:28-42` |
| 16 | `Turb_Isentropic_Efficiency` | (1 − T₄/T₃) / (1 − (P₄/P₃)^0.2857 + ε) | Turbine-side analogue of η_c; measures how effectively the turbine converts enthalpy drop into shaft work | Direct turbine-erosion indicator; verified physically plausible (83.4% from hand calculation) | `thermodynamics.py:66-70` |
| 17 | `Net_Specific_Work` | c_p×(T₃−T₄) − c_p×(T₂−T_amb), c_p=1005 J/(kg·K) | Net thermodynamic work per unit mass of working fluid. Drives thrust via momentum theorem | Fundamental driver of thrust production; captures combined compressor+turbine degradation | `thermodynamics.py:79-80` |
| 18 | `Estimated_Air_Mass_Flow` | (ṁ_fuel × LHV) / (c_p × (T₃−T₂) + ε), LHV=42.8×10⁶ J/kg | Air mass flow rate estimated from combustor energy balance. Not directly measured by any sensor | Critical derived quantity enabling thrust estimation without intrusive flow measurement equipment | `thermodynamics.py:90-95` |
| 19 | `Estimated_Thermal_Efficiency` | (ṁ_air × W_net) / (ṁ_fuel × LHV + ε) | Complete Brayton cycle thermal efficiency — fraction of fuel chemical energy converted into net work | Ultimate figure of merit for engine health; degraded engine wastes more fuel energy as heat | `thermodynamics.py:97-101` |

### Dropped Features (Collinearity Pruning)

| Feature | Formula | Reason for Removal |
|:--------|:--------|:-------------------|
| `Combustion_Temp_Rise` | T₃ − T₂ | Perfectly collinear with `Combustor_Heat_Addition` (same info, different scale) |
| `Overall_Pressure_Ratio` | P₃ / P_amb | Perfectly collinear with `PR_comp` given cascade structure |
| `Normalized_RPM` | RPM / √(T_amb) | Introduced collinearity with `PR_comp` in this dataset (both track operating point) |
| `Compressor_Specific_Work` | c_p × (T₂ − T_amb) | Intermediate quantity; retained only as component of `Net_Specific_Work` |
| `Turbine_Specific_Work` | c_p × (T₃ − T₄) | Intermediate quantity; retained only as component of `Net_Specific_Work` |
| `Combustor_Heat_Addition` | c_p × (T₃ − T₂) | Intermediate quantity used in `Estimated_Air_Mass_Flow`; collinear with `Combustion_Temp_Rise` |

## 6.2 Real-Gas Feature Variant

**Implementation**: `ThermodynamicsEngine.extract_real_gas_features()` in `thermodynamics.py`

A temperature-dependent variant was developed using:
- `cp(T) = 1005.0 + 0.0722 × (T − 300.0)` J/(kg·K) — linear fit for air
- `γ(T) = cp(T) / (cp(T) − R)` — derived from cp(T) and R=287.05

Average temperature at each component (e.g., `t_avg_c = (T_amb + T₂)/2` for the compressor) is used to evaluate cp and γ, producing more accurate efficiency estimates at high temperatures (T₃ > 3000K where constant cp=1005 diverges significantly).

**Status**: Developed and available but not used in the default pipeline. The constant-cp variant is the default in `extract_physics_features()`.

---

# 7. Complete Model Architecture

## 7.1 DigitalTwinModel (Primary Architecture)

**Implementation**: `src/models/pinn.py`, class `DigitalTwinModel`

### 7.1.1 Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `input_dim` | int | — | Number of input features per timestep (19 for combined, 12 for raw) |
| `hidden_dim` | int | 32 | Dimensionality of shared latent space |
| `dropout_rate` | float | 0.1 | MC Dropout probability |
| `model_type` | str | `'mlp'` | Architecture variant: `'mlp'` or `'gru'` |

### 7.1.2 MLP Variant Architecture

```
Input: x ∈ ℝ^(batch, seq_len×features) — flattened from 3D
                     │
                     ▼
        ┌─────────────────────────┐
        │ shared_fc1: Linear      │
        │   (seq_len×features, 32)│    e.g., Linear(95, 32) for N=5, 19 features
        │ + ReLU activation       │
        │ + MC Dropout(p=0.1)     │
        └─────────────┬───────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │ shared_fc2: Linear(32,32)│
        │ + ReLU activation       │
        │ + MC Dropout(p=0.1)     │
        └─────────────┬───────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │ shared_fc3: Linear(32,32)│
        │ + ReLU activation       │
        │ + MC Dropout(p=0.1)     │
        └─────────────┬───────────┘
                      │
        ┌─────┬──────┼──────┬──────┐
        ▼     ▼      ▼      ▼      ▼
    [comp]  [comb]  [turb]  [ovr]  [thrust]
     32→1   32→1    32→1    32→1    32→1
```

**Tensor Dimensions (MLP, N=5, 19 features)**:
- Input: `(B, 5, 19)` → reshape to `(B, 95)`
- After shared_fc1: `(B, 32)`
- After shared_fc2: `(B, 32)`
- After shared_fc3: `(B, 32)`
- Each head output: `(B, 1)`

### 7.1.3 GRU Variant Architecture (Primary)

```
Input: x ∈ ℝ^(batch, 5, 19) — 3D temporal sequence
                     │
                     ▼
        ┌─────────────────────────────┐
        │ GRU Layer                   │
        │   input_size=19             │
        │   hidden_size=32            │
        │   num_layers=1              │
        │   batch_first=True          │
        │                             │
        │   h_t = GRU(x_t, h_{t-1})   │
        │   Process N=5 timesteps     │
        │                             │
        │   gru_out: (B, 5, 32)       │
        │   h_n: (1, B, 32)           │
        └──────────────┬──────────────┘
                       │
                       ▼ h_n[-1] → (B, 32)
        ┌─────────────────────────┐
        │ MC Dropout(p=0.1)       │
        └─────────────┬───────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │ post_gru_fc: Linear(32,32)│
        │ + ReLU activation       │
        └─────────────┬───────────┘
                      │
        ┌─────────────────────────┐
        │ MC Dropout(p=0.1)       │
        └─────────────┬───────────┘
                      │
        ┌─────┬──────┼──────┬──────┐
        ▼     ▼      ▼      ▼      ▼
    [comp]  [comb]  [turb]  [ovr]  [thrust]
     32→1   32→1    32→1    32→1    32→1
```

**Tensor Dimensions (GRU, N=5, 19 features)**:
- Input: `(B, 5, 19)`
- GRU output: `(B, 5, 32)` — all hidden states
- GRU final hidden: `(1, B, 32)` → extract `[-1]` → `(B, 32)`
- After MC Dropout: `(B, 32)`
- After post_gru_fc + ReLU: `(B, 32)`
- After MC Dropout: `(B, 32)`
- Each head output: `(B, 1)`

### 7.1.4 GRU Internal Equations

The GRU at each timestep t computes:

- **Update gate**: z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)
- **Reset gate**: r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)
- **Candidate state**: h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}) + b_h)
- **New hidden state**: h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Where:
- W_z, W_r, W_h ∈ ℝ^(32×19) — input weight matrices
- U_z, U_r, U_h ∈ ℝ^(32×32) — recurrent weight matrices
- b_z, b_r, b_h ∈ ℝ^32 — bias vectors
- σ — sigmoid activation
- ⊙ — element-wise (Hadamard) product

### 7.1.5 Output Heads

Five independent linear layers, all mapping from the shared 32-dim representation:

| Head | Layer | Input → Output | Activation | Physical Output |
|:-----|:------|:---------------|:-----------|:----------------|
| `compressor_head` | `nn.Linear(32, 1)` | `(B, 32) → (B, 1)` | None (Z-score) | CompressorHealth |
| `combustor_head` | `nn.Linear(32, 1)` | `(B, 32) → (B, 1)` | None (Z-score) | CombustorHealth |
| `turbine_head` | `nn.Linear(32, 1)` | `(B, 32) → (B, 1)` | None (Z-score) | TurbineHealth |
| `overall_health_head` | `nn.Linear(32, 1)` | `(B, 32) → (B, 1)` | None (Z-score) | OverallHealth |
| `thrust_head` | `nn.Linear(32, 1)` | `(B, 32) → (B, 1)` | None (Z-score) | Thrust_N |

**Note**: TSFC is NOT a network output. It is computed deterministically post-inference.

### 7.1.6 Parameter Count (GRU variant)

| Component | Parameters |
|:----------|:-----------|
| GRU layer | 3 × (19×32 + 32×32 + 32 + 32) = 3 × (608 + 1024 + 64) = 5,088 |
| post_gru_fc | 32×32 + 32 = 1,056 |
| compressor_head | 32×1 + 1 = 33 |
| combustor_head | 33 |
| turbine_head | 33 |
| overall_health_head | 33 |
| thrust_head | 33 |
| **Total** | **~6,309** |

## 7.2 Transfer Learning Architecture

### 7.2.1 DomainAdapter

```
Input: x ∈ ℝ^(B, seq_len, input_dim)
         │
         ▼
  fc1: Linear(input_dim, 64) + ReLU
         │
         ▼
  ln: LayerNorm(64)
         │
         ▼
  fc2: Linear(64, 32)
         │
         ▼
Output: z ∈ ℝ^(B, seq_len, 32)  — Engine State Representation Space
```

**NCMAPSSAdapter**: input_dim=14  
**TurbojetAdapter**: input_dim=19

### 7.2.2 SharedEncoder

```
Input: z ∈ ℝ^(B, seq_len, 32)
         │
         ▼
  GRU(input=32, hidden=32, layers=1) → h_N ∈ ℝ^(B, 32)
         │
         ▼
  Dropout(p=0.1)
         │
         ▼
Output: h ∈ ℝ^(B, 32)
```

### 7.2.3 ContrastivePretrainingModel

```
N-CMAPSS Input: x ∈ ℝ^(B, 128, 14)
         │
         ▼
  NCMAPSSAdapter → z ∈ ℝ^(B, 128, 32)
         │
         ▼
  SharedEncoder → h ∈ ℝ^(B, 32)
```

### 7.2.4 TransferredDigitalTwinModel

```
Turbojet Input: x ∈ ℝ^(B, 5, 19)
         │
         ▼
  TurbojetAdapter → z ∈ ℝ^(B, 5, 32)
         │
         ▼
  SharedEncoder (pretrained or random) → h ∈ ℝ^(B, 32)
         │
         ▼
  post_gru_fc: Linear(32, 32) + ReLU + MC Dropout(0.1)
         │
     ┌───┼───┬───┬───┐
     ▼   ▼   ▼   ▼   ▼
  [comp][comb][turb][ovr][thrust]
   32→1 32→1 32→1 32→1  32→1
```

---

# 8. Physics-Informed Learning

## 8.1 Physical Assumptions

### 8.1.1 Calorically Perfect Gas Assumption (Default)
The default `extract_physics_features()` method assumes air behaves as a **calorically perfect gas** with:
- γ = 1.4 (constant specific heat ratio)
- c_p = 1005 J/(kg·K) (constant specific heat at constant pressure)
- R = 287.05 J/(kg·K) (specific gas constant for air)

**Known Limitation**: T₃ values in the dataset exceed 3000K, where real cp for air diverges significantly from 1005. The real-gas variant (`extract_real_gas_features()`) addresses this with temperature-dependent cp(T) = 1005.0 + 0.0722×(T−300) J/(kg·K), but is not used in the default pipeline.

### 8.1.2 Jet-A Fuel Lower Heating Value
LHV = 42.8 × 10⁶ J/kg. This is the standard accepted value for Jet-A aviation fuel, used in the combustor energy balance to estimate air mass flow rate.

### 8.1.3 Complete Combustion Assumption
The combustor energy balance assumes complete combustion: all fuel chemical energy is transferred to the working fluid. In reality, combustion efficiency is typically 98-99.5%, introducing a small systematic error in estimated air mass flow rate.

### 8.1.4 Single-Spool Simplification
No distinct Station 1 (compressor inlet) measurement is separate from ambient conditions. This is consistent with a single-spool engine without a separate inlet duct sensor. Station 0/1 measurements are taken as ambient: Tamb, Pamb.

### 8.1.5 Health-Weight Assumption
The overall health consistency constraint uses fixed engineering importance weights:
- Compressor: 0.40
- Turbine: 0.30
- Combustor: 0.30

These are documented engineering assumptions (not fitted to data) reflecting the relative contribution of each subsystem to overall engine health.

## 8.2 Thermodynamic Relationships

### 8.2.1 Isentropic Process Relations
For an ideal (isentropic) compression or expansion:

T₂/T₁ = (P₂/P₁)^((γ-1)/γ)

Where (γ-1)/γ = 0.4/1.4 ≈ 0.2857 for air.

### 8.2.2 Isentropic Efficiency
**Compressor**: η_c = (ideal work) / (actual work) = ((P₂/P_amb)^0.2857 − 1) / (T₂/T_amb − 1)
**Turbine**: η_t = (actual work) / (ideal work) = (1 − T₄/T₃) / (1 − (P₄/P₃)^0.2857)

### 8.2.3 Brayton Cycle Energy Balance
Q_combustor = ṁ_fuel × LHV = ṁ_air × c_p × (T₃ − T₂)

Solving for air mass flow: ṁ_air = (ṁ_fuel × LHV) / (c_p × ΔT_combustor)

### 8.2.4 Thermal Efficiency
η_thermal = (ṁ_air × W_net) / (ṁ_fuel × LHV)

### 8.2.5 TSFC Tautology
TSFC = ṁ_fuel [g/s] / F_thrust [N]

This is computed deterministically outside the network, making thermodynamic violation mathematically impossible.

## 8.3 Where Physics Enters the Model

Physics enters the model at **four** distinct points:

1. **Feature Engineering** (ThermodynamicsEngine): Transforms raw sensors into thermodynamically meaningful derived features before the network sees the data
2. **Input Space** (concatenation): Physics features are concatenated with raw features, providing gradient shortcuts to semantically meaningful combinations
3. **Loss Function** (PhysicsConstrainedLoss): The health consistency constraint enforces that the model's predicted overall health is consistent with its predicted subsystem healths
4. **Post-Inference** (TSFC computation): TSFC is structurally bound to predicted thrust via the deterministic formula, with 0% violation by construction

## 8.4 The Information Bottleneck Discovery

**Phase 2 Finding**: Initial benchmarking showed that Baseline-Raw (12 raw sensors) consistently outperformed Baseline-PhysFeat (9 physics features). Investigation revealed this was because the ThermodynamicsEngine's feature extraction is a **lossy, many-to-one transformation**. For example, Comp_Isentropic_Efficiency = f(Tamb, T2, Pamb, P2) compresses 4 variables into 1 scalar, irreversibly discarding nuanced interactions.

**Resolution**: Concatenate both representations: x_combined = [x_raw ‖ x_phys]. The Combined model achieves statistical parity with Baseline-Raw (Wilcoxon p=0.8457) while retaining full thermodynamic interpretability.

---

# 9. Loss Functions

## 9.1 PhysicsConstrainedLoss

**Implementation**: `src/models/loss.py`, class `PhysicsConstrainedLoss`

### 9.1.1 Total Loss Formulation

```
L_total = α × L_MSE + β_health × L_Health_Consistency
```

Where α = 1.0 and β_health = 1.0 (or 0.0 for ablation baselines).

### 9.1.2 Component 1: Multi-Head MSE (L_MSE)

```
L_MSE = MSE(ĥ_comp, h_comp) + MSE(ĥ_comb, h_comb) + MSE(ĥ_turb, h_turb) 
      + MSE(ĥ_overall, h_overall) + MSE(F̂_thrust, F_thrust)
```

All quantities are in Z-scored (normalized) space. The loss is the **sum** (not mean) of per-head MSE losses, ensuring balanced gradient flow across all 5 prediction heads.

### 9.1.3 Component 2: Health Consistency Constraint (L_Health)

This constraint operates in **real-world units** (denormalized) and then re-normalizes:

```
# Denormalize predictions to real units
ĥ_comp_real = ĥ_comp × σ_comp + μ_comp
ĥ_comb_real = ĥ_comb × σ_comb + μ_comb  
ĥ_turb_real = ĥ_turb × σ_turb + μ_turb

# Compute expected overall in real units
ĥ_overall_expected_real = 0.40 × ĥ_comp_real + 0.30 × ĥ_turb_real + 0.30 × ĥ_comb_real

# Re-normalize expected overall
ĥ_overall_expected_norm = (ĥ_overall_expected_real − μ_overall) / σ_overall

# Penalty
L_Health = MSE(ĥ_overall_norm, ĥ_overall_expected_norm)
```

**Why denormalize-then-renormalize**: The subsystem weights (0.40, 0.30, 0.30) are defined in real-unit space. Computing the weighted average in Z-score space would produce incorrect results because each subsystem has different mean and scale. The re-normalization ensures the gradient remains in the same scale as L_MSE.

**Fallback**: When target_mean and target_scale are not provided, a simplified version operates directly in normalized space: ĥ_overall_expected = 0.40×ĥ_comp + 0.30×ĥ_turb + 0.30×ĥ_comb.

### 9.1.4 TSFC Constraint (Structural, Not Loss-Based)

TSFC is NOT a network output and is NOT penalized in the loss function. It is computed deterministically:

```
TSFC = FuelFlow_g_s / (Thrust_N + ε)
```

This transforms the TSFC constraint from a learned behavior (which a soft penalty can violate) into a **structural tautology** — mathematically guaranteed 0% violation.

### 9.1.5 Return Values

The loss function returns a 4-tuple: `(total_loss, mse_total, 0.0, health_consistency)`

The third element is hardcoded to 0.0 (TSFC consistency is structurally enforced, not loss-penalized).

## 9.2 InfoNCE Contrastive Loss (Pretraining)

**Implementation**: `src/training/pretrain.py`, function `contrastive_loss()`

```
h1_norm = normalize(h1, dim=-1)  # L2 normalize, shape (B, 32)
h2_norm = normalize(h2, dim=-1)  # L2 normalize, shape (B, 32)

sim_12 = (h1_norm @ h2_norm.T) / τ  # (B, B) similarity matrix
sim_21 = (h2_norm @ h1_norm.T) / τ  # (B, B) similarity matrix

labels = arange(B)  # Diagonal = positive pairs

L = (CrossEntropy(sim_12, labels) + CrossEntropy(sim_21, labels)) / 2
```

Where τ = 0.1 (temperature hyperparameter).

**Intuition**: Positive pairs (adjacent windows from the same flight with the same degradation state) should have high cosine similarity; negative pairs (different engines/flights in the same batch) should have low similarity. The InfoNCE loss maximizes agreement between positive pairs while pushing negatives apart.

---

# 10. Training Pipeline

## 10.1 Main Training (`train.py`)

### 10.1.1 Optimizer
- **Algorithm**: Adam (Kingma & Ba, 2015)
- **Learning rate**: 0.001
- **Weight decay**: 1e-4 (L2 regularization)
- **Betas**: Default (0.9, 0.999)

### 10.1.2 Scheduler
No learning rate scheduler is used. The learning rate remains constant at 0.001 throughout training.

### 10.1.3 Training Loop

```
For epoch in range(300):
    model.train()
    For batch in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        total_loss = criterion(preds, targets, fuel_flow_g, target_mean, target_scale)
        total_loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = sum(mse_total for batch in val_loader) / len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= 20:
        EARLY STOP
```

### 10.1.4 Early Stopping
- **Patience**: 20 epochs
- **Monitor**: Validation MSE loss (not total loss — excludes physics penalty)
- **Criterion**: No improvement in val_loss for 20 consecutive epochs

### 10.1.5 Checkpointing
Best model state_dict is saved in-memory during training. After training completes (or early stops), the best weights are loaded and saved to disk:
- `dist/models/{variant_name}.pth`
- `dist/models/{variant_name}_scaler.joblib`
- `dist/models/{variant_name}_target_scaler.joblib`

### 10.1.6 Training Variants

| Variant | Features | Seq Len | Model Type | α | β_health | Purpose |
|:--------|:---------|:--------|:-----------|:--|:---------|:--------|
| Baseline-Raw | raw (12) | 1 | MLP | 1.0 | 0.0 | Isolate value of physics features |
| PhysFeat-Combined | combined (19) | 1 | MLP | 1.0 | 0.0 | Isolate value of physics-constrained loss |
| Full Model | combined (19) | 5 | GRU | 1.0 | 1.0 | Complete PCMN system |

### 10.1.7 Reproducibility
- **Random seeds**: Not explicitly set in the codebase. Reproducibility relies on PyTorch's default behavior.
- **CUDA determinism**: Not explicitly enforced via `torch.backends.cudnn.deterministic`

### 10.1.8 Model Saving
```python
torch.save(model.state_dict(), f'dist/models/{model_name_lower}.pth')
joblib.dump(scaler, f'dist/models/{model_name_lower}_scaler.joblib')
joblib.dump(target_scaler, f'dist/models/{model_name_lower}_target_scaler.joblib')
```

## 10.2 Contrastive Pretraining (`pretrain.py`)

| Parameter | Value |
|:----------|:------|
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch size | 256 |
| Epochs | 30 |
| Temperature τ | 0.1 |
| Loss | InfoNCE contrastive |
| Saved artifact | `models/ncmapss_pretrained_encoder.pth` (SharedEncoder only) |

## 10.3 Linear Probe Training (`linear_probe.py`)

| Parameter | Value |
|:----------|:------|
| Optimizer | Adam, lr=0.005 |
| Epochs | 120 (fixed, no early stopping) |
| Frozen layers | SharedEncoder (all parameters) |
| Trainable layers | TurbojetAdapter + output heads |
| Evaluation | LOEO-CV across all engines |

## 10.4 LOEO-CV Benchmark Training (`benchmark.py`)

| Parameter | Value |
|:----------|:------|
| Optimizer | Adam, lr=0.005 |
| Epochs | 120 (fixed, no early stopping) |
| Variants | MLP(N=3), GRU(N=3), MLP(N=5), GRU(N=5) |
| Folds | 10 (one per engine) |
| Total models trained | 4 variants × 10 folds = 40 |

---

# 11. Evaluation Framework

## 11.1 RMSE (Root Mean Square Error)

```
RMSE = √(1/N × Σ(ŷᵢ − yᵢ)²)
```

**Why it exists**: Primary competition metric for health estimation accuracy. Reports on Overall Health in real-world units (0-1 scale). Penalizes large errors quadratically, which is appropriate for safety-critical systems where large deviations are disproportionately dangerous.

**Where computed**: `benchmark.py` (per LOEO fold, then mean ± std across 10 folds), `linear_probe.py` (per LOEO fold), `loeo_benchmark.py` (per fold).

## 11.2 MAE (Mean Absolute Error)

```
MAE = 1/N × Σ|ŷᵢ − yᵢ|
```

**Why it exists**: Robust to outliers; provides a linear-scale complement to RMSE. **Status**: Defined in the evaluation protocol but not explicitly computed in the current codebase. **Planned** for future evaluation expansion.

## 11.3 R² (Coefficient of Determination)

```
R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
```

**Why it exists**: Measures the fraction of target variance explained by the model. R²=1 is perfect; R²=0 means the model is no better than predicting the mean. **Status**: Referenced in evaluation protocol but not explicitly computed in current codebase. **Planned**.

## 11.4 Leave-One-Engine-Out Cross-Validation (LOEO-CV)

```
For engine_i in {1, 2, ..., 10}:
    Train on Engines {1,...,10} \ {i}  (270 cycles)
    Test on Engine i                   (30 cycles)
    Record: RMSE_health(i), TSFC_RMSE(i)

Report: mean ± std across 10 folds
```

**Why it exists**: With only 300 samples across 10 engines, a single train/test split is statistically unreliable. LOEO-CV ensures every engine serves as a test case, providing robust estimates of expected performance on unseen engines. Engine-grouped splitting prevents temporal data leakage (training on an engine's future to predict its past).

## 11.5 Wilcoxon Signed-Rank Test

**Implementation**: `scipy.stats.wilcoxon(rmse_a, rmse_b)` — paired, non-parametric

**Why it exists**: The Wilcoxon test is preferred over a paired t-test because the distribution of per-fold RMSE values is not guaranteed to be Gaussian with only 10 folds. It makes no distributional assumptions. A p-value < 0.05 indicates statistically significant performance difference between two model variants.

## 11.6 Effect Size (Rank-Biserial Correlation)

**Implementation**: `loeo_benchmark.py`

```python
diffs = phys_rmse - raw_rmse
w_plus = sum(|diffs[diffs > 0]|)
w_minus = sum(|diffs[diffs < 0]|)
effect_size = (w_plus - w_minus) / (w_plus + w_minus)
```

**Why it exists**: Statistical significance (p-value) indicates whether a difference exists; effect size indicates how large the difference is. A negative rank-biserial correlation means PhysFeat is better than Raw (lower RMSE).

## 11.7 MC Dropout Calibration / Coverage

**Implementation**: Defined in evaluation protocol (system_architecture.md)

```
coverage = fraction of true values in [μ − σ, μ + σ]
```

Expected: ~68% for well-calibrated Gaussian uncertainty. **Status**: Protocol defined; coverage was measured at 0.0% in early experiments due to target scale normalization issues, prompting the target normalization fix.

## 11.8 TSFC Violation

**Implementation**: `src/evaluation/metrics.py`

```
violation% = |TSFC_predicted − TSFC_theoretical| / TSFC_theoretical × 100
```

**Why it exists**: Measures thermodynamic compliance. By construction (deterministic TSFC), this is 0.000% for all models using the hard constraint.

## 11.9 Surrogate Speed Benchmark

**Implementation**: `benchmark.py`

```
Slow path: Per-row ThermodynamicsEngine.extract_physics_features() × N_TRIALS
Fast path: Batched GRU forward pass (single inference)
Speedup = slow_time / fast_time
Per-sample latency = fast_time / N × 10⁶ (μs)
```

**Why it exists**: Satisfies "Surrogate Model Performance" (20% of rubric) and "Computational Efficiency" (10% of rubric). Demonstrates that the trained network replaces expensive iterative physics computation with sub-microsecond inference — critical for edge deployment viability.

---

# 12. Uncertainty Estimation

## 12.1 MC Dropout Method

**Implementation**: `DigitalTwinModel.predict_with_uncertainty()` in `pinn.py`

Monte Carlo Dropout (Gal & Ghahramani, 2016) approximates Bayesian posterior inference by keeping dropout active at test time. Each stochastic forward pass effectively samples a different sub-network from an implicit ensemble.

### 12.1.1 Algorithm

```python
model.train()  # Activate dropout at inference time

for k in range(K):  # K = 10 or 30 stochastic passes
    comp, comb, turb, overall, thrust = model.forward(x)
    # Collect predictions

# For each head:
μ = mean(predictions across K samples)
σ = std(predictions across K samples)
```

### 12.1.2 Confidence Intervals

The denormalized standard deviation provides a 1-sigma confidence bound:

```
σ_real = σ_normalized × σ_train
CI_68% = [μ_real − σ_real, μ_real + σ_real]
CI_95% = [μ_real − 2×σ_real, μ_real + 2×σ_real]
```

### 12.1.3 Calibration

A well-calibrated model should have approximately 68% of true ground-truth values fall within the predicted ±1σ band. The calibration percentage (empirical coverage rate) is the primary metric for assessing whether the model's expressed uncertainty is reliable.

### 12.1.4 Prediction Variance

High variance (large σ) indicates high **epistemic uncertainty** — the model is uncertain about its prediction, typically in regions of the input space poorly covered by training data. Low variance indicates high confidence. For safety-critical applications, high-variance predictions should trigger human review or more conservative operational decisions.

### 12.1.5 Coverage Computation

```
coverage = (1/N) × Σ 1[y_true ∈ (μ − σ, μ + σ)]
```

### 12.1.6 Number of MC Samples

| Context | K (samples) |
|:--------|:------------|
| Real-time telemetry streaming | 10 (speed-constrained) |
| Offline evaluation / benchmarking | 30 (accuracy-prioritized) |

---

# 13. Benchmark Experiments

## 13.1 Phase 1: Hard Physics Constraints (TSFC)

### Objective
Prove that the model achieves 0% thermodynamic violation rate for TSFC.

### Implementation
TSFC is removed from the network's output heads. It is computed deterministically: TSFC = FuelFlow_g/Thrust_N. No soft penalty is needed.

### Output
TSFC Violation Rate = 0.000% for all model variants (by construction).

### Conclusion
The architecture guarantees thermodynamic compliance as a **structural property**, not an empirical result. This eliminates the ~100% TSFC violation observed in earlier iterations where TSFC was predicted as a network output.

## 13.2 Phase 2: Information Bottleneck Resolution

### Objective
Investigate why Baseline-PhysFeat underperformed Baseline-Raw and resolve the "inversion anomaly."

### Implementation
`loeo_benchmark.py` runs LOEO-CV comparing:
- **Baseline-Raw**: DigitalTwinModel(input_dim=12, hidden_dim=32, model_type='mlp') fed raw sensor columns
- **Baseline-PhysFeat**: DigitalTwinModel(input_dim=N_phys, hidden_dim=32, model_type='mlp') fed physics-derived features

Both trained for 150 epochs with Adam(lr=0.001) and MSE loss. Evaluated on Overall Health RMSE.

### Output
Wilcoxon paired test p-value between Raw and PhysFeat RMSE vectors, plus rank-biserial effect size.

### Conclusion
The Combined feature set (raw + physics) achieves statistical parity with Baseline-Raw (p=0.8457), proving the inversion anomaly was a dimensionality/information-loss issue, not a thermodynamic flaw. The physics features are retained for interpretability without performance penalty.

## 13.3 Phase 3: GRU vs MLP Sequence Modeling Ablation

### Objective
Determine whether GRU's recurrent inductive bias provides a measurable advantage over dense concatenation of rolling temporal context on this dataset.

### Implementation
`benchmark.py` runs LOEO-CV for 4 variants:

| Variant | Backbone | Sequence Length | Input Dim |
|:--------|:---------|:---------------|:----------|
| MLP (N=3) | 3-layer MLP | 3 | 3×19=57 (flattened) |
| GRU (N=3) | 1-layer GRU | 3 | 19 per step |
| MLP (N=5) | 3-layer MLP | 5 | 5×19=95 (flattened) |
| GRU (N=5) | 1-layer GRU | 5 | 19 per step |

All use combined features, PhysicsConstrainedLoss with α=1.0 β_health=1.0, Adam(lr=0.005), 120 fixed epochs.

### Output
Per-variant: mean ± std RMSE across 10 LOEO folds, TSFC RMSE, Wilcoxon p-values for MLP vs GRU at each N.

### Conclusion
GRU robustly outperforms MLP at both sequence lengths with high statistical significance:
- N=3: GRU RMSE 0.0290 vs MLP 0.0350, p=0.0039
- N=5: GRU RMSE 0.0307 vs MLP 0.0420, p=0.0020

## 13.4 Phase 1 Transfer Learning: Linear Probe Ablation

### Objective
Determine whether the contrastive-pretrained SharedEncoder learned representations transferable from turbofan (N-CMAPSS) to turbojet.

### Implementation
`linear_probe.py` compares two variants under LOEO-CV:
- **Frozen Random Encoder**: TransferredDigitalTwinModel with randomly initialized, frozen SharedEncoder
- **Frozen Pretrained Encoder**: Same model, SharedEncoder loaded from `ncmapss_pretrained_encoder.pth`, frozen

Only the TurbojetAdapter and output heads are trainable. 120 epochs, Adam(lr=0.005), PhysicsConstrainedLoss.

### Output
Per-variant RMSE ± std, Wilcoxon p-value, SUCCESS/FAILURE verdict.

### Conclusion
**FAILURE**: Pretrained encoder (RMSE 0.0235) performed no better than random initialization (RMSE 0.0224), p=0.8457. Abort criterion triggered. Transfer learning from N-CMAPSS turbofan to turbojet is not viable — the domain gap between turbofan and turbojet engine architectures is too large for representations to transfer through a 32-dimensional bottleneck.

## 13.5 Surrogate Speed Benchmark

### Objective
Quantify the computational advantage of batched neural network inference over iterative physics computation.

### Implementation
```python
# Slow path: per-row physics calculation
for i in range(N_TRIALS):
    _ = thermo.extract_physics_features(df.iloc[[i]])

# Fast path: single batched forward pass
with torch.no_grad():
    _ = model(X_test_t)

speedup = slow_time / fast_time
```

### Output
Speedup factor, per-sample latency (μs), parameter count.

### Conclusion
Achieved >2000× speedup with sub-microsecond per-sample latency, demonstrating edge deployment viability.

---

# 14. Hyperparameter Reference

## Complete Hyperparameter Table

| Category | Parameter | Value | Where Set | Notes |
|:---------|:----------|:------|:----------|:------|
| **Architecture** | hidden_dim | 32 | `pinn.py`, `transfer.py` | Deliberately minimal for N=300 |
| | model_type | `'gru'` (Full Model) | `train.py` | `'mlp'` for baselines |
| | seq_length | 5 | `train.py`, `benchmark.py` | Also tested: 1, 3 |
| | n_features (raw) | 12 | `dataset.py` | Raw sensor channels |
| | n_features (combined) | 19 | `dataset.py` | 12 raw + 7 physics |
| | n_features (phys only) | 9 | `thermodynamics.py` | 7 phys + Altitude + Mach |
| | GRU num_layers | 1 | `pinn.py`, `transfer.py` | |
| | GRU hidden_size | 32 | `pinn.py`, `transfer.py` | |
| | output_heads | 5 | `pinn.py` | comp, comb, turb, overall, thrust |
| | adapter_hidden | 64 | `transfer.py` | DomainAdapter intermediate dim |
| | adapter_output | 32 | `transfer.py` | Engine State Representation Space |
| **Dropout** | dropout_rate | 0.1 | `pinn.py`, `transfer.py` | Active during inference for MC |
| | MC_samples (streaming) | 10 | `telemetry_streamer.py` | Speed-constrained |
| | MC_samples (evaluation) | 30 | `pinn.py` default | Accuracy-prioritized |
| **Training** | optimizer | Adam | `train.py`, `pretrain.py` | |
| | learning_rate (main) | 0.001 | `train.py` | |
| | learning_rate (benchmark) | 0.005 | `benchmark.py`, `linear_probe.py` | Higher for LOEO-CV speed |
| | learning_rate (pretrain) | 0.001 | `pretrain.py` | |
| | weight_decay | 1e-4 | `train.py`, `pretrain.py` | L2 regularization |
| | batch_size (main) | 64 | `dataset.py` | |
| | batch_size (pretrain) | 256 | `pretrain.py` | |
| | max_epochs (main) | 300 | `train.py` | |
| | max_epochs (benchmark) | 120 | `benchmark.py` | Fixed for LOEO speed |
| | max_epochs (pretrain) | 30 | `pretrain.py` | |
| | max_epochs (linear probe) | 120 | `linear_probe.py` | |
| | early_stopping_patience | 20 | `train.py` | On validation MSE |
| **Loss** | alpha (MSE weight) | 1.0 | `loss.py`, `train.py` | |
| | beta_health (health consistency) | 1.0 | `loss.py`, `train.py` | 0.0 for ablation baselines |
| | health_weight_comp | 0.40 | `loss.py` | Engineering assumption |
| | health_weight_turb | 0.30 | `loss.py` | Engineering assumption |
| | health_weight_comb | 0.30 | `loss.py` | Engineering assumption |
| | epsilon (div protection) | 1e-6 | `thermodynamics.py`, `loss.py` | Numerical stability |
| **Contrastive** | temperature τ | 0.1 | `pretrain.py` | InfoNCE temperature |
| | pretrain_seq_len | 128 | `ncmapss_extractor.py` | Per-window timesteps |
| **Physics Constants** | gamma (γ) | 1.4 | `thermodynamics.py` | Specific heat ratio for air |
| | R | 287.05 J/(kg·K) | `thermodynamics.py` | Gas constant for air |
| | LHV | 42.8×10⁶ J/kg | `thermodynamics.py` | Lower Heating Value, Jet-A |
| | cp | 1005 J/(kg·K) | `thermodynamics.py` | Specific heat at constant pressure |
| **Data Split** | test_engines (canonical) | [9, 10] | `train.py` | Organizer-provided |
| | val_engines (canonical) | [7, 8] | `train.py` | |
| **Server** | port | 3000 | `server.js` | Express/Socket.io |
| | offline_threshold | 3000 ms | `server.js` | Watchdog timer |
| | streaming_interval | 0.5 s | `telemetry_streamer.py` | Per-cycle delay |

---

# 15. Current Results

## 15.1 Phase 3 GRU Ablation Results (LOEO-CV)

| Model | Backbone | N | Health RMSE (mean ± std) | TSFC RMSE | Wilcoxon p (vs MLP) |
|:------|:---------|:--|:------------------------|:----------|:--------------------|
| MLP | Dense 3×32 | 3 | 0.0350 ± 0.0101 | 0.0027 | — |
| **GRU** | Recurrent | 3 | **0.0290 ± 0.0075** | **0.0019** | **0.0039** |
| MLP | Dense 3×32 | 5 | 0.0420 ± 0.0065 | 0.0026 | — |
| **GRU** | Recurrent | 5 | **0.0307 ± 0.0078** | **0.0019** | **0.0020** |

## 15.2 Phase 1 Transfer Learning (LOEO-CV)

| Variant | Health RMSE (mean ± std) | Wilcoxon p |
|:--------|:------------------------|:-----------|
| Frozen Random Encoder | 0.0224 ± ? | — |
| Frozen Pretrained Encoder | 0.0235 ± ? | 0.8457 |

**Verdict**: FAILURE. No significant difference. Abort criterion triggered.

## 15.3 Phase 2 Information Bottleneck

Combined model vs Baseline-Raw: Wilcoxon p = 0.8457 (statistical parity). The inversion anomaly was resolved by concatenating raw + physics features.

## 15.4 TSFC Violation

0.000% for all models using deterministic TSFC computation (by construction).

## 15.5 Surrogate Speed

>2000× speedup over iterative physics computation. Sub-microsecond per-sample latency. ~6,300 parameters (GRU variant).

## 15.6 Statistical Significance

All GRU vs MLP comparisons are statistically significant at p < 0.01:
- N=3: p = 0.0039
- N=5: p = 0.0020

## 15.7 Limitations of Current Results

- Results are from LOEO-CV with 120 fixed epochs (no early stopping per fold), potentially suboptimal per-fold convergence
- No explicit random seed control → results may vary across runs
- Calibration coverage not reported in Phase 3 results
- Effect size not reported for GRU vs MLP comparison (only p-values)

---

# 16. Current Limitations

## 16.1 Dataset Limitations

1. **Extremely small sample size**: N=300 total samples across 10 engines. This is dangerously close to the regime where any model can memorize the training set, and generalization guarantees are weak regardless of cross-validation methodology.
2. **Fixed operating profiles**: All engines have exactly 30 cycles. Real engines have vastly different operational histories (thousands to tens of thousands of cycles).
3. **Synthetic data**: While verified to be physics-based (isentropic efficiencies in realistic range), the data comes from a simulation model with idealized sensor characteristics — no sensor noise, calibration drift, or missing values.
4. **No measured air mass flow rate**: The dataset lacks a direct mass flow measurement, forcing the estimation via combustor energy balance with assumed complete combustion and known LHV. This introduces systematic error.
5. **Uncharacteristically high T₃**: Combustor exit temperatures exceed 3000K in some rows, which is above typical turbine material limits and would produce significant divergence from the constant cp=1005 assumption.

## 16.2 Architecture Limitations

1. **No explicit cold-start handling**: Zero-padding for engines with fewer than N cycles is a naive approach — the GRU must learn to ignore zeros, which wastes representational capacity.
2. **No attention mechanism**: The GRU processes the temporal sequence with a fixed, position-invariant recurrence. A self-attention or temporal attention layer could selectively weight more informative timesteps.
3. **Single-layer GRU**: Only one recurrent layer is used. Deeper GRU stacks could capture more complex temporal dynamics but risk overfitting with N=300.
4. **No residual connections**: The shared backbone has no skip connections. For deeper variants, residual connections would improve gradient flow.
5. **Fixed subsystem health weights**: The 0.40/0.30/0.30 weighting in the health consistency constraint is an engineering assumption, not learned from data. The true relationship may be nonlinear or condition-dependent.

## 16.3 Uncertainty Limitations

1. **MC Dropout is an approximation**: It approximates Bayesian inference but is not equivalent. The posterior distribution it samples from may not match the true posterior, especially with limited data.
2. **Calibration not validated in Phase 3**: The coverage percentage was measured at 0.0% in early experiments (target scale normalization bug). Current Phase 3 results do not report calibration.
3. **Dropout rate not tuned**: p=0.1 was chosen heuristically, not optimized for calibration quality.
4. **Only epistemic uncertainty**: MC Dropout captures model uncertainty (epistemic) but not data noise (aleatoric). A heteroscedastic model or deep ensemble would capture both.

## 16.4 Physics Limitations

1. **Constant cp assumption**: The default pipeline uses cp=1005 J/(kg·K) throughout, which is increasingly inaccurate above ~500K. The real-gas variant exists but is not integrated into the default pipeline.
2. **No real-gas effects**: Dissociation, non-ideal gas behavior at high temperatures, and variable specific heat ratio are not modeled.
3. **Complete combustion assumed**: The combustor energy balance assumes 100% combustion efficiency. Real combustors operate at 98-99.5% efficiency.
4. **No bleed air or secondary flows**: The model assumes all air passes through the core cycle. Real engines extract bleed air for cabin pressurization, turbine cooling, and ice protection.
5. **Algebraic, not differential constraints**: The physics constraints are closed-form algebraic relationships, not PDE residuals. This limits the depth of physical reasoning the model can enforce.

---

# 17. Future Roadmap

## 17.1 Three-Pillar Strategy

```
                    ┌─────────────────────────────────┐
                    │     Checkpoint 0 (Current)       │
                    │  Physics-Informed Digital Twin    │
                    │  PCMN on 300-sample dataset      │
                    └────────┬───────┬────────┬────────┘
                             │       │        │
                    ┌────────▼──┐ ┌──▼─────┐ ┌▼────────┐
                    │  Path A   │ │ Path B │ │ Path C  │
                    │Competition│ │N-CMAPSS│ │TurboJet │
                    │ Dataset   │ │Transfer│ │   Sim   │
                    └───────────┘ └────────┘ └─────────┘
```

## 17.2 Checkpoint 0: Current System (Complete)

The current PCMN system is operational with:
- Verified physics-consistent data pipeline
- GRU-based multi-head network with MC Dropout
- LOEO-CV evaluation with Wilcoxon significance testing
- 0% TSFC violation by construction
- >2000× surrogate speedup
- Live dashboard streaming

## 17.3 Path A: Competition Dataset Optimization

### Objectives
- Improve RMSE below 0.03 on LOEO-CV
- Achieve well-calibrated uncertainty (coverage ≈ 68% at 1σ)
- Integrate real-gas thermodynamics into default pipeline
- Add permutation feature importance for explainability

### Methodology
- Integrate `extract_real_gas_features()` as the default feature extractor
- Tune dropout rate for optimal calibration
- Add explicit random seed control for reproducibility
- Implement learning rate scheduling (cosine annealing or reduce-on-plateau)
- Add gradient clipping for training stability

### Expected Benefits
- Marginal RMSE improvement from better physics modeling
- Trustworthy uncertainty bounds for safety-critical decision support
- Reproducible results for publication

### Risks
- Marginal improvements may not be statistically significant with N=300
- Real-gas features may introduce noise at the calorically-perfect operating points

## 17.4 Path B: N-CMAPSS Representation Learning (Concluded)

### Objectives (Original)
Transfer thermodynamic representations from the data-rich N-CMAPSS turbofan dataset to the data-poor turbojet competition dataset.

### Methodology
Contrastive pretraining of a shared GRU encoder on adjacent flight windows (positive pairs) from N-CMAPSS, followed by fine-tuning on turbojet data through a domain-specific adapter.

### Result
**FAILED**. Linear probe ablation showed no significant difference between pretrained and random encoder (p=0.8457). The domain gap between turbofan and turbojet architectures is too large for 32-dimensional representation transfer.

### Risks Realized
- No transfer (Risk #1) — confirmed
- Architecture-specific representations could not bridge the turbofan/turbojet gap

## 17.5 Path C: TurboJetSim (Planned — Future Work)

### Objectives
Build a parametric Brayton cycle simulator specifically for single-spool turbojet engines to generate thousands of realistic degradation trajectories, removing the N=300 bottleneck.

### Methodology
1. Implement a station-by-station Brayton cycle solver:
   - Ambient → Compressor (with efficiency degradation parameter)
   - Compressor → Combustor (with heat addition and pressure loss)
   - Combustor → Turbine (with efficiency degradation parameter)
   - Turbine → Nozzle/Exhaust
2. Parametrize degradation modes: compressor fouling (reduced η_c), turbine erosion (reduced η_t), combustor coking (increased pressure loss, reduced combustion efficiency)
3. Generate run-to-failure trajectories with stochastic degradation injection
4. Validate synthetic data against the 300-sample competition dataset

### Expected Benefits
- Orders of magnitude more training data
- Controllable degradation modes for targeted evaluation
- Native turbojet physics (no domain gap)
- Arbitrary sequence lengths and engine lifetimes

### Risks
- Simulator fidelity may not match the organizers' model
- Sim-to-real gap if the competition model uses different physics
- Development time investment without guaranteed improvement

---

# 18. Architecture Diagram Specification

This section provides sufficient detail for another engineer to recreate a publication-quality architecture diagram without reading the code.

## 18.1 System-Level Architecture Diagram

### Modules (boxes, left to right)

1. **Raw Sensor Input** (leftmost)
   - Label: "Raw Sensor Telemetry"
   - Sub-label: "12 channels"
   - Contents: {Tamb, Pamb, T2, P2, T3, P3, T4, P4, RPM, Fuel_Flow, Altitude, Mach}
   - Shape: Rectangle

2. **ThermodynamicsEngine** (grey-box)
   - Label: "Grey-Box Feature Engineering"
   - Sub-label: "ThermodynamicsEngine (Deterministic)"
   - Contents: PR_comp, PR_turb, η_c, η_t, W_net, ṁ_air, η_thermal
   - Shape: Rectangle with dashed border (indicating non-learned, deterministic)
   - Color: Light grey fill

3. **Feature Concatenation**
   - Label: "Combined Feature Space"
   - Sub-label: "x ∈ ℝ¹⁹"
   - Shape: Diamond (merge node)
   - Two inputs: arrow from Raw Sensor (12 dims) + arrow from ThermodynamicsEngine (7 dims)

4. **StandardScaler**
   - Label: "Z-Score Normalization"
   - Sub-label: "Fit on train only"
   - Shape: Small rounded rectangle

5. **Sliding Window Buffer**
   - Label: "Temporal Window"
   - Sub-label: "N=5 timesteps"
   - Shape: Stack of 5 horizontal bars
   - Annotation: "x ∈ ℝ^(5×19)"

6. **GRU Encoder** (central, prominent)
   - Label: "GRU Recurrent Backbone"
   - Sub-label: "hidden=32, layers=1"
   - Contents: Show 5 GRU cells in sequence with hidden state arrow flowing left-to-right
   - Final cell has emphasized output arrow: "h_N ∈ ℝ³²"
   - Shape: Large rounded rectangle

7. **MC Dropout + Dense Layer**
   - Label: "MC Dropout (p=0.1) + Dense(32→32) + ReLU"
   - Shape: Rounded rectangle

8. **Multi-Head Output** (rightmost, 5 parallel branches)
   - 5 boxes arranged vertically:
     - "Compressor Health (32→1)"
     - "Combustor Health (32→1)"
     - "Turbine Health (32→1)"
     - "Overall Health (32→1)"
     - "Thrust (32→1)"
   - Each has an arrow from the shared representation
   - Shape: Small rectangles

9. **Deterministic TSFC** (separate from network, below Thrust)
   - Label: "TSFC = ṁ_fuel / Thrust"
   - Sub-label: "Structural Tautology (0% violation)"
   - Shape: Rectangle with solid border
   - Color: Green fill (indicating physics guarantee)
   - Input: Arrow from Thrust output + arrow from Fuel_Flow (original raw sensor)

### Arrows

| From | To | Label | Style |
|:-----|:---|:------|:------|
| Raw Sensor Input | ThermodynamicsEngine | "12 channels" | Solid |
| Raw Sensor Input | Feature Concatenation | "12 raw" | Solid |
| ThermodynamicsEngine | Feature Concatenation | "7 physics" | Dashed (derived) |
| Feature Concatenation | StandardScaler | "19 combined" | Solid |
| StandardScaler | Sliding Window | "normalized" | Solid |
| Sliding Window | GRU Encoder | "(B, 5, 19)" | Solid, thick |
| GRU Encoder | MC Dropout + Dense | "h_N ∈ ℝ³²" | Solid |
| MC Dropout + Dense | Each Head | "(B, 32)" | Solid, branching |
| Thrust Head | Deterministic TSFC | "Thrust_N" | Solid |
| Raw Sensor Input | Deterministic TSFC | "FuelFlow_g" | Dotted (bypass) |
| Overall Health Head | Loss Function | "ĥ_overall" | Dashed (training only) |
| Comp/Turb/Comb Heads | Loss Function | "ĥ_subsystems" | Dashed (training only) |

### Annotations
- Loss function box (below the network): "L = α·L_MSE + β·L_Health_Consistency"
- Parameter count: "~6,300 trainable parameters"
- Inference speed: "Sub-μs per sample"

## 18.2 Four-Stage Engine Station Diagram

```
┌────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐
│Ambient │───▶│ Compressor │───▶│ Combustor  │───▶│  Turbine  │───▶ Exhaust
│Sta 0/1 │    │  Sta 2     │    │  Sta 3     │    │  Sta 4    │
│Tamb,Pamb│   │ T2, P2     │    │ T3, P3     │    │ T4, P4    │
│        │    │ η_c, PR_c  │    │ Q_comb     │    │ η_t, PR_t │
│        │    │ W_comp     │    │            │    │ W_turb    │
└────────┘    └────────────┘    └────────────┘    └───────────┘
                                                        │
                                                   RPM, Fuel_Flow
```

Each box should be colored by its subsystem health score (green=healthy, yellow=degraded, red=critical).

---

# 19. Presentation Assets

## 19.1 System Overview Diagram

**Content**: The complete end-to-end pipeline from Section 2.1. Show 8 stages as a vertical flow: Raw Telemetry → Feature Engineering → Preprocessing → Sequence Generation → Model → Prediction → Evaluation → Deployment.

**Key emphasis**: The grey-box nature of the system — physics enters at multiple stages, not just the loss function.

## 19.2 Data Preprocessing Diagram

**Content**: Detailed flow from raw CSV files through merging, renaming, physics feature extraction, concatenation, normalization, sequence windowing, to PyTorch DataLoaders.

**Key emphasis**: Train-only scaler fitting, engine-boundary-respecting windows, zero-padding with masks.

## 19.3 Model Architecture Diagram

**Content**: The GRU network from Section 18.1 with all tensor dimensions annotated. Show the 5 output heads branching from the shared representation, and the deterministic TSFC computation as a separate, physics-guaranteed pathway.

**Key emphasis**: MC Dropout active at inference, deterministic TSFC as structural tautology.

## 19.4 Training Pipeline Diagram

**Content**: Training loop with optimizer, loss computation (MSE + health consistency), early stopping, and model checkpointing.

**Key emphasis**: Health consistency constraint denormalize-then-renormalize flow.

## 19.5 Inference Pipeline Diagram

**Content**: Streaming inference: raw telemetry → ThermodynamicsEngine → concatenation → scaler → sliding buffer → GRU → denormalize → deterministic TSFC → HTTP POST → WebSocket → Dashboard.

**Key emphasis**: Buffer management (cold-start handling), MC Dropout for uncertainty.

## 19.6 Research Roadmap Diagram

**Content**: Three-pillar strategy from Section 17. Show Checkpoint 0 (current) branching into Path A (Competition optimization), Path B (N-CMAPSS, crossed out as concluded/failed), Path C (TurboJetSim, marked as planned).

**Key emphasis**: Honest reporting of Path B failure and pivot to Path C.

## 19.7 Three-Pillar Strategy Diagram

**Content**: Expanded version of roadmap showing per-path objectives, risks, and expected outcomes. Path B should have a clear "ABORT" marker with the p=0.8457 result.

## 19.8 Ablation Results Bar Chart

**Content**: Phase 3 results showing 4 bars (MLP N=3, GRU N=3, MLP N=5, GRU N=5) with error bars from LOEO-CV. Annotate Wilcoxon p-values between MLP/GRU pairs.

**Key emphasis**: This is the single strongest visual — it shows that the physics-constrained GRU with temporal context outperforms all alternatives with statistical significance.

---

# 20. Research Contributions

## 20.1 Engineering Contributions

| Contribution | Novel vs. Known | Description |
|:-------------|:----------------|:------------|
| **Deterministic TSFC enforcement** | Novel application | Removing TSFC from network outputs and computing it deterministically achieves 0% thermodynamic violation by construction — more robust than soft penalty approaches |
| **Combined feature space resolution** | Novel finding | Discovered and resolved the Information Bottleneck where physics features alone underperformed raw sensors; combined concatenation preserves both interpretability and performance |
| **Collinearity-aware feature pruning** | Standard practice, rigorously applied | Systematic identification and removal of collinear physics features (6 features dropped) to prevent rank deficiency |
| **Engine-grouped data splitting** | Known best practice | Standard in prognostics literature (Saxena et al., 2008); rigorously applied here with LOEO-CV |
| **Surrogate speed benchmarking** | Known methodology | Standard SBAO practice (Queipo et al., 2005); applied here with >2000× speedup measurement |

## 20.2 Algorithmic Contributions

| Contribution | Novel vs. Known | Description |
|:-------------|:----------------|:------------|
| **Denormalize-renormalize physics constraint** | Novel implementation detail | Computing health consistency in real-world units then re-normalizing ensures correct subsystem weighting while maintaining gradient scale compatibility with MSE |
| **Multi-head GRU with physics loss** | Novel combination | GRU backbone + 5 independent output heads + physics consistency loss = first documented application to turbojet health monitoring at this architecture scale |
| **MC Dropout across multi-head architecture** | Extension of known technique | MC Dropout (Gal & Ghahramani, 2016) applied independently per output head with per-head uncertainty estimates |
| **Sliding window with zero-padding and masking** | Standard technique, clean implementation | Cold-start handling via explicit zero-padding and binary masks for engines with insufficient history |

## 20.3 Physics Contributions

| Contribution | Novel vs. Known | Description |
|:-------------|:----------------|:------------|
| **Hand-verified dataset physics** | Novel validation | Independent verification that competition dataset obeys real thermodynamics (η_c = 78-90%) — not just asserted, but computed by hand |
| **LHV-based air mass flow estimation** | Known thermodynamic relation, novel application | Using combustor energy balance with Jet-A LHV to estimate unmeasured air mass flow rate from the sensor suite |
| **Temperature-dependent cp(T) variant** | Known thermodynamic correction | Linear approximation cp(T) = 1005 + 0.0722×(T−300) for improved accuracy at high temperatures |
| **Terminology precision (PCMN vs PINN)** | Methodological rigor | Explicitly distinguishing "Physics-Constrained Multi-Head Network" from "Physics-Informed Neural Network" (Raissi et al., 2019) because the available physics are algebraic, not differential |

## 20.4 Evaluation Contributions

| Contribution | Novel vs. Known | Description |
|:-------------|:----------------|:------------|
| **LOEO-CV with Wilcoxon testing** | Standard statistical practice, rigorously applied | Non-parametric significance testing avoids Gaussian assumptions inappropriate for 10-fold RMSE distributions |
| **Three-phase ablation framework** | Novel experimental design | Systematic isolation of (1) physics constraint value, (2) feature space composition, (3) temporal modeling benefit through controlled ablations |
| **Transfer learning abort criterion** | Novel protocol | Formal abort criterion (linear probe no better than random → terminate path) prevents sunk-cost fallacy in transfer learning |
| **Multi-metric evaluation** | Best practice | Simultaneous reporting of RMSE, TSFC RMSE, p-values, speedup, and parameter count rather than cherry-picked single metrics |

## 20.5 Research Contributions (Distinguishing Novel from Literature)

### Already Established in Literature
- MC Dropout for uncertainty estimation (Gal & Ghahramani, 2016)
- Grey-box physics-data fusion (Arias Chao et al., 2022)
- Digital Twin concept (Grieves & Vickers, 2017)
- Surrogate-based analysis and optimization (Queipo et al., 2005)
- C-MAPSS/N-CMAPSS engine degradation datasets (Saxena et al., 2008; Arias Chao et al., 2021)
- GRU for time-series processing (Cho et al., 2014)
- Contrastive self-supervised learning (Chen et al., 2020)

### Novel to This Project
- The specific combination of deterministic TSFC enforcement + combined feature space + GRU multi-head architecture + physics-constrained loss for turbojet health monitoring
- The Information Bottleneck diagnosis and resolution for physics features
- The three-phase iterative architecture development methodology (each phase motivated by specific observed failures)
- The formal transfer learning abort criterion with linear probe evaluation
- The honest reporting of transfer learning failure as a negative result (Path B abort at p=0.8457)
- Hand-verification of dataset physics as a first-principles validation step

---

# Appendix A: Verified Literature References

| Reference | Citation | Used To Justify |
|:----------|:---------|:----------------|
| Raissi, Perdikaris & Karniadakis (2019) | J. Comput. Phys. 378:686–707 | PCMN vs PINN terminology distinction |
| Saxena, Goebel, Simon & Eklund (2008) | PHM Society/IEEE, DOI 10.1109/PHM.2008.4711414 | Engine-level train/test split, station numbering |
| Arias Chao, Kulkarni, Goebel & Fink (2021) | N-CMAPSS dataset paper | N-CMAPSS dataset provenance, full-life trajectory framing |
| Arias Chao, Kulkarni, Goebel & Fink (2022) | Reliability Eng. & System Safety 217:107961 | Grey-box physics-data fusion architecture pattern |
| Gal & Ghahramani (2016) | ICML — "Dropout as a Bayesian Approximation" | MC Dropout uncertainty methodology |
| Gerdes (2019) | LTU doctoral thesis | Interpretable-model-over-black-box framing |
| Grieves & Vickers (2017) | Transdisciplinary Perspectives on Complex Systems, Springer, pp.85–113 | Foundational Digital Twin definition |
| Queipo, Haftka, Shyy, Goel, Vaidyanathan & Tucker (2005) | Progress in Aerospace Sciences 41(1):1–28 | Surrogate-based analysis and optimization methodology |
| Forrester & Keane (2009) | Progress in Aerospace Sciences 45(1–3):50–79 | Surrogate-model validation practice |
| Farhat & Altarawneh (2025) | Energies 18(20):5523 | Physics-constrained neural networks classification |

---

# Appendix B: Glossary

| Term | Definition |
|:-----|:-----------|
| **PCMN** | Physics-Constrained Multi-Head Network — the project's neural architecture |
| **PINN** | Physics-Informed Neural Network (Raissi et al., 2019) — embeds PDE residuals in loss; NOT what this project builds |
| **LOEO-CV** | Leave-One-Engine-Out Cross-Validation — 10-fold CV grouped by engine |
| **MC Dropout** | Monte Carlo Dropout — Bayesian approximation via stochastic inference |
| **TSFC** | Thrust-Specific Fuel Consumption — fuel efficiency metric (g/N/s) |
| **LHV** | Lower Heating Value — energy content of fuel (42.8 MJ/kg for Jet-A) |
| **GRU** | Gated Recurrent Unit — recurrent neural network architecture |
| **InfoNCE** | Noise-Contrastive Estimation — contrastive learning loss function |
| **Grey-box** | Model combining physics-based and data-driven components |
| **Brayton cycle** | Thermodynamic cycle used by gas turbine engines |
| **Station numbering** | Aerospace convention: 0/1=inlet, 2=compressor exit, 3=combustor exit, 4=turbine exit |
| **Isentropic** | An ideal, reversible adiabatic process (no entropy change) |

---

*This document is the single source of truth for the Zero and Already Behind project. All future documentation, presentations, and publications should reference this document as the definitive technical specification.*
