# 🔥 ROAST.MD — Revision 2 Post-Mortem of "Zero and Already Behind"

**Date:** July 14, 2026  
**Evaluator:** Independent Code & Architecture Auditor  
**Scope:** Full re-audit after Revision 2 changes guided by `Zero_and_Already_Behind_Scientific_Proposal 1.md` and `DASHBOARD_BUILD_GUIDE.md`. Every updated Python file, the benchmark output, the training output, and the telemetry streamer were re-examined.  
**Prior Audit:** The original `roast.md` (July 13) identified 10+ critical defects. This revision checks which were actually fixed, which were partially fixed, and which new problems the fixes introduced.

---

## EXECUTIVE SUMMARY

**Revision 2 fixed the worst sins but introduced new ones.** The hardcoded benchmark is gone. The physics loss now constrains outputs. There's a proper train/val/test split. An ablation framework exists. These are real, material improvements — the project has gone from "theatre" to "an honest attempt with serious engineering problems."

But the training output you just showed me tells a brutal story:

```
Baseline-Raw:       TSFC violation=100.00%  Overall-health RMSE=0.0879
Baseline-PhysFeat:  TSFC violation=99.99%   Overall-health RMSE=0.0879
Full Model:         TSFC violation=99.99%   Overall-health RMSE=0.0879
Coverage (mean ± 1 std): 0.0%
```

**Every model variant has ~100% TSFC violation. All three produce identical RMSE. The uncertainty calibration coverage is literally 0%.** The physics-constrained loss made zero measurable difference. The ablation — which the Scientific Proposal correctly identified as "your single strongest visual" — currently proves that the physics constraints **do not help at all**.

And the telemetry streamer **still cheats** (lines 132–133 of the updated file). The exact same `constrained_tsfc = fuel_flow_g / thrust` and `pinn_violation = 0.0` hack survived the rewrite.

---

## TABLE OF CONTENTS

1. [What Was Actually Fixed (Credit Where Due)](#1-what-was-actually-fixed)
2. [The Training Output: A Model That Hasn't Learned](#2-the-training-output-a-model-that-hasnt-learned)
3. [The Telemetry Streamer: STILL Cheating](#3-the-telemetry-streamer-still-cheating)
4. [The Loss Function: Fixed in Spirit, Broken in Practice](#4-the-loss-function-fixed-in-spirit-broken-in-practice)
5. [The Dataset Split: Good Idea, Problematic Implementation](#5-the-dataset-split-good-idea-problematic-implementation)
6. [The ThermodynamicsEngine: Better, But Still Incomplete](#6-the-thermodynamicsengine-better-but-still-incomplete)
7. [The Model Architecture: Undersized for the Problem](#7-the-model-architecture-undersized-for-the-problem)
8. [The Benchmark: Honest Now, But Revealing Failure](#8-the-benchmark-honest-now-but-revealing-failure)
9. [Missing Deliverables: Dashboard Rebuild Not Started](#9-missing-deliverables-dashboard-rebuild-not-started)
10. [Remaining Critical Fixes](#10-remaining-critical-fixes)
11. [Revised Verdict & Scorecard](#11-revised-verdict--scorecard)

---

## 1. What Was Actually Fixed

Before tearing into what's broken, let's acknowledge what Revision 2 got right — these were real engineering improvements, not cosmetic:

| Original Defect | Status | What Changed |
|----------------|--------|-------------|
| Physics loss constrains inputs, not outputs | ✅ **FIXED** | [loss.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/models/loss.py) now has `PhysicsConstrainedLoss` with TSFC consistency and health consistency constraints on the model's **outputs** |
| No train/test split | ✅ **FIXED** | [dataset.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/data_pipeline/dataset.py) and [train.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/training/train.py) implement engine-level split: Train engines, Val engines (7,8), Test engines (9,10) |
| 10 epochs of training | ✅ **FIXED** | Now trains for 300 epochs with patience-20 early stopping |
| Benchmark uses hardcoded print statements | ✅ **FIXED** | [benchmark.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/evaluation/benchmark.py) computes all metrics from actual model predictions |
| No ablation comparison | ✅ **FIXED** | 3-variant ablation (Baseline-Raw, Baseline-PhysFeat, Full Model) with the same architecture |
| PINN inputs unnormalized | ✅ **FIXED** | `StandardScaler` fitted on train only, applied to all splits |
| `hidden_dim=128` over-parameterized | ✅ **FIXED** | Reduced to `hidden_dim=32` with `weight_decay=1e-4` |
| Competitor baseline trained on random noise | ✅ **FIXED** | The `train_baseline.py` strawman is no longer used; all comparisons are now internal ablations |
| MC Dropout calibration not measured | ✅ **FIXED** | [benchmark.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/evaluation/benchmark.py#L80-L95) computes actual coverage percentage |
| Surrogate speed not measured | ✅ **FIXED** | Benchmark now includes real timing with warmup |
| Results not saved for dashboard | ✅ **FIXED** | Saves `benchmark_results.json` to `public/data/` |

**This is genuine progress.** The structural honesty issues identified in the original roast have been mostly addressed. But the model **still doesn't work**, and some critical issues from the original audit survived intact.

---

## 2. The Training Output: A Model That Hasn't Learned

### The Numbers Don't Lie

From the training output you shared:

| Variant | Final Train Loss | Final Val Loss | Converging? |
|---------|-----------------|---------------|-------------|
| Baseline-Raw | 211,030,058 | 269,755,584 | Slowly, still dropping |
| Baseline-PhysFeat | 531,626,954 | 612,562,368 | Slowly, still dropping |
| Full Model | 592,865,962 | 659,882,304 | Slowly, still dropping |

### Problem 1: The losses are in the hundreds of millions

These are **raw MSE losses with no normalization** on the target values. The Thrust_N column has values in the range ~20,000–60,000 N. A single squared error on thrust dominates: `(50000 - 25000)² = 625,000,000`. This means:

- **The loss is completely dominated by the Thrust head** — the health heads (values 0-1, squared errors ~0.001) contribute essentially nothing to the gradient
- The model is trying to minimize `MSE(thrust_pred, thrust_true)` and effectively ignoring all 5 other outputs
- This is why all three variants give identical RMSE=0.0879 for overall health — the health heads barely get gradient signal

### Problem 2: Baseline-Raw outperforms the physics-constrained model

Look at the final losses:
- **Baseline-Raw:** 269M val loss — **lowest**
- **Baseline-PhysFeat:** 612M val loss — 2.3× worse
- **Full Model:** 659M val loss — 2.4× worse

The physics features and constraints are **making the model worse**, not better. This is the exact opposite of what you need to demonstrate. The ablation — which the Scientific Proposal correctly calls "your single strongest visual" — currently proves that your entire thesis is wrong.

### Problem 3: No early stopping triggered

None of the three variants triggered early stopping. All three ran the full 300 epochs and were still improving. This means:
- The `patience=20` is working but the models need significantly more epochs
- Or the learning rate is too high/too low for convergence
- The losses are plateau-ing in the hundreds of millions, suggesting a fundamental scaling issue

### Problem 4: The benchmark confirms total failure

```
Baseline-Raw:       TSFC violation=100.00%  Overall-health RMSE=0.0879
Baseline-PhysFeat:  TSFC violation=99.99%   Overall-health RMSE=0.0879
Full Model:         TSFC violation=99.99%   Overall-health RMSE=0.0879
```

**100% TSFC violation** means the model's predicted TSFC has absolutely no relationship to `FuelFlow / Thrust`. The physics constraint in the loss function is present but ineffective — the model hasn't learned the TSFC relationship at all.

**Identical RMSE** across all three variants means the physics features and constraints made zero measurable difference to health prediction accuracy.

**0% calibration coverage** means the MC-Dropout uncertainty bands are so narrow that the true value never falls within them — the model is confidently wrong about everything.

---

## 3. The Telemetry Streamer: STILL Cheating

### The Crime That Survived the Rewrite

[telemetry_streamer.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/evaluation/telemetry_streamer.py#L131-L133), lines 131–133:

```python
# Enforce strict thermodynamic rules on PINN output explicitly (Physics-Informed inference)
constrained_tsfc = fuel_flow_g / thrust if thrust > 0 else 0.0
pinn_violation = 0.0 # Mathematically perfectly constrained
```

**This is the exact same cheat from the original codebase.** The model predicts a TSFC value. The streamer throws it away, hand-calculates TSFC from `fuel_flow / thrust`, sends THAT to the dashboard, and reports a violation of 0.0%.

The Scientific Proposal (Section 5) explicitly identified this as the corrected approach — constrain the loss, not the output. But the telemetry_streamer.py was never updated to match. The dashboard is still showing fabricated physics consistency.

### Additional Streamer Issues

- **Line 65:** Still loads from `turbojet_complete_dataset.csv` instead of using the proper train/test split — the streamer is streaming training data, not held-out test data
- **Line 97:** `physics_consistency = min(efficiency * 100, 100)` — still a meaningless metric (raw efficiency × 100 ≠ physics consistency score)
- **Lines 100-116:** The XGBoost and GRU baselines are loaded from the OLD `src/models/` directory (the ones trained on the original unsplit data), not the new ablation variants from `dist/models/`

### The Fix

```python
# REMOVE lines 131-133 and REPLACE with:
# Use the model's actual TSFC prediction
pinn_violation = calc_violation(tsfc, thrust)  # Same function used for baselines

payload = {
    # ...
    "tsfc": tsfc,           # The actual model prediction, not hand-calculated
    "pinn_violation": pinn_violation,  # The actual violation, not 0.0
}
```

---

## 4. The Loss Function: Fixed in Spirit, Broken in Practice

### What's Right

The [new loss.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/models/loss.py) correctly implements two constraints on the model's **outputs**:
1. TSFC consistency: `MSE(tsfc_pred, fuel_flow_g / thrust_pred)`
2. Health consistency: `MSE(overall_pred, 0.40*comp + 0.35*turb + 0.25*comb)`

Gradients flow through the outputs. This is structurally correct.

### What's Wrong: The Scale Problem

The loss sums 6 raw MSE terms:
```python
mse_total = (MSE(comp, comp_t) + MSE(comb, comb_t) + MSE(turb, turb_t) + 
             MSE(overall, overall_t) + MSE(thrust, thrust_t) + MSE(tsfc, tsfc_t))
```

But the targets have wildly different scales:
| Output | Range | Typical squared error |
|--------|-------|--------------------|
| CompressorHealth | 0.85–1.0 | ~0.001 |
| CombustorHealth | 0.85–1.0 | ~0.001 |
| TurbineHealth | 0.85–1.0 | ~0.001 |
| OverallHealth | 0.85–1.0 | ~0.001 |
| **Thrust_N** | **20,000–60,000** | **~625,000,000** |
| TSFC_g_N_s | 0.01–0.1 | ~0.001 |

**Thrust dominates the loss by a factor of ~10⁹.** The health heads receive essentially zero gradient. The TSFC head receives essentially zero gradient. This is why:
- All three models give the same RMSE=0.0879 for health (the health heads barely update)
- TSFC violation is 100% (the TSFC head barely updates)
- The model only learns thrust (poorly, given the magnitude)

### The Fix: Normalize Targets or Use Per-Head Weighting

```python
# Option 1: Normalize targets (preferred)
# Scale Thrust_N to [0,1] range before training using the same StandardScaler approach
# This ensures all 6 heads contribute equally to the gradient

# Option 2: Per-head loss weighting
mse_total = (self.mse(comp_p.squeeze(), comp_t) + 
             self.mse(comb_p.squeeze(), comb_t) +
             self.mse(turb_p.squeeze(), turb_t) +
             self.mse(overall_p.squeeze(), overall_t) +
             self.mse(thrust_p.squeeze(), thrust_t) / (thrust_scale**2) +  # Normalize by target variance
             self.mse(tsfc_p.squeeze(), tsfc_t) * tsfc_weight)
```

The Scientific Proposal missed this entirely. It focused on the structural correctness of the loss (constraining outputs vs. inputs) but never addressed the scale mismatch — the single biggest reason the model isn't learning.

---

## 5. The Dataset Split: Good Idea, Problematic Implementation

### What's Right

[dataset.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/data_pipeline/dataset.py) and [train.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/training/train.py) implement:
- Engine-based splits: Train (engines not in {7,8,9,10}), Val (engines 7,8), Test (engines 9,10)
- Scaler fitted on train only
- Test engines never seen during training
- Results: Train=180, Val=60, Test=60

### What's Wrong

**Line 27 of dataset.py:**
```python
df_all_sensors = pd.concat([df_train, df_test], ignore_index=True).drop_duplicates(subset=['EngineID', 'Cycle'])
```

This **concatenates train.csv and test.csv** into a single pool, then re-splits by EngineID. This means:
1. The hackathon's own intended split (train.csv vs test.csv) is destroyed
2. The split is now entirely determined by `test_engines=[9, 10]` in `get_engine_split()`
3. If the hackathon intended specific engines in train vs test (which the Scientific Proposal's own analysis suggests), this ignores that intent

The Scientific Proposal (Section 1) explicitly noted that `train.csv` and `test.csv` contain different EngineIDs and recommended using them as-is. The implementation contradicts its own proposal by re-pooling them.

### Additional Issue: GroupKFold is imported but never used

[dataset.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/data_pipeline/dataset.py#L6) line 6:
```python
from sklearn.model_selection import GroupKFold
```
This import is unused — the Scientific Proposal recommended optional GroupKFold for hyperparameter tuning within training, but it was never implemented.

---

## 6. The ThermodynamicsEngine: Better, But Still Incomplete

### What Was Added

[thermodynamics.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/data_pipeline/thermodynamics.py) now computes 11 features (up from 5):

| Feature | Status | Notes |
|---------|--------|-------|
| PR_comp | ✅ Kept | |
| PR_turb | ✅ Kept | |
| Comp_Isentropic_Efficiency | ✅ Kept | |
| Combustion_Temp_Rise | ✅ Kept | |
| Normalized_RPM | ✅ Kept | |
| **Turb_Isentropic_Efficiency** | ✅ **NEW** | (1 - T4/T3) / (1 - (P4/P3)^0.2857) — directly from the Scientific Proposal |
| **Compressor_Specific_Work** | ✅ **NEW** | cp × (T2 - Tamb) |
| **Turbine_Specific_Work** | ✅ **NEW** | cp × (T3 - T4) |
| **Net_Specific_Work** | ✅ **NEW** | Turb - Comp specific work |
| **Combustor_Heat_Addition** | ✅ **NEW** | cp × (T3 - T2) |
| **Overall_Pressure_Ratio** | ✅ **NEW** | P3 / Pamb |

### What's Still Missing

The Scientific Proposal (Section 4) explicitly recommended adding `Fuel-Flow-Normalized Work = Net_Specific_Work / Fuel_Flow`:

> "Finally uses Fuel_Flow in the physics layer — a genuine efficiency-like indicator"

This feature — the one the Proposal called the most important addition because it finally uses the Fuel_Flow column — **was not implemented.** The physics layer still does not use the Fuel_Flow column.

### Additional Concern: Redundant Features

`Combustion_Temp_Rise = T3 - T2` and `Combustor_Heat_Addition = cp × (T3 - T2)` are linearly related (differ by a constant factor of 1005). The Scientific Proposal itself flagged this:

> "Same as Combustion_Temp_Rise scaled by cp — keep both only if you show they're used differently downstream"

Both are kept. Since `StandardScaler` normalizes them, they become numerically identical inputs — one should be removed to avoid wasting capacity in the 32-wide hidden layer.

---

## 7. The Model Architecture: Undersized for the Problem

### The Change

`hidden_dim` was reduced from 128 to 32 per the Scientific Proposal's recommendation. With 3 shared layers of width 32 and 6 output heads, the model has **~2,700 parameters** (from ~145K before).

### The Concern

32 neurons × 3 layers is very tight for a model that must simultaneously learn:
- 3 subsystem health trajectories (compressor, combustor, turbine)
- 1 composite overall health
- Thrust (a highly nonlinear function of RPM, fuel flow, pressure ratios, temperature)
- TSFC (a ratio of fuel flow to thrust)

With 11 physics features as input and 6 multi-scale outputs, the representational capacity may be insufficient. The training curves show losses still declining after 300 epochs with no early stopping triggered — this is consistent with an underfitting model that needs more capacity, not less.

### Recommendation

Try `hidden_dim=64` as a middle ground. The Scientific Proposal recommended "32–64" — the implementation went with the lower bound. Given that the model isn't converging, the upper bound is worth testing. The `weight_decay=1e-4` is appropriate regularization to prevent overfitting at N=300.

---

## 8. The Benchmark: Honest Now, But Revealing Failure

### What's Good

[benchmark.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/evaluation/benchmark.py) is now scientifically sound:
- All metrics computed from actual model predictions
- TSFC violation computed correctly: `|tsfc_pred - fuel_flow/thrust_pred| / (fuel_flow/thrust_pred) × 100`
- MC-Dropout calibration coverage computed against held-out test engines
- Surrogate speed measured with warmup and real timing
- Results saved to JSON for dashboard consumption

### What's Revealed

The honest benchmark exposes that the model isn't working:

| Metric | Result | What It Means |
|--------|--------|--------------|
| TSFC violation | ~100% all variants | The model has not learned the TSFC relationship at all — predictions bear no resemblance to `FuelFlow/Thrust` |
| Overall RMSE | 0.0879 (identical) | Physics features and constraints make zero difference; suggests health heads aren't receiving gradient |
| MC-Dropout coverage | 0.0% | The model is maximally overconfident — uncertainty bands are far too narrow to capture any true values |
| Surrogate speedup | 935.7× | This number is legitimate and impressive — the only positive result |

### The Speedup Is Real, But Misleading

The 935.7× speedup is real: batched GPU/CPU tensor inference is dramatically faster than Python-loop pandas operations per row. But note the comparison:
- **Slow path:** Python `for i in range(N): df.iloc[[i]]` — this is a slow Python loop with pandas indexing overhead. A vectorized `thermo_engine.extract_physics_features(df)` call on the full dataframe would be much faster.
- **Fast path:** Single batched `model(X_tensor)` — this is inherently faster regardless of model quality.

The speedup measures "Python loop vs. batched tensor ops," not "high-fidelity simulation vs. surrogate." The Scientific Proposal's Section 11 acknowledged this by calling it the "slow path" proxy, which is defensible — but the 935× number overstates the real speedup a turbojet operator would care about. A more honest framing: "0.003 ms per sample enables real-time inference at rates far exceeding typical 1–10 Hz sensor sampling rates."

---

## 9. Missing Deliverables: Dashboard Rebuild Not Started

The `DASHBOARD_BUILD_GUIDE.md` is excellent — detailed, aviation-grounded, and specific about every panel. But **none of it has been implemented.** The dashboard (index.html, js/app.js) is still the original version from before the roast:

| Required Panel | Status |
|---------------|--------|
| Engine operating conditions (raw telemetry) | ❌ Not implemented |
| Predicted thrust (own panel) | ❌ Not implemented |
| 4-stage engine schematic (signature element) | ❌ Not implemented |
| Per-engine trajectory view | ❌ Not implemented |
| Calibration panel (MC-Dropout coverage) | ❌ Not implemented |
| Ablation result panel | ❌ Not implemented |
| Surrogate speed panel | ❌ Not implemented |
| Explainability panel | ❌ Not implemented |
| Model Card | ❌ Not implemented |
| Aviation design token system | ❌ Not implemented |

The `DASHBOARD_BUILD_GUIDE.md` is a build specification, not a build. The dashboard is still the old SaaS-styled version displaying rigged telemetry from the unreformed streamer.

---

## 10. Remaining Critical Fixes

Ordered by impact on the benchmark numbers, which is what matters for the competition:

### Fix 1: TARGET NORMALIZATION (Critical — this is why nothing works)

**The single most important fix.** Without this, all other improvements are wasted.

```python
# In dataset.py, normalize targets the same way you normalize features:
target_cols = ['CompressorHealth', 'CombustorHealth', 'TurbineHealth', 
               'OverallHealth', 'Thrust_N', 'TSFC_g_N_s']

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train_raw)
y_val_scaled = target_scaler.transform(y_val_raw)

# During inference, inverse-transform predictions:
# preds_original_scale = target_scaler.inverse_transform(preds_scaled)
```

Alternatively, normalize per-head: divide Thrust by its training-set mean/std, divide health values by theirs, so all 6 heads contribute roughly equally to the gradient.

**The sigmoid/softplus output activations also need rethinking:** 
- Sigmoid clamps health to [0,1], which is correct in principle but means the model can't predict normalized target values outside [0,1] — if you normalize targets, remove sigmoid
- Softplus guarantees positive thrust/TSFC, but at normalized scale these might need to be negative (below-mean predictions) — remove softplus if normalizing

The cleanest approach: **normalize targets, remove all output activations, let the heads predict freely, and de-normalize at inference time.**

### Fix 2: STOP CHEATING IN THE TELEMETRY STREAMER (Critical)

Lines 131–133 of [telemetry_streamer.py](file:///c:/Users/anant/Downloads/zero%20and%20already%20behind/src/evaluation/telemetry_streamer.py#L131-L133):

```python
# DELETE THESE:
constrained_tsfc = fuel_flow_g / thrust if thrust > 0 else 0.0
pinn_violation = 0.0

# REPLACE WITH:
pinn_violation = calc_violation(tsfc, thrust)
# And in the payload, use `tsfc` not `constrained_tsfc`
```

Also fix line 65: stream from the test split, not `turbojet_complete_dataset.csv`.

### Fix 3: ADD FUEL-FLOW-NORMALIZED WORK FEATURE (High)

The Scientific Proposal explicitly recommended this as the key missing feature. Add to thermodynamics.py:

```python
if all(col in df.columns for col in ['Fuel_Flow']) and 'Net_Specific_Work' in df_phys.columns:
    df_phys['Fuel_Flow_Norm_Work'] = df_phys['Net_Specific_Work'] / (df['Fuel_Flow'].values + 1e-6)
```

### Fix 4: REMOVE REDUNDANT FEATURE (Low)

Drop either `Combustion_Temp_Rise` or `Combustor_Heat_Addition` — they're linearly identical after scaling.

### Fix 5: TRY hidden_dim=64 (Medium)

The model may be underfitting at 32. The training curves never triggered early stopping, consistent with insufficient capacity.

### Fix 6: IMPLEMENT THE DASHBOARD (High — required deliverable)

The `DASHBOARD_BUILD_GUIDE.md` is ready to be handed to a coding agent. The `benchmark_results.json` is already being written. Wire up the dashboard to consume it.

### Fix 7: USE THE HACKATHON'S OWN SPLIT (Medium)

Instead of re-pooling train.csv + test.csv and making your own engine split, use `train.csv` engines for training and `test.csv` engines for testing, as the organizers presumably intended. Verify with:

```python
train_engines = set(pd.read_csv('Dataset/train.csv')['EngineID'].unique())
test_engines = set(pd.read_csv('Dataset/test.csv')['EngineID'].unique())
print(f"Overlap: {train_engines & test_engines}")  # Should be empty
```

### Fix 8: FIX THE system_architecture.md (Quick)

The doc still falsely claims the dataset is "internally generated mock data." This must be corrected before any presentation — it's the first thing a judge reviewing the docs would see.

---

## 11. Revised Verdict & Scorecard

### Score Card — Revision 2

| Aspect | Rev 1 | Rev 2 | Comment |
|--------|-------|-------|---------|
| **Dataset Integrity** | 2/10 | 5/10 | Proper engine-level split exists, but re-pools CSVs instead of using the hackathon's own split; old docs still uncorrected |
| **Physics Integration** | 2/10 | 5/10 | Loss constrains outputs (correct), 11 features (up from 5), but Fuel_Flow still not used, and the constraints have zero measured effect |
| **Model Architecture** | 3/10 | 4/10 | Right-sized but possibly underfitting; no temporal modeling; multi-head scale imbalance is unaddressed |
| **Training Rigor** | 1/10 | 6/10 | 300 epochs, early stopping, weight decay, engine-based split — major improvement. But target normalization is missing, which sabotages everything |
| **Benchmark Honesty** | 0/10 | 8/10 | All metrics now computed from real predictions. Surrogate speed measured. MC-Dropout coverage measured. Results saved to JSON. Only deduction: the streamer still cheats separately |
| **Paper Alignment** | 1/10 | 6/10 | Scientific Proposal grounds terminology correctly (PCMN not PINN), cites Raissi et al. for distinction, correct citations. But implementation doesn't fully follow the proposal |
| **Deliverables Coverage** | 3/10 | 3/10 | No change — dashboard is still the old version, no new panels implemented |
| **Code Quality** | 4/10 | 7/10 | Clean modular structure (dataset.py, train.py, loss.py, benchmark.py). Proper imports. Scripts actually run. Ablation variants are cleanly parameterized |
| **Innovation** | 2/10 | 4/10 | The ablation framework + honest benchmarking is uncommon in hackathon projects. The grey-box framing is correctly positioned. But no novel technique beyond textbook approaches |
| **Dashboard** | 7/10 | 4/10 | Score drops because the dashboard guide is excellent but unimplemented, and the old dashboard is still showing fabricated data from the cheating streamer |
| **OVERALL** | **2.4/10** | **5.2/10** | |

### The Trajectory

The project has moved from **"theatre"** to **"an honest attempt that doesn't work yet."** That's a real and meaningful improvement — honesty is the foundation everything else is built on.

The path from 5/10 to 7/10 requires exactly two things:
1. **Fix the target normalization** so the model actually trains (Fix 1 above)
2. **Implement the dashboard** using the already-written build guide

The path from 7/10 to 8+/10 requires:
3. **Remove the streamer cheat** (Fix 2)
4. **Show that the ablation produces different numbers** — i.e., that the Full Model actually outperforms Baseline-Raw on TSFC violation and health RMSE
5. **Fix the system_architecture.md** provenance claim

### Can It Win Now?

**Not yet, but it could.** The structural honesty is there. The ablation framework is there. The benchmark infrastructure is there. What's missing is a model that actually works — and that's a target normalization fix away from being testable. If the Full Model's TSFC violation drops to, say, <5% while Baseline-Raw stays at 100%, you have a genuinely compelling and defensible result that most hackathon teams will never produce.

### The One-Sentence Verdict (Revised)

> **Revision 2 converted a Potemkin project into an honest-but-broken one: the infrastructure is real, the benchmarking is real, the ablation framework is real — but the model can't learn because thrust losses dwarf everything else, the telemetry streamer is still faking results, and the dashboard rebuild exists only as a specification, not an implementation.**

---

*"Zero and Already Behind" — still behind, but at least now running in the right direction.*
