PROJECT: ZERO AND ALREADY BEHIND — REVISION 2
Corrected Scientific Methodology & Proposal
Physics-Constrained Digital Twin for a Four-Stage Single-Spool Turbojet — IIT Indore × HAL, Statement #2

Purpose of this document: This is the working paper behind your proposal PPT. It replaces the previous
system_architecture.md / judge.md with a version that is defensible under direct questioning, grounded in verified
literature, and validated against your actual dataset by hand-calculation below.

What changed since roast.md: Every fabricated result identified in the audit (hardcoded violation rates, post-hoc
TSFC substitution, a physics loss that penalizes inputs instead of outputs, an undefined import that would crash on
execution, a competitor baseline trained on random noise) is corrected here with real code and a real experimental
protocol — not just a diagnosis.

  ✓  GOOD NEWS FIRST: YOUR DATASET IS PHYSICALLY REAL

I hand-computed compressor and turbine isentropic efficiency from two raw rows of your actual
turbojet_complete_dataset.csv (Section 1 below shows the arithmetic). Both land at 78–90%, which is exactly the
realistic range for real turbomachinery. This matters enormously: it means the organizers generated this data from an
actual thermodynamic engine model, not column-independent random noise. Your previous documentation's fear that
the data was “internally generated mock data” was itself wrong — the data is trustworthy. What was fake was the
benchmark reporting on it, not the data itself.

1. Verifying the Data Is Physically Real (worked by hand, not asserted)

Before proposing any model, the first scientific question is whether the dataset actually obeys thermodynamics — if it
doesn't, a physics-informed approach has nothing to constrain against. Using two rows you pasted from
turbojet_complete_dataset.csv (Engine 1, Cycle 1 and Cycle 2):

Quantity

Cycle 1

Cycle 2

Formula used

Tamb / Pamb

240.05 K / 39779 Pa

259.89 K / 58204 Pa

P2 / T2

76228 Pa / 302.37 K

373738 Pa / 462.44 K

PR_comp =
P2/Pamb

T2/Tamb

1.916

1.259

6.42

1.779

given

given

pressure ratio

temperature ratio

η_compressor

0.204 / 0.260 = 78.6%

0.701 / 0.779 = 90.0%

(PR^0.2857−1)/(TR−1)

η_turbine

0.0543/0.0651 = 83.4%

(2nd row not computed here)

(1−TR)/(1−PR^0.2857)

Both efficiencies land in the 78–90% band that real jet-engine compressors and turbines actually operate in.
That is not a coincidence a random-noise generator would produce — uncorrelated random P/T columns would
routinely produce efficiencies outside [0,1] or near-zero/near-infinite. This is your strongest single piece of evidence
for the presentation: you independently verified the organizers' claim that this is “a synthetic but physics-based dataset
generated from a four-stage single-spool turbojet engine model,” rather than just repeating their claim.

A structural discovery that changes your entire training strategy
train.csv (240 rows) + test.csv (60 rows) = 300 rows = the entirety of turbojet_complete_dataset.csv. This is not a
coincidence. The organizers already built you a held-out test split — ground_truth.csv almost certainly carries the
target labels for both, meant to be merged on ['EngineID','Cycle']. Better still: the sample rows you pasted show
train.csv containing Engines 8, 2, 1 while test.csv contains Engines 7, 9, 6 — different engine IDs. If that holds across
the full files (verify with a one-line set-intersection check), the organizers already gave you a leave-engines-out
generalization test, which is exactly what the rubric's “Generalization Capability” (15%) is designed to reward.
Training only on turbojet_complete_dataset.csv as a monolith, as the current code does, throws this away for free.

•  Action: run set(train.EngineID.unique()) & set(test.EngineID.unique()) — if the intersection is
empty, you have organizer-provided cross-engine validation ready to use, no custom split logic needed.

2. What Was Actually Wrong (precise, not just "needs work")

Six concrete, independently verifiable defects across the five files you sent, each with the exact mechanism:

File

Defect

Why it matters

loss.py

benchmark.py

benchmark.py

train_pinn.py

train_baseline.py

thermodynamics.py

Physics penalty reads
physics_features[:,2] — an INPUT,
precomputed from real sensors — not
any model output

Imports generate_mock_data from
train_pinn.py, which no longer exists in
that file (it has load_real_data instead)

The penalty checks whether your sensor data
violates physics (it never will, it's real data),
not whether the model's predictions do.
gamma=5.0 multiplies a number that is
always ≈0. It is a mathematical no-op.

The script cannot run. It would crash on line
1 of execution — the “benchmark results” in
judge.md could not have come from actually
running this file.

"Baseline Thermodynamic Violation
Rate: 42.5%" and "PINN ...: 0.0%" are
print() literals, not computed values

These are the exact numbers judge.md
presents as "live execution" results. There is
no live execution behind them.

All 300 rows used for training, zero held
out; 10 epochs; PINN inputs
unnormalized while baselines get
StandardScaler

No way to measure generalization (15% of
rubric); 10 epochs on 300 rows is ~50
gradient steps — the model barely starts
learning; unequal preprocessing makes any
PINN-vs-baseline comparison invalid in
either direction.

generate_mock_raw_data() is
np.random.uniform(...) for both X and
y; the class is named BaselineLSTM but
contains no LSTM/GRU layer, just 2
Linear+ReLU layers

This "competitor" is not a competitor. It is a
straw man trained on noise, and mislabeled as
a recurrent model it isn't. Any comparison
against it is meaningless regardless of what
number comes out.

Only 5 features; Fuel_Flow — the
single most important performance
measurement — is never used anywhere
in the physics layer

You cannot claim "physics-informed fuel
efficiency modeling" while never reading the
fuel flow column in the physics layer. This is
the easiest fix in the whole list.

The honest framing for your presentation: none of this was malicious — it reads like an LLM (Gemini) generated a
plausible-looking pipeline and a self-congratulatory benchmark without executing and checking either. The fix is not
“work harder,” it's “actually run the numbers,” which is what Sections 3–6 below do.

3. A Terminology Correction That Will Make You Look More Rigorous, Not
Less

Raissi, Perdikaris & Karniadakis (2019, Journal of Computational Physics 378:686–707) define a Physics-Informed
Neural Network as one where the loss function embeds the residual of a governing differential equation — e.g.
penalizing how far the network's output violates a PDE like conservation of momentum or energy, evaluated via
automatic differentiation.

What this project actually builds is a multi-task neural network with algebraic self-consistency constraints:
physics-derived features go in, and the outputs are penalized for violating closed-form algebraic relationships (TSFC =
fuel flow / thrust; overall health as a function of subsystem healths). That is a legitimate, well-precedented “grey-box”
technique — but it is not the same thing as a PDE-residual PINN, and a judge who has read Raissi et al. will notice the
difference immediately if you call it a PINN without qualification.

Recommended fix: call it a "Physics-Constrained Multi-Head Network (PCMN)" or "grey-box digital twin" in the
report, and explicitly cite Raissi et al. (2019) as the canonical PINN definition your architecture deliberately simplifies
from, given the algebraic (not differential) nature of the available physics at this measurement resolution. Stating this
yourself, before a judge finds it, converts a potential “you called this the wrong thing” moment into a demonstration
that you understand the literature precisely.

4. Extended Physics Feature Set (with the honesty caveat roast.md missed)

Keep your 5 existing features (PR_comp, PR_turb, Comp_Isentropic_Efficiency, Combustion_Temp_Rise,
Normalized_RPM — all verified sound above) and add:

New feature

Formula

Honesty note

Turbine Isentropic Efficiency

(1 − T4/T3) / (1 − (P4/P3)^0.2857)

Compressor Specific Work

cp × (T2 − Tamb), cp = 1005 J/(kg·K)

Turbine Specific Work

cp × (T3 − T4)

Net Specific Work

Turbine_Specific_Work −
Compressor_Specific_Work

Combustor Heat Addition
(proxy)

cp × (T3 − T2)

Overall Pressure Ratio (OPR)

P3 / Pamb

Fuel-Flow-Normalized Work

Net_Specific_Work / Fuel_Flow

Direct turbine-erosion indicator;
verified physically plausible above
(83%)

This is J/kg, NOT total work —
rename from roast.md's
"Compressor_Work" since you have
no measured air mass-flow rate to
multiply by

Same caveat — specific work, per
kg of gas

Proxy for available shaft/thrust
energy per unit mass

Same as Combustion_Temp_Rise
scaled by cp — keep both only if
you show they're used differently
downstream

Standard gas-turbine performance
parameter, cheap to add

Finally uses Fuel_Flow in the
physics layer — a genuine
efficiency-like indicator, without
inventing an LHV/fuel-air-ratio
constant you can't verify against this
dataset

Explicitly do NOT add: a full fuel-air-ratio proxy using an assumed LHV constant, or absolute thrust/work equations
requiring air mass flow rate — you do not have a measured mass flow rate column, and inventing one to force a
textbook thrust equation would be exactly the kind of unfounded-assumption problem this rebuild is trying to
eliminate. State this limitation directly in the report; judges scoring “Engineering Justification” reward acknowledged
constraints over invented ones.

5. Corrected Physics-Constrained Loss (constrains outputs, not inputs)

Replaces loss.py entirely. Two real self-consistency constraints, both on the model's own predictions:

class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta_tsfc=2.0, beta_health=1.0):
        super().__init__()
        self.alpha, self.beta_tsfc, self.beta_health = alpha, beta_tsfc, beta_health
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, fuel_flow_g):
        comp_p, comb_p, turb_p, overall_p, thrust_p, tsfc_p = preds
        comp_t, comb_t, turb_t, overall_t, thrust_t, tsfc_t = targets

        mse_total = sum(self.mse(p.squeeze(), t) for p, t in
                         zip(preds, targets))

        # CONSTRAINT 1: TSFC must equal FuelFlow / Thrust  (on OUTPUTS)
        theoretical_tsfc = fuel_flow_g / (thrust_p.squeeze() + 1e-6)
        tsfc_consistency = self.mse(tsfc_p.squeeze(), theoretical_tsfc)

        # CONSTRAINT 2: Overall health must be explained by subsystem
        # health (weights are a documented engineering assumption,
        # not fitted -- state this explicitly in the report)
        expected_overall = (0.40*comp_p.squeeze() + 0.35*turb_p.squeeze()
                            + 0.25*comb_p.squeeze())
        health_consistency = self.mse(overall_p.squeeze(), expected_overall)

        total = (self.alpha*mse_total + self.beta_tsfc*tsfc_consistency
                 + self.beta_health*health_consistency)
        return total, mse_total, tsfc_consistency, health_consistency

Why this is real: both penalty terms are computed from thrust_p and overall_p — the network's own predictions,
gradients flow through them, and the constraint is violated (nonzero) whenever the model hasn't yet learned the
relationship, unlike the old code where the penalty term was always ≈ 0 by construction.

6. Training and Evaluation Protocol

6.1 Split (uses the organizer-provided structure, Section 1)

1.  Merge train.csv + ground_truth.csv on ['EngineID','Cycle'] → training set with real targets.

2.  Merge test.csv + ground_truth.csv the same way → held-out test set, evaluated exactly once at the end.

3.  Fit any StandardScaler / normalization on the training features only, then apply to test — never fit on combined

data.

4.  If GroupKFold is wanted for hyperparameter tuning, fold only within train.csv, grouped by EngineID, so test.csv

stays untouched until final evaluation.

6.2 Model capacity, corrected for N=300

Hidden dimension 128 in a 3-layer MLP for 300 rows is materially over-parameterized (roast.md flagged the training
schedule but not the capacity mismatch). Recommendation: hidden_dim=32–64, add weight decay (1e-4) and early
stopping on a small validation slice carved from train.csv, and train for as many epochs as early stopping allows
(expect 100–300, not 10).

6.3 A fair baseline comparison (fixes the strawman)

Do not compare the physics-constrained network against a model trained on random noise. That is not a
competitor, it's a strawman, and any judge will see through it in one question. The scientifically defensible comparison
is an ablation: same architecture, same features, same split — the only thing that changes is whether the physics-
constraint terms are included.

Model variant

Features used

What it isolates

Baseline-Raw

Raw sensor columns, no physics features

Value of the physics feature layer
itself

Baseline-Physics-Features

Same engineered features as your model,
MSE loss only

Value of the physics-constrained
LOSS specifically

Full Model

Physics features + physics-constrained
loss

Combined effect — your headline
result

This ablation table, filled with real numbers, is worth more to your score than any XGBoost/GRU strawman
comparison — it directly answers “show me that physics constraints actually help,” which is exactly what roast.md's
audit flagged as the missing evidence, and it's something a generic AI-generated hackathon project almost never
includes.

6.4 Fixed benchmark.py

def compute_tsfc_violation(thrust_pred, tsfc_pred, fuel_flow_g):
    theoretical = fuel_flow_g / (thrust_pred + 1e-6)
    return (torch.abs(tsfc_pred - theoretical) / (theoretical + 1e-6)
            * 100).mean().item()

# On the held-out test.csv split only:
for name, preds in [('Baseline-Raw', raw_preds),
                    ('Baseline-PhysFeat', physfeat_preds),
                    ('Full Model', full_preds)]:
    thrust_p, tsfc_p = preds['thrust'], preds['tsfc']
    v = compute_tsfc_violation(thrust_p, tsfc_p, test_fuel_flow_g)

    rmse = torch.sqrt(torch.mean((preds['overall_health']
                                   - test_overall_health_true)**2))
    print(f'{name}: TSFC violation={v:.2f}%  Overall-health RMSE={rmse:.4f}')

No print statements report a number that wasn't just computed on the line above it. This is the entire fix.

7. Uncertainty Quantification — Calibrated, Not Just Implemented

MC Dropout is correctly implemented in pinn.py — keep it as-is structurally. What's missing is calibration evidence:

5.  After held-out test evaluation, compute how often the true value actually falls within the predicted mean ± 1 std

band. For well-calibrated uncertainty, this should be roughly 68% of the time.

6.  Report this calibration percentage explicitly in the technical report and dashboard — a number here (even if it's

55% or 80%, not the textbook 68%) is more credible than an unvalidated “confidence bound” graphic, and directly
answers a judge's likely question about what “Bayesian Uncertainty” claims actually mean with N=300.

7.  Avoid the word “Bayesian” unqualified — MC Dropout is a widely used approximation to Bayesian inference
(Gal & Ghahramani, 2016), not equivalent to it. Say “MC-Dropout-approximated uncertainty” once, then
“confidence band” afterward.

6.5 Surrogate Model Performance & Computational Efficiency (35% of the rubric combined —
previously unaddressed)

Gap being closed: Surrogate Model Performance (20%) and Computational Efficiency (10%) together outweigh
Physics Consistency alone, and the original rebuild had no methodology for either. “Surrogate model” in this challenge
means your network stands in for the expensive path of resolving the four-stage cycle station-by-station (iteratively
evaluating pressure ratios, temperature ratios, and isentropic-efficiency relations at each station in sequence). The
claim that a trained network is faster than that path has to be measured, not asserted.

import time

# Path A: the 'slow' station-by-station route (your own
# ThermodynamicsEngine, called per-row like a naive simulator would)
t0 = time.perf_counter()
for _ in range(N_TRIALS):
    _ = thermo_engine.extract_physics_features(test_df)
slow_path_ms = (time.perf_counter() - t0) / N_TRIALS * 1000

# Path B: the trained surrogate, one forward pass
model.eval()
t0 = time.perf_counter()
for _ in range(N_TRIALS):
    with torch.no_grad():
        _ = model(X_test_tensor)
surrogate_ms = (time.perf_counter() - t0) / N_TRIALS * 1000

speedup = slow_path_ms / surrogate_ms
n_params = sum(p.numel() for p in model.parameters())
print(f'Slow path: {slow_path_ms:.3f} ms | Surrogate: {surrogate_ms:.3f} ms')
print(f'Speedup: {speedup:.1f}x | Parameters: {n_params:,}')

Report three numbers, not adjectives: latency per sample (ms), parameter count, and measured speedup factor. This
single measurement satisfies both rubric lines at once — “fast” becomes “14× faster than recomputing the physics
chain, at 3ms per prediction on CPU” (illustrative — use your own measured numbers). If a GPU is available, report
both CPU and GPU latency, since a HAL reviewer will reasonably ask whether this could run onboard or only in a
ground station.

8. Literature Grounding (verified citations, mapped to what you actually
built)

Reference

What it's used to justify in YOUR report

Raissi, Perdikaris & Karniadakis (2019), J.
Comput. Phys. 378:686–707

Saxena, Goebel, Simon & Eklund (2008),
PHM Society / IEEE, DOI
10.1109/PHM.2008.4711414

Arias Chao, Kulkarni, Goebel & Fink
(2021), N-CMAPSS dataset paper

Canonical PINN definition (PDE residual in the loss) — cited
to justify why your architecture is termed “physics-
constrained,” not “PINN,” given only algebraic (not
differential) physics is available at your sensor resolution

Origin of the C-MAPSS methodology (run-to-failure
trajectories across a fleet, multiple operating conditions) —
cited to justify your group-based, engine-level train/test split
as field-standard practice, not an invented convention

Cited to justify full-life (healthy-to-failure) trajectory framing
and to correctly acknowledge the unused N-CMAPSS file in
your Dataset/ folder as a scoped-out future-work item, not
silently ignored

Arias Chao, Kulkarni, Goebel & Fink
(2022), Reliability Eng. & System Safety
217:107961

“Fusing physics-based and deep learning models for
prognostics” — directly justifies the grey-box (physics-
features-then-ML) architecture pattern as an established,
published approach, not an ad hoc idea

Gal & Ghahramani (2016), ICML —
“Dropout as a Bayesian Approximation”

Correct, precise citation for MC Dropout — use this instead of
the unqualified word “Bayesian”

Gerdes (2019), LTU doctoral thesis (already
in your library)

Justifies interpretable-model-over-black-box framing for the
explainability layer

Grieves & Vickers (2017), in
Transdisciplinary Perspectives on Complex
Systems, Springer, pp.85–113

Queipo, Haftka, Shyy, Goel, Vaidyanathan
& Tucker (2005), Progress in Aerospace
Sciences 41(1):1–28

Forrester & Keane (2009), Progress in
Aerospace Sciences 45(1–3):50–79

The foundational Digital Twin definition itself — linking a
physical system to its virtual equivalent to mitigate
unpredictable behavior. Cite this once, early in the report,
before using the term “Digital Twin” at all; a HAL reviewer
will expect the term traced to its origin, not just used
colloquially

The standard aerospace reference for surrogate-based analysis
and optimization — cite this to justify Section 6.5's
speed/accuracy trade-off framing as established aerospace-
engineering practice, not a machine-learning-only concept
borrowed in from outside the field

Companion/follow-up survey to Queipo et al. — use if the
report needs a second citation on surrogate-model validation
practice (e.g. justifying why speedup and accuracy must both
be reported together, not speedup alone)

Drop entirely: the PIESRGAN reactive-flow GAN paper and the ducted-flame kinematic-model paper flagged in
roast.md — both were auto-scraped by an arXiv keyword search and are about combustion CFD, not engine health
monitoring. Citing them would itself be evidence the literature review wasn't actually done by a person, which is the
opposite of what you want a HAL reviewer to conclude.

9. Dashboard — Made Actually Useful, Not Just Honest

The frontend (Socket.io + Chart.js, per roast.md) is worth keeping. Rewire it to show only real, computed values. Two
panels below are explicitly named in the hackathon guide's deliverables list and were missing from the original rebuild
— they come first because they're required, not optional:

•  [REQUIRED] Engine operating conditions panel: live/selected-cycle values for RPM, fuel flow, altitude,

Mach, and all four station pressures/temperatures (Tamb/Pamb, P2/T2, P3/T3, P4/T4). This is the guide's literal
first bullet under “Digital Twin Dashboard” and the original rebuild jumped straight past it to health scores — a
judge checking the deliverables list against your dashboard will look for this first.

•  [REQUIRED] Predicted thrust panel: its own labeled time-series, not folded silently into another chart — the

guide lists “Predicted thrust” as a distinct required panel (Section 5, “Digital Twin Dashboard”).

•  Four-stage engine schematic: a simple horizontal station diagram (Ambient → Station 2 Compressor Exit →
Station 3 Combustor Exit → Station 4 Turbine Exit) with each stage recolored by its live health score. Cheap to
build, and it mirrors the problem statement's own “four-stage single-spool turbojet” language back at the judges —
a small, deliberate signal that the team read the brief closely rather than building a generic engine dashboard.

•  Per-engine trajectory view: a dropdown over the ~8–10 engines, plotting real health/TSFC/thrust history across

that engine's actual cycles — immediately more informative than a single live-streaming number, and honest about
the small-N regime.

•  Calibration panel: the MC-Dropout coverage percentage from Section 7, shown plainly (e.g. “72% of true values

fell within the predicted 1-std band on held-out engines”) — this is a real number a judge can't dispute.

•  Ablation panel: a small bar chart of the 3-variant comparison from Section 6.3 — this is your single strongest

visual, because it's the one thing a generic hackathon entry never shows.

•  Surrogate-speed panel: the three numbers from Section 6.5 (latency, parameter count, speedup vs. the

recomputed physics path), displayed as plainly as a spec sheet — this is what actually answers the guide's
“Surrogate Model Performance” and “Computational Efficiency” lines, which together outweigh Physics
Consistency alone.

•  Explainability panel: permutation feature importance computed on the actual trained model (cheap even with 300
rows, unlike SHAP's compute cost) mapped to the causal-chain rule table from plan.md's own design — that part
of the original plan was good and doesn't need to change.

•  A visible “Model Card”: sample size (300 rows, ~8–10 engines), known limitations (no measured air mass flow

— Section 4's caveat), and the calibration number — stated on the dashboard itself, not buried in the report. Judges
explicitly scoring “Engineering Justification” and “Physics Consistency” respond well to visible self-awareness of
limitations.

10. Slide-by-Slide Script for the Proposal PPT

Ten slides, each with the one sentence that should anchor it — use this as your speaker-note skeleton:

8.  Title — “Physics-Constrained Digital Twin for Four-Stage Turbojet Health Monitoring.” Team + HAL/IIT Indore

branding.

9.  The Problem, in HAL's language — restate the 6 challenge tasks and 6 weighted rubric criteria directly from the

problem statement, so judges see you're answering their brief, not a generic one.

10. Why naive ML fails here — one slide: raw sensors fed to a black-box regressor can predict a compressor

“efficiency” above 100%; show this as a real failure mode, not a strawman.

11. Data validation slide — THE hand-calculation from Section 1 of this document. Show the two isentropic-

efficiency numbers landing at 78% and 90%. This is your most original, most defensible slide — nobody else will
have independently verified their dataset by hand.

12. Architecture — the 6-layer pipeline (sensor → physics features → subsystem models → fusion → explainability
→ dashboard), correctly labeled “physics-constrained,” with the Raissi et al. terminology distinction stated as a
footnote of rigor, not hidden.

13. The physics constraint, precisely — show the TSFC = FuelFlow/Thrust and Overall = f(subsystems) equations
from Section 5, and say explicitly: “these constrain the model's outputs, not its inputs” — preempting the exact
question that would otherwise sink you.

14. The ablation result — Section 6.3's 3-bar comparison (Raw vs Physics-Features vs Full Model) with real numbers.

This slide is your evidence, not your claim.

15. Uncertainty & calibration — the coverage percentage from Section 7, framed honestly given N=300.

16. Dashboard walkthrough — live or screenshot, emphasizing the Model Card and ablation panel over the generic

health gauges.

17. Roadmap & limitations — explicitly name what's out of scope (full Brayton-cycle energy balance, N-CMAPSS

integration, temporal/GRU modeling) as documented future work, not silently omitted gaps.

11. Closing the Rubric Gap: Surrogate Model Performance (20%) &
Computational Efficiency (10%)

This is the single most consequential gap in Revision 2: 30% of the total rubric (Surrogate Model Performance 20%
+ Computational Efficiency 10%) had no methodology at all in the previous version. Both are satisfied by one
measurement, done once, reported twice.

11.1 What “surrogate” means for THIS problem
The problem statement's own framing (Background section) is explicit: high-fidelity engine simulations are accurate
but too slow for real-time use, and surrogate models exist to approximate that behavior at a fraction of the cost. You
don't have the organizers' internal high-fidelity simulator, but you do have an honest, defensible proxy: your own
ThermodynamicsEngine plus subsystem health models represent the “slow path” if computed iteratively per sample
from raw sensors, versus a single forward pass through the trained network representing the “fast path.” Measuring the
gap between them is a legitimate, literature-grounded surrogate-performance benchmark, not an invented metric.

11.2 The measurement protocol
import time

# Path A: 'slow' — recompute physics features + run subsystem
# logic per sample, one at a time (simulates non-vectorized,
# per-cycle recomputation as a physical test rig would do it)
t0 = time.perf_counter()
for i in range(len(X_test)):
    _ = thermo_engine.extract_physics_features(df_test.iloc[[i]])
slow_path_s = time.perf_counter() - t0

# Path B: 'fast' — trained surrogate, single batched forward pass
t0 = time.perf_counter()
with torch.no_grad():
    _ = model(X_test_tensor)
fast_path_s = time.perf_counter() - t0

speedup = slow_path_s / fast_path_s
per_sample_ms = (fast_path_s / len(X_test)) * 1000
n_params = sum(p.numel() for p in model.parameters())
print(f'Speedup: {speedup:.1f}x | {per_sample_ms:.3f} ms/sample | {n_params} params')

11.3 Report these four numbers — nothing more is needed

Metric

Satisfies

Where it goes

Speedup factor (slow-path / fast-
path)

Per-sample inference latency
(ms)

Surrogate Model Performance (20%)

Headline number, Section 11.2
result

Computational Efficiency (10%)

Dashboard footer + report

Parameter count

Computational Efficiency (10%)

One line, shows deliberate right-
sizing (Section 6.2)

Ablation RMSE deltas (Section
6.3)

Cross-links back to Health Estimation
(30%) and Physics Consistency (15%)

Same experiment run, reused

Grounding citation: Queipo, Haftka, Shyy, Goel, Vaidyanathan & Tucker (2005), Progress in Aerospace Sciences
41(1):1–28 — the foundational aerospace surrogate-modeling survey. Cite this once to justify why “replacing an
expensive high-fidelity evaluation with a fast approximation, validated against the original” is exactly the standard
SBAO (surrogate-based analysis and optimization) methodology, not a term you're using loosely.

12. Dashboard — Two Required Panels That Were Missing

The hackathon guide names 8 specific dashboard panels. Revision 2's Section 9 covered 6 of them well but silently
substituted its own panels for two literal guide requirements. Fixed below — add these to the existing panel list, don't
replace it.

Required panel (from
guide)

What Revision 2 had instead

Fix

“Engine operating conditions”

Nothing — jumped straight to
health/calibration panels

“Predicted thrust”

Folded silently into the ablation panel, no
dedicated view

Add a raw telemetry strip: RPM,
Fuel Flow, Altitude, Mach, and P/T
at all 4 stations, live or per selected
cycle

Add its own time-series panel:
predicted Thrust_N over cycles,
with the MC-Dropout confidence
band from Section 7 shown as a
shaded envelope

12.1 Full corrected dashboard panel list (supersedes Section 9)
•  Raw telemetry strip — operating conditions (NEW, closes guide gap)

•  Per-engine trajectory view (dropdown across ~8–10 engines)

•  Predicted thrust time series with confidence envelope (NEW, closes guide gap)

•  Compressor / Combustor / Turbine / Overall health gauges

•  Degradation trend chart per subsystem

•  Calibration panel (MC-Dropout coverage %, Section 7)

•  Ablation panel (Raw vs Physics-Features vs Full Model, Section 6.3)

•  Surrogate speed panel — the 4 numbers from Section 11.3 (NEW, closes rubric gap, doubles as evidence for 2

criteria)

•  Explainability panel (causal chain + permutation feature importance)

•  Visible Model Card (sample size, limitations, calibration number)

13. The Four-Stage Station Diagram (cheap, and mirrors the guide's own
language)

Both the report and the dashboard should open with a single schematic using the exact station-numbering convention
the aerospace prognostics literature already uses, rather than inventing your own labels:

Station

Physical location

Station 0/1

Ambient / compressor inlet

Your columns

Tamb, Pamb

Station 2

Station 3

Station 4

Compressor exit

Combustor exit / turbine inlet

Turbine exit

T2, P2

T3, P3

T4, P4

Worth stating explicitly in the report: your dataset has no distinct “station 1” (compressor inlet) measurement
separate from ambient conditions — a simplification consistent with a single-spool engine without a separate inlet duct
sensor. Naming this convention explicitly, and citing Saxena, Goebel, Simon & Eklund (2008), whose Figure 1
defines this exact station-numbering scheme for the original CMAPSS model, does two things: it shows you recognize
the standard notation the judges likely already know, and it turns what could look like a missing sensor into a stated,
justified simplification.

A simple rendering for both the slide and the dashboard header: four boxes left to right (Ambient → Compressor →
Combustor → Turbine), each showing its live T/P readout and colored by that subsystem's current health score — this
is the single visual a judge will screenshot to remember your team by.

14. Additional Academic References (verified)

Reference

What it grounds

Queipo, Haftka, Shyy, Goel, Vaidyanathan
& Tucker (2005), Progress in Aerospace
Sciences 41(1):1–28

Foundational surrogate-based analysis and optimization
(SBAO) survey — grounds Section 11's speedup/fast-path-vs-
slow-path methodology in established aerospace practice, not
an invented metric

Farhat & Altarawneh (2025), Energies
18(20):5523, DOI 10.3390/en18205523

“Physics-Informed Machine Learning for Intelligent Gas
Turbine Digital Twins: A Review” — its own classification
explicitly names “physics-constrained neural networks
(PcNNs) with CFD surrogates” as a maturity category, directly
supporting the surrogate-modeling framing without
overclaiming full CFD-level fidelity

Saxena, Goebel, Simon & Eklund (2008),
PHM Society/IEEE, DOI
10.1109/PHM.2008.4711414

Already cited for the train/test split rationale in Section 8 —
now doubles as the source of the station-numbering
convention used in Section 13's diagram

Forrester & Keane (2009), Progress in
Aerospace Sciences 45(1–3):50–79

“Recent advances in surrogate-based optimization” —
optional second citation if a reviewer wants more than one
source on surrogate methodology; covers model-
update/refinement strategies if asked how the surrogate would
be kept current post-deployment

Do not cite: any paper using the word “PINN” in its title without checking it solves a differential equation — several
surrogate-modeling papers use “physics-informed” loosely; citing them uncritically would repeat the exact
terminology imprecision Section 3 corrected.

15. Immediate Next Actions (updated)

18. Run the EngineID intersection check between train.csv and test.csv (Section 1) — 5 minutes, changes your entire

validation story if confirmed.

19. Merge train/test with ground_truth.csv to recover real labels; stop using turbojet_complete_dataset.csv as a

monolithic, unsplit source.

20. Implement Section 5's corrected loss and Section 4's extended features — both are small, mechanical changes to

existing files.

21. Run the Section 6.3 ablation AND the Section 11.2 speed benchmark in the same script run — both reuse the same

trained model and test set.

22. Add the two missing dashboard panels (Section 12) before touching the panels that were already planned —

they're required deliverables, not enhancements.

23. Build the PPT around Section 10's outline once real ablation and speedup numbers exist — do not write the

“results” slide before the numbers do.

