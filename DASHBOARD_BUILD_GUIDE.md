# Dashboard Build Guide — Turbojet Digital Twin Mission Control

**Read this before writing any dashboard code.** This is a build specification for an AI coding agent (or a human developer) implementing the Digital Twin dashboard for the IIT Indore × HAL turbojet health-monitoring challenge. It defines exactly what to build, how it should look, what data it consumes, and what to explicitly avoid. It is a companion to `Zero_and_Already_Behind_Scientific_Proposal.docx` (Section 9) — this document goes deeper on implementation so a coding agent doesn't have to guess.

---

## 0. What this dashboard is actually for

Not a generic analytics SaaS dashboard with cards and gradients. This is an **instrument panel for a jet engine** — the digital equivalent of what an aerospace test-cell engineer or flight-line technician looks at. Every design decision below is grounded in that: aviation glass-cockpit and avionics multi-function-display (MFD) conventions, not consumer dashboard conventions. If a choice doesn't make sense on a real cockpit display or an engine test-cell monitor, it doesn't belong here.

**Explicitly avoid these three defaults** (common in AI-generated UI, and wrong for this brief specifically):
1. Warm cream background with a serif display face and a terracotta/clay accent — reads as a lifestyle SaaS product, not instrumentation.
2. Near-black background with one bright neon accent color — too minimal for a panel that needs to encode multiple simultaneous status states (nominal / caution / warning) the way real avionics does.
3. Broadsheet/newspaper layout with hairline rules and dense text columns — this is a live telemetry panel, not a report.

---

## 1. Design Token System

### 1.1 Color — aviation caution/warning semantics, not decorative

Real cockpit and engine-monitoring displays use a specific, learned color grammar: green = nominal, amber = caution, red = warning, cyan/white = neutral data. Reuse that grammar deliberately — it is both authentic to the subject and instantly legible to a HAL reviewer who has seen a real EICAS/ECAM display.

| Token | Hex | Use |
|---|---|---|
| `--bg-base` | `#0B0F14` | Page background — near-black graphite, not pure black |
| `--bg-panel` | `#141B23` | Panel/card surface |
| `--bg-panel-raised` | `#1B2430` | Hovered/active panel, modal surfaces |
| `--line-hairline` | `#2A3441` | Panel borders, dividers — thin, never decorative shadows |
| `--text-primary` | `#E8EDF2` | Primary readout text |
| `--text-secondary` | `#8A97A6` | Labels, captions, units |
| `--accent-data` | `#5CC8FF` | Primary data line/accent — trend lines, active selection, the one "electric" color on the page |
| `--status-nominal` | `#3DDC84` | Health ≥ 90%, values within normal operating envelope |
| `--status-caution` | `#F0A93A` | Health 70–90%, values approaching limits |
| `--status-warning` | `#E5484D` | Health < 70%, physics-consistency violation, out-of-bounds reading |
| `--status-unknown` | `#5B6470` | No data / model has not seen this regime |

Do not introduce additional accent colors beyond `--accent-data`. Status color is the only place color carries meaning — everywhere else stays grayscale, so the status colors actually stand out when something needs attention (this mirrors real instrument-panel discipline: color is reserved for things that matter).

### 1.2 Typography

| Role | Face | Use |
|---|---|---|
| Numeric readouts | `IBM Plex Mono` or `JetBrains Mono` | All live numbers — RPM, pressures, temperatures, health percentages. Tabular figures, fixed-width, so digits don't jitter as values update. This is the single most important typographic choice: it's what makes it read as instrumentation instead of a web app. |
| Labels / UI chrome | `IBM Plex Sans` or `Inter` | Panel titles, button text, navigation |
| Panel eyebrows | Same as labels, uppercase, `letter-spacing: 0.08em`, `--text-secondary`, small size (11–12px) | e.g. "COMPRESSOR HEALTH", "STATION 2" — mimics labeled instrument bezels |

Numeric type scale: readouts should be large and unambiguous (28–40px for headline numbers like Overall Health %), with units set smaller and dimmer immediately after (e.g. `937.1` large, `K` small and `--text-secondary`) — exactly how altimeters and EGT gauges set units.

### 1.3 Layout grid

- 12-column grid, `--bg-base` background, panels as `--bg-panel` cards with 1px `--line-hairline` borders and **zero or minimal border-radius** (2–4px, not the rounded-corner SaaS default) — instrument bezels are rectilinear.
- Consistent 16px/24px spacing scale. No panel drop-shadows; use the hairline border and a 1px difference between `--bg-panel` and `--bg-base` to separate surfaces, the way a real console does with bezels, not depth.
- Grid should visually read as a **wall of instruments**, not a dashboard of "cards." Panels are dense and information-rich, not padded-out with whitespace to look minimal.

### 1.4 Motion

- Numeric readouts update by direct value replacement, not counting animations — real gauges don't animate through intermediate values, they show the current reading.
- The one deliberate animated moment: the four-stage schematic (Section 3.3) pulses/glows a station's color when its health crosses a threshold — this is the signature element, spend the animation budget here and nowhere else.
- Respect `prefers-reduced-motion`: disable the schematic glow pulse, keep instant value updates only.

---

## 2. Layout Wireframe

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ZERO AND ALREADY BEHIND — TURBOJET DIGITAL TWIN      [Engine: 07 ▾]     │  <- header, engine selector
├───────────────────────────────┬───────────────────────┬─────────────────┤
│  4-STAGE ENGINE SCHEMATIC      │  OVERALL ENGINE HEALTH │  MODEL CARD     │
│  (signature element,           │  (large readout +      │  N=300, 8-10    │
│   Section 3.3)                 │   confidence band)     │  engines, known │
│                                 │                         │  limitations    │
├───────────────────────────────┴───────────────────────┴─────────────────┤
│  ENGINE OPERATING CONDITIONS  [REQUIRED — guide §5]                      │
│  RPM · Fuel Flow · Altitude · Mach · Tamb/Pamb · P2/T2 · P3/T3 · P4/T4   │
├───────────────────────────────┬───────────────────────────────────────┤
│  SUBSYSTEM HEALTH              │  PREDICTED THRUST  [REQUIRED — guide §5] │
│  Compressor · Combustor ·      │  time-series, own labeled panel          │
│  Turbine (3 gauges + trend)    │                                           │
├───────────────────────────────┼───────────────────────────────────────┤
│  DEGRADATION TREND             │  UNCERTAINTY / CALIBRATION                │
│  per-engine cycle history      │  MC-Dropout coverage %                    │
├───────────────────────────────┴───────────────────────────────────────┤
│  ABLATION RESULT          │  SURROGATE SPEED          │  EXPLAINABILITY   │
│  Raw vs PhysFeat vs Full  │  latency/params/speedup   │  causal chain      │
└─────────────────────────────────────────────────────────────────────────┘
```

Mobile/narrow viewport: collapse to single column, schematic and Overall Health stay pinned at top, remaining panels stack in the same top-to-bottom order.

---

## 3. Panel-by-Panel Specification

For each panel: purpose, exact data contract, visual treatment, and required states (loading / empty / error — every panel needs all three, this is not optional polish).

### 3.1 Engine Operating Conditions `[REQUIRED per hackathon guide §5]`

**Purpose:** raw telemetry as the engine is actually reporting it, before any model touches it. This is the guide's literal first required dashboard item and must be visually distinct from the modeled/predicted panels — put a small "RAW SENSOR DATA" eyebrow on it so it's unambiguous which numbers are measured vs. inferred.

**Data contract:**
```json
{
  "engine_id": 7,
  "cycle": 24,
  "altitude_m": 1439.5,
  "mach": 0.840,
  "tamb_k": 272.49,
  "pamb_pa": 86197.8,
  "rpm_rev_min": 38995.1,
  "fuel_flow_kg_s": 1.316,
  "p2_pa": 252000.8, "t2_k": 394.4,
  "p3_pa": 233889.5, "t3_k": 3149.9,
  "p4_pa": 210171.5, "t4_k": 3075.2
}
```

**Visual treatment:** an 8-cell instrument grid (2 rows × 4), each cell showing one measurement in monospace, unit in dimmed small text, `--text-primary` color (raw data is neutral, not status-colored — it hasn't been evaluated yet).

**States:** loading → skeleton cells with `--line-hairline` shimmer, not spinners. Empty (no engine selected) → "Select an engine to view telemetry" in `--text-secondary`. Error → red hairline border + "Telemetry feed unavailable" (no fake fallback numbers, ever).

### 3.2 Overall Engine Health

**Data contract:** `{ "overall_health": 0.9828, "confidence_std": 0.021 }`

**Visual treatment:** the largest readout on the page — big percentage in monospace, colored by the status thresholds in §1.1, with the confidence band shown as a thin ± range directly beneath it in `--text-secondary` (e.g. `98.3% ± 2.1%`), not a separate chart. This is the one number a HAL reviewer glances at first; it must be unambiguous within half a second.

### 3.3 Four-Stage Engine Schematic (signature element)

**Purpose:** the one thing this dashboard will be remembered by. A horizontal strip: `Ambient → Station 2 (Compressor Exit) → Station 3 (Combustor Exit) → Station 4 (Turbine Exit)` — directly mirroring the problem statement's own four-stage language back at the reader.

**Visual treatment:** four rectangular station nodes connected by thin flow lines (left to right = direction of gas flow). Each node fill color = that subsystem's health status color (§1.1). Each node shows: station label (eyebrow), health %, and the one or two features driving that score (e.g. "PR_comp ↓ · η_c 78%"). On health-threshold crossing, the node does a single slow glow-pulse (respecting reduced-motion) — this is the only recurring animation on the page.

**Data contract:**
```json
{
  "stations": [
    { "name": "Ambient", "health": null, "features": {"tamb_k": 272.5, "pamb_pa": 86197.8} },
    { "name": "Compressor Exit (S2)", "health": 0.94, "features": {"pr_comp": 2.92, "eta_c": 0.82} },
    { "name": "Combustor Exit (S3)", "health": 0.97, "features": {"temp_rise_k": 2755.5} },
    { "name": "Turbine Exit (S4)", "health": 0.91, "features": {"pr_turb": 0.83, "eta_t": 0.79} }
  ]
}
```

### 3.4 Subsystem Health (Compressor / Combustor / Turbine)

**Visual treatment:** three compact radial or linear gauges side by side, status-colored, each with a small sparkline of that subsystem's trend across the selected engine's cycles beneath it. Not three separate full-width charts — these are glanceable gauges, the full trend detail belongs in §3.6.

### 3.5 Predicted Thrust `[REQUIRED per hackathon guide §5]`

**Purpose:** distinct, labeled panel — do not fold into the operating-conditions or health panels. The guide names this as its own deliverable item.

**Data contract:** `{ "cycles": [1,2,3,...], "thrust_n": [21227.1, 39800.4, 47814.9, ...] }`

**Visual treatment:** a single clean line chart, `--accent-data` line color, y-axis in Newtons, x-axis in cycle number for the selected engine. Ground-truth (if available for that engine) plotted as a dimmer secondary line for visual comparison — this quietly doubles as evidence of model accuracy without a separate "accuracy" panel.

### 3.6 Degradation Trend

**Purpose:** per-engine cycle-by-cycle history of overall health, letting a reviewer see the actual monotonic decline pattern your data validation confirmed exists.

**Visual treatment:** line chart, x = cycle, y = health %, with the three status bands shown as faint horizontal shaded zones (green/amber/red) behind the line — like the colored arcs on a real gauge, translated to a time series.

### 3.7 Uncertainty / Calibration

**Purpose:** the one panel that makes the "confidence" claims falsifiable instead of decorative.

**Data contract:** `{ "coverage_1std_pct": 72.4, "n_held_out": 60, "description": "% of true values falling within predicted mean ± 1 std on held-out test engines" }`

**Visual treatment:** one honest number, large, with the description directly beneath it in `--text-secondary`. Do not visualize this as a generic "confidence gauge" — a stated percentage with its definition is more credible than a decorative meter.

### 3.8 Ablation Result

**Data contract:**
```json
{
  "variants": [
    { "name": "Baseline-Raw", "rmse": 0.041, "tsfc_violation_pct": 18.2 },
    { "name": "Baseline-PhysFeatures", "rmse": 0.028, "tsfc_violation_pct": 11.4 },
    { "name": "Full Model", "rmse": 0.015, "tsfc_violation_pct": 0.9 }
  ]
}
```

**Visual treatment:** grouped bar chart, two bars per variant (RMSE and TSFC violation %), status-colored by whether each bar represents an improvement. This is the evidence panel — it should look like a results table a reviewer, not a marketing chart.

### 3.9 Surrogate Speed

**Data contract:** `{ "surrogate_ms": 2.8, "recompute_ms": 41.3, "speedup_x": 14.75, "n_params": 18500 }`

**Visual treatment:** three stat readouts side by side (latency, speedup multiplier, parameter count) — no chart needed, this is a spec-sheet panel, not a trend.

### 3.10 Explainability

**Purpose:** the causal chain from `plan.md`'s original design (kept, it was good): feature moved → physical interpretation → efficiency delta → recommendation.

**Visual treatment:** a horizontal flow of 4–5 connected chips/nodes (similar visual language to §3.3 but simpler), e.g. `PR_comp ↓12%  →  η_c ↓8%  →  Likely: compressor fouling  →  Est. efficiency loss: 3%  →  Recommend: inspect`. Text-forward, not chart-forward — this panel's job is to be read, not glanced at.

---

## 4. Global Data Contract (WebSocket message shape)

If reusing the existing Socket.io telemetry streamer, each tick should emit a single structured payload combining all panel data, so the frontend never has to guess at field names or make multiple round-trips per update:

```json
{
  "engine_id": 7,
  "cycle": 24,
  "timestamp": "2026-07-15T10:32:00Z",
  "raw_telemetry": { /* §3.1 shape */ },
  "predictions": {
    "compressor_health": 0.94,
    "combustor_health": 0.97,
    "turbine_health": 0.91,
    "overall_health": 0.9828,
    "confidence_std": 0.021,
    "thrust_n": 47814.9,
    "tsfc_g_n_s": 0.0159
  },
  "physics_features": { "pr_comp": 2.92, "eta_c": 0.82, "pr_turb": 0.83, "eta_t": 0.79 },
  "explainability": { "top_features": ["pr_comp", "eta_c"], "interpretation": "Likely compressor fouling" }
}
```

Static, load-once-per-session data (ablation results, surrogate speed benchmark, calibration coverage, model card) should come from a separate REST endpoint or a bundled JSON file — these don't change per telemetry tick and don't belong in the streaming payload.

---

## 5. States & Failure Modes (apply to every panel, not just some)

| State | Rule |
|---|---|
| Loading | Skeleton shimmer using `--line-hairline`, never a spinner — spinners read as "waiting for a server," this should read as "instrument warming up" |
| No engine selected | Explicit prompt text, `--text-secondary`, never blank space |
| Data unavailable / feed dropped | Red hairline border + explicit message naming what's missing. **Never substitute a placeholder or last-known-good number without labeling it as stale** — this is the exact failure mode that produced the fabricated results in the earlier version of this project. If a value is stale, label it "STALE — last updated Xs ago." |
| Out-of-distribution input | If the model's own uncertainty spikes beyond a threshold, the affected panel border goes `--status-warning` with a small "OOD — outside training envelope" tag, rather than silently displaying a number the model isn't confident about |

---

## 6. Tech Notes

- Existing stack per prior audit: Socket.io backend + Chart.js frontend. Keep this; it's adequate and doesn't need a framework rewrite this close to deadline.
- Chart.js: use its native line/bar chart types for §3.5/3.6/3.8; the schematic (§3.3) and explainability flow (§3.10) are custom SVG/HTML, not chart-library output — they're diagrams, not data plots.
- If a component library is used for the base grid, keep it invisible — no default card shadows, no rounded-corner defaults; override to match §1.3.

---

## 7. Acceptance Checklist (map back to the hackathon guide before calling this done)

- [ ] Engine operating conditions panel present and visually distinct from modeled panels (guide §5, required)
- [ ] Predicted thrust as its own labeled panel (guide §5, required)
- [ ] Compressor / Combustor / Turbine health all present (guide §5, required)
- [ ] Overall health index present (guide §5, required)
- [ ] Degradation trends present (guide §5, required)
- [ ] Prediction confidence present, and calibrated against real held-out data, not decorative (guide §5, required + Section 7 of the proposal doc)
- [ ] No panel displays a number that wasn't computed from a real model run or real sensor row
- [ ] Every panel has loading / empty / error states implemented, not just the happy path
- [ ] Four-stage schematic present as the signature visual element
- [ ] Ablation and surrogate-speed panels present (these satisfy 35% of the rubric that has no other dashboard evidence)
