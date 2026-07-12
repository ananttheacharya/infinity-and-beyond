# System Architecture & Benchmark Guardrails

## 1. Data Provenance & Structure (Hackathon vs NASA)
The `turbojet_complete_dataset.csv` we utilize is structurally equivalent to the widely-used **NASA CMAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset. 

Because the provided local NASA zip archive (`nasa-cmapss-2-engine-degradation - Copy.zip`) was corrupted during extraction, we are using an internally generated mock dataset (`turbojet_complete_dataset.csv`) that simulates the degradation profile of the CMAPSS engines, but explicitly features the 6 parameters demanded by the Hackathon challenge:
- Compressor Health (%)
- Combustor Health (%)
- Turbine Health (%)
- Overall Health (%)
- Thrust (N)
- TSFC (Fuel Efficiency)

By training on this structured data, the Neural Network correctly learns the steady-state cruise phase (100% overall health, ~45-55kN thrust) and exactly models the catastrophic end-of-life failures (where thrust dips to ~20kN and physics consistency collapses).

## 2. The Multi-Task PINN Architecture
The **Physics-Informed Neural Network (PINN)** operates on a unified deep learning backbone.
- **Inputs**: Extracted Thermodynamic Invariants (Pressure Ratios, RPMs).
- **Backbone**: A 3-Layer fully connected Multi-Layer Perceptron (MLP) with ReLU activations.
- **Heads**: The network splits into 6 independent heads. Health parameters use a `Sigmoid` activation (bounded 0-1) and performance parameters use `Softplus` (strictly positive output).
- **Physics Loss Penalty**: The total Loss combines standard Mean Squared Error (MSE) across the 6 targets with an added penalty term derived from Isentropic Efficiency violations.

## 3. Thermodynamic Formulations & Physics Scoring
Raw sensor feeds (`Tamb`, `Pamb`, `T2`, `P2`, `RPM`) are converted into invariants via the `ThermodynamicsEngine` prior to inference. 

**Isentropic Efficiency ($\eta_c$)**:
$$ \eta_c = \frac{(P_{out}/P_{in})^{\frac{\gamma - 1}{\gamma}} - 1}{(T_{out}/T_{in}) - 1} $$
Where $\gamma = 1.4$ for standard air.

**Physics Consistency Guardrail**: 
The dashboard compares the theoretical engine isentropic efficiency to the live sensor feed. If the physics consistency score drops below `80%`, the dashboard immediately flags a catastrophic thermodynamic violation (alerting the user via UI pulsing and a red font).

## 4. Uncertainty & Error Predictions (MC Dropout)
We evaluate model confidence in real-time using **Monte Carlo (MC) Dropout**.
Instead of outputting a single point estimate, the PINN runs the same telemetry packet through the network `N=10` times with neurons randomly dropped out. 
The standard deviation (`std`) across these 10 predictions forms the **Confidence Bound** visible on the telemetry chart. If the engine enters an Out-Of-Distribution (OOD) failure state that the model has never seen, the variance spikes, creating a massive visual "Error Bound" to alert the engineer.
