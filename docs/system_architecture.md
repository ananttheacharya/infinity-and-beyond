# System Architecture: Physics-Informed Digital Twin (PIDT)

## 1. Executive Summary & The Problem Statement
In traditional machine learning applied to aerospace engineering, purely data-driven "black-box" models (like standard Neural Networks) fail. They find mathematical patterns without understanding physics, often predicting impossible scenarios (e.g., an engine part operating at 150% efficiency). Aerospace engineers do not trust black boxes; they need causal explainability and strict adherence to the laws of thermodynamics.

Our goal is to build a competition-winning, physics-informed, interpretable digital twin for a four-stage single-spool turbojet engine. 
We achieve this through a **Physics-Informed Neural Network (PINN)** that is interpretable, robust, and mathematically constrained by thermodynamics.

## 2. Baseline Comparisons: Why Pure ML Fails
When evaluating competitor benchmark repositories, we saw a clear divide:
1. **Recurrent Neural Networks (RNN/GRU)**: Extremely powerful at time-series predictions. However, they lack physical constraints and treat thermodynamic variables (like Pressure $P2$ and Temperature $T2$) as arbitrary numbers. They regularly predict impossible heat transfers or negative entropy.
2. **Artificial Neural Networks (ANN/MLP) & XGBoost**: Fast and highly accurate in terms of raw Mean Squared Error (MSE), but fail "Engineering Justification." They do not offer true confidence intervals unless explicitly wrapped in complex Bayesian structures.

**Our Advantage**: Our PINN architecture offloads the physics calculation to a deterministic engine before ML, and enforces a Physics Penalty Loss Function to trap the network within reality. It also inherently outputs **Uncertainty Quantification** via Monte Carlo Dropout, a feature the baselines lack.

## 3. Data Provenance (Hackathon vs NASA)
The `turbojet_complete_dataset.csv` we utilize is structurally equivalent to the widely-used **NASA CMAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset. 

Because the provided local NASA zip archive was corrupted during extraction, we are using an internally generated mock dataset (`turbojet_complete_dataset.csv`) that simulates the degradation profile of the CMAPSS engines, but explicitly features the exact 6 parameters demanded by the Hackathon challenge:
- Compressor Health (%)
- Combustor Health (%)
- Turbine Health (%)
- Overall Health (%)
- Thrust (N)
- TSFC (Fuel Efficiency)

## 4. The 6-Layer Pipeline
The digital twin operates in a multi-layered pipeline mimicking actual aerospace data processing:

1. **Sensor Stream & Data Validation Layer**: Ingests raw CSV telemetry. Validates physical bounds.
2. **Physics Layer (Feature Engineering) [CRITICAL]**: Calculates physical invariants using a `ThermodynamicsEngine`.
3. **Subsystem Models Layer**: Maps physics features to individual subsystem health indices using PINN heads.
4. **Health Fusion & Prediction Layer**: Aggregates subsystem states to predict Overall Engine Health, Thrust, and TSFC.
5. **Prediction & Explainability Layer**: Uses MC Dropout for uncertainty bounds.
6. **Decision & Dashboard Layer**: The real-time Node.js/Vue.js mission control dashboard.

## 5. The Thermodynamics Engine (The "Grey Box")
Standard ML feeds raw Temperatures (T2, T3) and Pressures (P2, P3) directly into a neural network. We intercept this data.
The `ThermodynamicsEngine` calculates physical invariants:

1. **Compressor Pressure Ratio ($PR_{comp}$):** 
   $$PR_{comp} = \frac{P_2}{P_{amb}}$$

2. **Isentropic Efficiency ($\eta_c$):** A measure of how much actual work is done compared to ideal work. It is calculated using the specific heat ratio of air ($\gamma = 1.4$) and the relationship between temperature and pressure ratios.
   $$\eta_c = \frac{ \left(\frac{P_2}{P_{amb}}\right)^{\frac{\gamma-1}{\gamma}} - 1 }{ \left(\frac{T_2}{T_{amb}}\right) - 1 }$$

By feeding these calculated features to the AI, we drastically reduce the complexity the AI needs to learn.

## 6. The PINN Model & Physics Penalty
Built in PyTorch, our model is a 6-headed **Physics-Informed Neural Network (PINN)**:
- **Shared Feature Extractor**: A series of dense layers (MLP with ReLU) that learn a common representation from the physical features.
- **Multi-Head Output**: Independent heads for the 6 target parameters. Health parameters use a `Sigmoid` activation (bounded 0-1) and performance parameters use `Softplus` (strictly positive output).
- **Physics-Informed Loss Function**: During backpropagation, the model is penalized if it breaks physical laws:
  $$ \mathcal{L}_{Total} = \alpha \cdot \mathcal{L}_{MSE} + \gamma \cdot \mathcal{L}_{Physics} $$
  Where the Physics Penalty is defined by bounding $\eta_c$ strictly within valid limits:
  $$ \mathcal{L}_{Physics} = \sum_{i} \text{ReLU}(\eta_c^{(i)} - 1.0)^2 $$
  This massive gradient penalty forces the optimizer to step away from the unphysical region.

## 7. Dashboard & Explainability
### Monte Carlo (MC) Dropout
A major requirement is **Uncertainty Quantification**. How confident is the AI? We use Monte Carlo (MC) Dropout. During inference, we randomly drop neurons and run the same data point $N=10$ times. By seeing how much the answers vary, we output a standard deviation (`std`) forming the **Confidence Bound** visible on the telemetry chart. If the engine enters an Out-Of-Distribution (OOD) failure state that the model has never seen, the variance spikes, creating a massive visual "Error Bound".

### Physics Consistency Guardrails
The dashboard compares the theoretical engine isentropic efficiency to the live sensor feed. If the physics consistency score drops below `80%`, the dashboard immediately flags a catastrophic thermodynamic violation (alerting the user via UI pulsing and a red font).
