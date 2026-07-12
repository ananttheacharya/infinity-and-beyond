# Project Context & Development Memory

**Project Name:** Zero and Already Behind (Digital Twin for Aircraft Engine Maintenance)
**Current Phase:** Execution & Architecture Setup

## Background
The team is participating in the IIT Indore × HAL competition to build a Physics-Informed Digital Twin (PIDT) for a four-stage single-spool turbojet engine. The core differentiator of this project is avoiding "black-box" pure Machine Learning, and instead incorporating first-principles thermodynamics into the ML pipeline.

## Architectural Decisions
1. **Hybrid Architecture:** We are combining the robustness of enterprise MLOps (FastAPI, MLflow, Docker) with a cutting-edge Physics-Informed Neural Network (PINN).
2. **Physics Layer:** Raw sensor data (Altitude, Mach, Tamb, Pamb, RPM, Fuel Flow, P2, T2, P3, T3, P4, T4) is first transformed into thermodynamic invariants (Compressor/Turbine Pressure Ratios, Isentropic Efficiencies).
3. **ML Layer:**
   - **Physics-Informed Loss Function:** The PyTorch model is penalized if it violates thermodynamic laws (e.g., efficiency > 100%).
   - **Multi-Head Output:** 
     - *Risk Classifier:* Immediate health status.
     - *RUL Regressor:* Remaining Useful Life prediction.
   - **Uncertainty Quantification:** Monte Carlo (MC) Dropout is used to provide confidence intervals for predictions.
4. **Explainability & Telemetry Layer:** Predictions are mapped to a causal logic tree (e.g., `P2/Pamb drops` -> `T2/Tamb rises` -> `Compressor Fouling`) to provide readable explanations for HAL engineers. A Python telemetry simulator will stream live data to the Node.js dashboard.

## Current Progress
- Formulated the comprehensive action plan.
- **Dual-Dataset Testing Methodology Defined:** 
  - **Phase 1 (CSV Showdown):** Train competitor models (Project Icarus/GRU, Titan/XGBoost) and our PINN on `train.csv`. Evaluate and compare against `ground_truth.csv`.
  - **Phase 2 (NASA Deep-Tech):** Map the massive `N-CMAPSS_DS03-012.h5` datasets `X_s_dev` (for training) and test against `Y_test`.
- Established the hybrid Deep Learning architecture requirements (PyTorch, MC Dropout).
- Set up the **Mission Control Dashboard** using Node.js and Socket.io to stream real-time physics telemetry.

## Next Steps
- Implement `thermodynamics.py` to calculate explicit physics features.
- Build the base PyTorch module for the Physics-Informed Neural Network in the Jupyter Notebook / Python scripts.
- Develop the telemetry data streaming producer.
