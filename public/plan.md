# Action Plan: Physics-Informed Digital Twin for Turbojet Health Monitoring
**Project Name:** Zero and Already Behind  
**Objective:** Build a competition-winning, physics-informed, interpretable digital twin for a four-stage single-spool turbojet engine.

> **Executive Summary**
> This document outlines the comprehensive strategy for developing a Physics-Informed Digital Twin (PIDT) for the IIT Indore × HAL competition. Diverging from purely data-driven "black-box" models, our solution embeds first-principles thermodynamics into the machine learning pipeline to yield an interpretable, robust, and generalizable virtual engine. The architecture features a multi-layer design (Sensor → Physics → ML → Explainability → Dashboard) that predicts subsystem health, thrust, and degradation while providing causal explanations for its predictions. Given the aggressive 3-day timeline, this plan relies on strict modularity, parallel execution, and a clear division of labor that maximizes the interdisciplinary strengths of our team (AI, Systems Engineering, and Electrical Engineering).

---

## 1. System Architecture: The Multi-Layer Digital Twin

We will build the software as a virtual engine, where data flows through hierarchical layers mimicking actual aerospace data processing. 

1.  **Sensor Stream & Data Validation Layer:** Ingests raw CSV data (Altitude, Mach, Tamb, Pamb, RPM, Fuel Flow, P2, T2, P3, T3, P4, T4). Validates physical bounds (e.g., pressures > 0, temperatures within operational limits).
2.  **Physics Layer (Feature Engineering):** Calculates thermodynamic indicators and physically meaningful features rather than feeding raw data to the ML.
3.  **Subsystem Models Layer:** Three distinct, lightweight surrogate models for the Compressor, Combustor, and Turbine, mapping physics features to subsystem health indices.
4.  **Health Fusion & Digital Twin State Layer:** Aggregates subsystem states into an Overall Engine Health index and maintains a continuous virtual representation.
5.  **Prediction & Explainability Layer:** Estimates performance (Thrust, Degradation trajectory). Uses SHAP/LIME combined with a deterministic logic tree to generate human-readable causal chains (e.g., *Pressure ratio decreased → Temperature increased → Compressor fouling → 3% efficiency loss*).
6.  **Decision & Dashboard Layer:** A mission-control style interface providing interactive visualization, mission replay, and maintenance recommendations.

---

## 2. Physics Before AI: Engineering Feature Derivation

Instead of standard scaling, we will compute thermodynamic quantities. **Parth and Shreyansh** will own the mathematical formulation of these features.

*   **Compressor Pressure Ratio (PR_comp):** $P2 / Pamb$
*   **Turbine Pressure Ratio (PR_turb):** $P4 / P3$
*   **Compressor Isentropic Efficiency Index:** Function of $(P2/Pamb)$ and $(T2/Tamb)$
*   **Turbine Isentropic Efficiency Index:** Function of $(P4/P3)$ and $(T4/T3)$
*   **Combustion Temperature Rise:** $T3 - T2$
*   **Normalized Shaft Speed:** $RPM / \sqrt{Tamb}$ (Corrected speed for aerodynamic similarity)
*   **Fuel-to-Air Ratio Proxy:** Derived from Fuel Flow, Pamb, Tamb, and RPM.

*Why this is better:* ML models struggle to learn complex non-linear thermodynamic invariants from small datasets. By providing these explicitly, the surrogate models become incredibly efficient and physically constrained, satisfying the "computational efficiency" and "physics consistency" evaluation criteria.

---

## 3. Work Distribution & Execution Plan (3 Days)

To maximize our 72 hours, we will divide the work based on core competencies.

### **Anant (Tech Lead / Software Architect)**
*   **Role:** ML Pipeline, Digital Twin Architecture, Framework for Monitoring, and Benchmarks.
*   **Action Items:**
    *   Set up the Python data pipeline and hybrid (PINN + XGBoost) backend architecture.
    *   Create the framework for real-time monitoring and anomaly detection.
    *   Define explicit test cases and benchmarks to rigorously validate the physics consistency of our model.
*   **Deliverable:** Working inference pipeline, trained surrogate models, and a robust benchmark suite.

### **Minisha (AI Researcher / Documentation Lead)**
*   **Role:** Literature Review, Executive Summaries, Explainability framework.
*   **Action Items:**
    *   Read the target literature (including the latest 2025 reviews on hybrid AI for gas turbines) and extract executive summaries.
    *   Define the causal chain rules for the Explainability Layer (e.g., mapping feature deviations to specific physical faults like fouling or erosion).
    *   Write the comprehensive Technical Report (Methodology, Engineering Justification, Architecture).
*   **Deliverable:** Rule-base for explainability, completed Technical Report, and Pitch Deck.

### **Parth Sharma & Shreyansh Mangal (Systems / UI Engineers)**
*   **Role:** Dashboard UI/UX, Data Procurement, Physics Modeling.
*   **Action Items:**
    *   Design and build the "Mission Control" Dashboard (Vue/React or Vanilla SPA).
    *   Find and procure additional datasets (e.g., N-CMAPSS or specific test rig data) to augment our training pipeline.
    *   Research and write the exact mathematical formulas for the thermodynamic features.
*   **Deliverable:** Fully functional interactive Dashboard UI, augmented datasets, and validated physics transformation code.

---

## 4. Literature Review Strategy (For Minisha)

Minisha must quickly review the following concepts to ground our report in accepted science:

1.  **"Physics-Informed Machine Learning for Intelligent Gas Turbine Digital Twins: A Review" (Farhat et al., 2025)**
    *   *Key Bit to Extract:* The 4-layer maturity framework (Physics Backbone -> AI Modeling -> Robustness & Uncertainty -> Optimization). This proves our PINN + Hybrid architecture is cutting-edge and surpasses entry-level ANN models. It highlights the necessity of bounding generative models with thermodynamic constraints.
2.  **"Physics-Informed Neural Networks (PINNs) for Gas Turbine Diagnostics"**
    *   *Key Bit to Extract:* How embedding thermodynamic equations (mass, energy balance) into the loss function prevents physically impossible states.
3.  **"Thermodynamic modeling of compressor fouling and turbine erosion"**
    *   *Key Bit to Extract:* The specific signature of fouling vs. turbine erosion.

---

## 5. Innovation Opportunities

To score maximum points in the "Innovation (15%)" and "Presentation (10%)" criteria, we will implement:

1.  **Causal Failure Explanation Graph:** The dashboard will not just show "Compressor Health: 70%". It will display a dynamic node-graph: `[RPM Constant] -> [P2/Pamb drops] -> [T2/Tamb rises] -> [Diagnosed: Compressor Fouling]`. 
2.  **Mission Replay Engine:** A slider on the dashboard that lets HAL engineers "scrub" through a flight cycle and watch the thermodynamic stresses propagate through the engine schematic in real-time.
3.  **Confidence-Aware Predictions:** Utilizing Bayesian approaches (like MC Dropout) to output a "Confidence Score" alongside every health prediction.

---

## 6. Why This Plan Wins (Self-Critique)

**Is this something any prompt can generate?** 
No. A generic prompt generates an XGBoost script that treats `T3_K` and `P4_Pa` as arbitrary numerical features. This plan treats them as thermodynamic realities. 

**Will your generated plan actually help the team or is it just a summary?**
This plan is an actionable, modular blueprint. It prevents the team from falling into the "black-box" trap that disqualifies most hackathon entries. By strictly defining the interface between the Physics Layer and the ML Layer, it allows Anant to build the ML backend in parallel while Parth/Shreyansh code the thermodynamic invariants, directly accelerating execution.

**Will the dashboard actually provide insights or is it just a novelty?**
The dashboard is *not* a novelty—it is a functional "Virtual Sensor" and "Decision Intelligence" interface. While standard dashboards just show static graphs, ours translates abstract AI outputs (like MC Dropout uncertainty bounds and hybrid classifier predictions) into localized component degradation warnings. It gives operators actionable insights: exactly *what* is failing, *why* it's failing (causal chain), and *how confident* the AI is about the diagnosis.

**Why is it better than the competition?**
Most teams will optimize purely for Mean Squared Error (MSE) on the provided dataset. They will fail the "Engineering Justification" (20%) and "Physics Consistency" (15%) criteria. By offloading the physics to Parth and Shreyansh, we guarantee that the inputs to Anant's ML models are already physically meaningful. By having Minisha focus on the causal rule-base, we guarantee that the output is interpretable to a HAL engineer. This is an architecture built for aerospace compliance, not a Kaggle leaderboard.

---

## 7. Open-Source Benchmark Analysis & The Master Plan (v3.0)

To guarantee our winning edge, we conducted a massive architectural review of five leading open-source predictive maintenance repositories (including NASA C-MAPSS winners and MathWorks Digital Twin projects). 

### The Competitive Landscape
1.  **The Purely Data-Driven Repositories:** Several top projects (e.g., *nasim-raj-laskar*, *dattatejaofficial*) built incredibly robust, production-ready MLOps pipelines (Docker, MLflow, FastAPI, S3, Redis). **Their Weakness:** They use standard Deep Learning (GRUs and LSTMs) directly on raw sensor data. For our competition, this is a massive penalty.
2.  **The Simulation & IoT Repositories:** Projects like *Ocramnaig94* use Mosquitto MQTT and Node-RED for real-time agent communication, while others rely heavily on MATLAB/Simulink for injecting physical faults. **Their Weakness:** Very heavy on simulation tools which can be clunky for a modern web-based digital twin dashboard.

### What We Are Stealing (The Best Parts)
*   **Enterprise MLOps Architecture:** We will adopt their data pipelines and use **Monte Carlo (MC) Dropout** for uncertainty quantification (providing a confidence % for predictions—critical for aerospace).
*   **The Hybrid Model Approach:** We will use a classifier for immediate fault risk assessment combined with a regressor for Remaining Useful Life (RUL) estimation.
*   **Real-Time Telemetry Producer:** We will build a Python script that streams flight data cycle-by-cycle via WebSockets to our dashboard, simulating a live engine test cell.

### The Master Plan: Our Differentiating Factor (The PINN)
We will combine the enterprise MLOps architecture of the top repositories with a custom **Physics-Informed Neural Network (PINN)**. 

1.  **The Physics-Guided Data Pipeline (The "Grey-Box"):** Instead of feeding raw temperatures and pressures into an LSTM, we will first pass the data through our Thermodynamics Engine to calculate unmeasured states (Isentropic Efficiency, Enthalpy drops).
2.  **Physics-Informed Loss Function:** When training our PyTorch model, the loss function will penalize the model if its predictions violate the laws of thermodynamics (e.g., if predicted turbine exit temperature implies an impossible efficiency > 100%).
    *   `Total Loss = α * MSE(RUL) + β * CrossEntropy(Health) + γ * Physics_Violation_Penalty`
3.  **The Streaming Digital Twin:** Our Node.js backend will ingest live telemetry via Socket.io. The backend will run incoming data through the exported PyTorch model (via ONNX runtime) and broadcast the predictions to our web dashboard in real-time.
