# System Architecture: Physics-Informed Digital Twin (PIDT)

This document provides an in-depth breakdown of the architecture we are building for the Aircraft Engine Predictive Maintenance System. The goal is to move away from purely data-driven "black-box" models and embrace a Physics-Informed Neural Network (PINN) that is interpretable, robust, and mathematically constrained by thermodynamics.

## 1. Overall Data Flow

The digital twin operates in a multi-layered pipeline:

```mermaid
graph TD
    A[Raw Sensor Telemetry] --> B[Data Validation]
    B --> C[Thermodynamics Engine]
    C --> D[Physics Features: Efficiencies, PR]
    D --> E[Physics-Informed Neural Network PINN]
    E --> F[Hybrid Output]
    F --> G[RUL Regressor]
    F --> H[Health Risk Classifier]
    G --> I[MC Dropout: Uncertainty Bounds]
    H --> I
    I --> J[Decision Dashboard]
```

## 2. The Components in Excruciating Detail

### A. The Thermodynamics Engine (`src/data_pipeline/thermodynamics.py`)
Standard ML feeds raw Temperatures (T2, T3) and Pressures (P2, P3) directly into a neural network. We intercept this data.
The `ThermodynamicsEngine` calculates physical invariants:
1. **Compressor Pressure Ratio (PR_comp):** $P_2 / P_{amb}$
2. **Isentropic Efficiency:** A measure of how much actual work is done compared to ideal work. It is calculated using the specific heat ratio of air ($\gamma = 1.4$) and the relationship between temperature and pressure ratios.

By feeding these calculated features to the AI, we drastically reduce the complexity the AI needs to learn. It no longer has to discover the laws of physics on its own.

### B. The Physics-Informed Neural Network (`src/models/pinn.py`)
Built in PyTorch, this is a multi-head architecture:
1. **Shared Feature Extractor:** A series of dense layers that learn a common representation from the physical features.
2. **Monte Carlo (MC) Dropout:** During both training *and* inference, we randomly drop 30% of the neurons. By running the same data through the network 50 times (each with different dropped neurons), we get a distribution of answers. The mean is our prediction, and the variance gives us a **Confidence Interval**.
3. **Multi-Head Output:**
   - **Head 1:** Predicts Remaining Useful Life (RUL). Uses a `Softplus` activation to ensure the predicted RUL is always positive.
   - **Head 2:** Predicts categorical health risk (e.g., Healthy, Degrading, Critical).

### C. The Physics-Informed Loss Function (`src/models/loss.py`)
This is the most critical innovation. During backpropagation, the model is penalized based on three criteria:
$$ Total\_Loss = \alpha \cdot MSE_{RUL} + \beta \cdot CrossEntropy_{Risk} + \gamma \cdot Penalty_{Physics} $$

The **Physics Penalty** mathematically forbids the network from predicting future states that violate thermodynamics. For example, if the model's weights start adjusting in a way that implies an Isentropic Efficiency $> 100\%$, the loss explodes, forcing the model back into the realm of physical reality.

## 3. Deployment Architecture (Next Steps)
Once the model is trained in the Jupyter Notebook, it will be exported (e.g., via ONNX) and wrapped in a FastAPI microservice. A separate Python Telemetry Simulator will stream real-time flight cycle data via WebSockets to a Node.js/Vue.js dashboard, creating a true, live Digital Twin.
