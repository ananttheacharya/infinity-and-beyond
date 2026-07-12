# AEROTHON 2026 - HAL x IITI: Presentation Deck Guide

**Owner:** Minisha (AI Researcher / Documentation Lead)  
**Objective:** Translate our Physics-Informed Digital Twin (PIDT) architecture into the official Aerothon 2026 slide template. Avoid generic AI buzzwords; focus on thermodynamics, verifiable ML, and aerospace compliance.

---

## Slide 1: Title Slide
*   **Problem Statement Title:** Physics-Informed Digital Twin for Four-Stage Single-Spool Turbojet Engine Health Monitoring
*   **Team Name:** Zero and Already Behind
*   **Participants:** Anant, Minisha, Parth, Shreyansh

## Slide 2: Problem Understanding & Motivation
*   **What to do:** Do NOT just repeat the prompt. Explain *why* HAL needs this specific approach. 
*   **Key Points:** Gas turbines experience complex, nonlinear degradation (fouling, erosion). Traditional flight-hour schedules cause unnecessary downtime, but purely data-driven AI acts as a "black box" that aviation regulators (like DGCA/FAA) reject due to safety risks.
*   **Where to look:** Read the "NASA's early Digital Twin implementations" literature to understand the shift to Condition-Based Maintenance (CBM) via sensor fusion.

## Slide 3: Literature Review / Existing Approaches
*   **What to do:** Compare our approach against open-source benchmarks to prove we did our homework and understand the landscape.
*   **Key Points / Papers to cite:**
    *   *Reference 1:* "Physics-Informed ML for Intelligent Gas Turbine Digital Twins (Farhat et al., 2025)". Use this to explain the 4-layer maturity model and why Physics-Constrained Neural Networks (PcNNs) are the future.
    *   *Reference 2:* Standard C-MAPSS GitHub repos (e.g., `nasim-raj-laskar`). Highlight that they use GRUs and LSTMs directly on raw sensor data, which we classify as a fatal "black box" flaw for this competition.
*   **Where to look:** The `Library` tab on our Dashboard (specifically the "Competitor Benchmarks" and 2025 Review Paper sections).

## Slide 4: Proposed Methodology & Technical Approach
*   **What to do:** Introduce our unique "Physics-Guided Hybrid Model" (The Master Plan). 
*   **Key Points:** We aren't just throwing data at an XGBoost model. We use a **Grey-Box** approach. We derive thermodynamic invariants using first principles, and feed those into a dual-head model (Risk Classifier + RUL Regressor) bounded by physical loss functions.
*   **Where to look:** Open `plan.md` Section 1 and Section 7. Use the "Multi-Layer Digital Twin" architecture breakdown.

## Slide 5: Data Sources & Preprocessing
*   **What to do:** Detail the dataset and our unique feature engineering process.
*   **Key Points:**
    *   *Source:* The provided official training CSVs (Altitude, Mach, Tamb, Pamb, RPM, Fuel Flow, P2-P4, T2-T4). Mention any supplementary datasets Parth procures (e.g., N-CMAPSS) for robustness testing.
    *   *Preprocessing:* Instead of generic Min-Max scaling, we calculate real engineering features: Normalized Shaft Speed ($RPM / \sqrt{Tamb}$), Compressor Pressure Ratios ($P2/Pamb$), and Combustor Temperature Rises.
*   **Where to look:** `plan.md` Section 2 (Physics Before AI) and the "Methodology" tab on the Dashboard.

## Slide 6: Model Architecture / Implementation Plan
*   **What to do:** Show the block diagram of our data pipeline and backend.
*   **Key Points:**
    1.  **Ingestion:** Python backend receiving simulated telemetry via WebSockets.
    2.  **Physics Engine:** Calculates thermodynamic states.
    3.  **ML Inference:** PyTorch PINN checks physics constraints; Monte Carlo (MC) Dropout calculates uncertainty/confidence limits.
    4.  **UI:** Real-time dashboard for mission replay and causal fault graphing.
*   **Where to look:** The Dashboard's "Architecture" tab. Copy the 6-layer interactive accordion layout.

## Slide 7: Expected Outcomes & Evaluation Metrics
*   **What to do:** Map our features directly to the IIT Indore/HAL competition grading rubric to show exactly how we score points.
*   **Key Points:**
    *   *Health Estimation (30%) & Optimization (25%):* Handled by the Dual-Head Hybrid Model.
    *   *Engineering Justification (20%) & Physics Consistency (15%):* Handled by the Thermodynamics Engine and physics-penalized loss function.
    *   *Dashboard & Interpretability (10%):* Handled by our Causal Failure Explanation Graph and Mission Replay.
*   **Where to look:** The Dashboard's "Overview" tab (Evaluation Criteria).

## Slide 8: Team Composition & Contributions
*   **What to do:** Clearly delineate roles to show a highly organized, parallel-executing engineering team.
*   **Key Points:**
    *   **Anant:** Architect, ML Pipeline, Monitoring Framework.
    *   **Minisha:** AI Researcher, Literature Review, Explainability Rule-base.
    *   **Parth & Shreyansh:** UI/UX Engineers, Physics Formulas, Data Procurement.
*   **Where to look:** The "Workstreams" tab on the Dashboard.

## Slide 9: Appendix & References
*   **What to do:** List all papers and open-source repos we analyzed to prove depth of research.
*   **Key Points:** Include the Farhat 2025 paper, PIESRGAN paper, PINN Industrial Gas Turbine trends, and links to the competitor benchmarks on GitHub.
*   **Where to look:** The "Library" tab on the Dashboard.
