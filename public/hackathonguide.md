Hybrid-Electric Propulsion
Optimization
for a Fixed-Wing UAV

Problem Statement

Indian Institute of Technology Indore

In collaboration with

Hindustan Aeronautics Limited (HAL)

1. Background

Hybrid-electric propulsion is an emerging aerospace technology that combines conventional fuel-
powered gas turbine engines, Wankel engines, or Internal Combustion (IC) engines with electric
propulsion systems to improve endurance, efficiency, and operational flexibility.

This challenge focuses on the conceptual design and optimization of a hybrid-electric propul-
sion architecture for a fixed-wing UAV mission, preferably employing a turboshaft engine with
a rated power of approximately 60 kW.

2. Objective

Design and optimize a hybrid-electric propulsion system for the UAV configuration defined
below.

Teams must develop a methodology/framework to:

• Size the propulsion system (preferably a gas turbine engine such as a turbojet or tur-

boshaft).

• Estimate mission feasibility.

• Optimize the overall system performance.

1 of 5

3. Baseline UAV Specifications

Parameter

Value

Take-Off

UAV Type
Maximum
(MTOW)
Payload Capacity
Cruise Speed
Cruise Altitude

Weight

Fixed-Wing UAV
∼1000 kg

∼200 kg
∼250 km/h
3–10 km

4. Design Variables

Participants may optimize:

• Gas Turbine Engine (Turboshaft Engine) size

• Generator architecture (AC/DC)

• Battery chemistry and capacity

• Number of propulsion motors

• Power-sharing strategy between thermal and electric systems

5. Mission Profile

The UAV mission consists of:

• Take-off

• Climb

• Cruise

• Loiter

• Descent and Landing

The primary objective is to maximize endurance while satisfying mission and system con-

straints.

6. Constraints

The final design must:

• Remain within MTOW limits.

• Carry the specified payload.

2 of 5

• Complete all mission phases.

• Satisfy propulsion power requirements.

• Maintain safe battery operating limits.

7. Assumptions

Participants may use reasonable engineering assumptions and publicly available data sources
for:

• Propulsion efficiencies

• Battery characteristics

• Fuel consumption

• Aerodynamic estimates

• Component sizing

All assumptions must be clearly documented and justified.

8. Deliverables

The final submission should include the following.

8.1 Technical Report

• Problem formulation and design methodology

• Propulsion system architecture

• Component sizing methodology

• Mission analysis and power management strategy

• Optimization approach and objective functions

• Design assumptions and engineering justification

• Performance evaluation and trade-off analysis

8.2 Source Code

Complete implementation of the proposed optimization framework, including simulation models,
sizing algorithms, optimization routines, and any supporting scripts used for analysis.

3 of 5

8.3 Simulation Dashboard

Interactive visualization demonstrating

• Mission profile and flight phases

• Power distribution between thermal and electric propulsion systems

• Battery State of Charge (SoC)

• Fuel consumption throughout the mission

• Engine and motor operating conditions

• System efficiency

• Endurance estimation

• Optimization results and design trade-offs

8.4 Presentation

A concise summary explaining

• Propulsion architecture selection

• Optimization strategy

• Engineering rationale

• Key design trade-offs

• Final optimized configuration

• Mission performance and endurance improvements

9. Evaluation Criteria

Criteria

Mission Feasibility
Optimization Quality
Engineering Justification
Innovation
Endurance Improvement
Presentation & Visualization

Weightage (%)

20
25
20
15
10
10

4 of 5

10. Notes

1. Detailed CFD, structural analysis, and thermal simulations are not required.

2. The focus shall be on system-level design space exploration and propulsion architecture

optimization.

Indian Institute of Technology Indore
In collaboration with
Hindustan Aeronautics Limited (HAL)

5 of 5

Physics-Informed Digital Twin

for Real-Time Four-Stage Turbojet
Health Monitoring

Problem Statement

Indian Institute of Technology Indore

In collaboration with

Hindustan Aeronautics Limited (HAL)

1. Background

Modern aerospace propulsion systems operate under demanding thermal, mechanical, and aero-
dynamic conditions. Throughout their operational life, engine components experience gradual
degradation due to phenomena such as compressor fouling, turbine erosion, and combustor ef-
ficiency loss. These degradation mechanisms affect engine performance, fuel efficiency, thrust
generation, and operational reliability.

Traditional engine monitoring approaches rely on direct sensor measurements and periodic
inspections. However, many critical engine states and component health indicators cannot be
measured directly during operation. Consequently, aerospace industries are increasingly adopt-
ing Digital Twin technologies that combine sensor data, engineering knowledge, and intelligent
models to estimate hidden engine states and predict future performance.

High-fidelity engine simulations provide valuable insight into engine behavior but are compu-
tationally expensive and unsuitable for real-time applications. Surrogate models offer a practical
alternative by approximating complex engine behavior with significantly lower computational
cost while enabling continuous health assessment and monitoring.

2. Objective

Develop a Physics-Informed Digital Twin capable of reconstructing the operational and health
state of a single-spool four-stage turbojet engine using limited sensor measurements.

1 of 5

The proposed framework should estimate hidden subsystem health indicators, predict en-
gine performance, and maintain a continuously updated virtual representation of the engine
throughout its operational life.

Participants are expected to combine engineering principles, data-driven methods, and sur-
rogate modeling techniques to develop an interpretable and computationally efficient solution.

3. Dataset Description

3.1 Official Dataset Repository

The complete dataset, along with supporting files and documentation, can be accessed using
the following Google Drive repository:

Official Dataset

A synthetic but physics-based dataset generated from a four-stage single-spool turbojet

engine model is provided.

The dataset includes multiple virtual engines operating under varying flight conditions and

progressive degradation scenarios.

3.2 Available Measurements

Parameter

Engine ID
Cycle
Altitude
Mach Number
Ambient Temperature (Tamb)
Ambient Pressure (Pamb)
Shaft Speed (RPM)
Fuel Flow Rate
Compressor Exit Pressure (P2)
Compressor Exit Temperature (T2)
Combustor Exit Pressure (P3)
Turbine Inlet Temperature (T3)
Turbine Exit Pressure (P4)
Turbine Exit Temperature (T4)

Unit

–
–
m
–
K
Pa
rev/min
kg/s
Pa
K
Pa
K
Pa
K

The dataset represents engine operation over multiple degradation cycles and varying flight

conditions.

4. Challenge Tasks

4.1 1. Engine Digital Twin Construction

Create a digital representation of the turbojet engine capable of continuously estimating its
operational state from available sensor measurements.

2 of 5

4.2 2. Subsystem Health Estimation

Estimate the health state of the following engine subsystems:

• Compressor

• Combustor

• Turbine

Participants should justify the methodology used for health estimation and demonstrate

engineering interpretability.

4.3 3. Overall Engine Health Assessment

Develop a unified engine health indicator capable of representing the overall condition of the
propulsion system.

4.4 4. Surrogate Modeling

Develop computationally efficient surrogate models that can approximate engine behavior and
subsystem states significantly faster than conventional simulation-based approaches.

4.5 5. Performance Prediction

Estimate critical performance parameters such as

• Engine thrust

• Fuel efficiency metrics

• Degradation trajectory

using available measurements and inferred engine states.

4.6 6. Uncertainty Quantification

Provide confidence estimates or uncertainty bounds associated with predictions wherever appli-
cable.

5. Deliverables

The final submission should include the following.

Technical Report

• Methodology

• Feature engineering strategy

• Physics integration approach

• Model architecture

• Validation methodology

3 of 5

Source Code

Complete implementation of the proposed framework.

Digital Twin Dashboard

Interactive visualization demonstrating

• Engine operating conditions

• Compressor health

• Combustor health

• Turbine health

• Overall health index

• Predicted thrust

• Degradation trends

• Prediction confidence

Presentation

A concise summary explaining

• Engineering rationale

• Surrogate modeling strategy

• Health estimation methodology

• Key results and insights

6. Evaluation Criteria

Criterion

Health Estimation Accuracy
Surrogate Model Performance
Physics Consistency
Generalization Capability
Computational Efficiency
Dashboard and Interpretability

Weight

30%
20%
15%
15%
10%
10%

7. Expected Impact

The challenge aims to encourage the development of next-generation aerospace Digital Twin
technologies capable of real-time engine monitoring, health assessment, and performance pre-
diction.

4 of 5

Successful solutions may contribute toward predictive maintenance, intelligent propulsion

diagnostics, and future autonomous aerospace systems.

Indian Institute of Technology Indore

In collaboration with

Hindustan Aeronautics Limited (HAL)

5 of 5

