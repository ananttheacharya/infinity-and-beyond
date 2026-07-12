# Project Context

## The Problem
We are competing in an aerospace hackathon. The goal is to build an engine digital twin that can predict remaining useful life and system health. The original NASA CMAPSS dataset zip was corrupted, so we generated `turbojet_complete_dataset.csv`, a mock dataset simulating a realistic multi-parameter CMAPSS degradation profile.

## Current State
- **Model**: A 6-headed Physics-Informed Neural Network (PINN) that outputs Compressor Health, Combustor Health, Turbine Health, Overall Health, Thrust, and TSFC.
- **Backend/Frontend**: A real-time Node.js dashboard that plots live MC Dropout uncertainty and physics violations.
- **Theme**: Cyberpunk Pink/Black aesthetic with circular gauges and a live Chart.js plot.
- **Stability**: Added `RequestException` exception handling and UI fallback values (`|| 0`) to prevent Node.js packet drops from crashing the Python telemetry streamer or breaking the UI with `NaN` artifacts.

## How to Run
1. Start the Node Server: `npm start` -> Nav to `http://localhost:3000` (You will see the "System Offline" pink cyberpunk overlay).
2. Start the AI Stream: Open a second terminal and run `python src/evaluation/telemetry_streamer.py`.
3. The dashboard will instantly animate with live data, physics scoring, and uncertainty boundaries.
