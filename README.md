# 🚀 Rocket Valve Data Logger & Anomaly Detector

Simulates high-frequency pressure and temperature telemetry from a liquid
rocket engine feedline during a hot-fire test, detects anomalies (valve
cavitation, structural leaks, temperature spikes) in **real time** using a
causal streaming detector, and visualizes the test live — the trace turns
red the instant a fault is flagged.

![demo](assets/demo.gif)

## Why this exists

Rocket engine test stands live and die by pressure and temperature sensors.
A cavitating valve or a developing leak can go from "fine" to "catastrophic"
in a fraction of a second, so a test conductor needs automated flags, not
just raw charts. This project is a small, self-contained sandbox for
experimenting with that idea end-to-end: simulate realistic sensor noise and
failure modes, build a detector that only ever sees data "as it happens"
(no lookahead), and watch it work live.

## Features

- 🛰️ **Realistic simulated telemetry** — 500 Hz pressure + temperature data with a startup ramp, steady-state noise/drift, and a shutdown ramp.
- 💥 **Three injected fault types** — cavitation (fast, chattery dip), structural leak (slow, accelerating decline), and temperature spike.
- 🧠 **Causal anomaly detection** — a streaming detector that combines a rate-of-change z-score, a slow-adapting baseline with hysteresis, a noise/chatter ratio, and a temperature z-score. It never looks into the future.
- 📊 **Live visualization** — a matplotlib animation that turns the trace red exactly over a flagged fault, with a persistent shaded region and a live status readout.
- ✅ **Ground-truth validation** — the simulator labels every injected fault so you can score detector accuracy (`python detector.py`).

## Quick start

```bash
git clone <your-repo-url>
cd rocket_valve_monitor
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install PyQt6               # GUI backend for the live plot window
python main.py
```

A window opens and plays back a simulated 60-second hot-fire test in real
time. Watch for the trace turning red and the "ANOMALY: ..." readout in the
top-right when a fault is injected.

## Project structure

```
.
├── simulator.py       # Generates the synthetic 500 Hz sensor stream
├── detector.py        # Causal, streaming-safe anomaly detector
├── visualize.py        # Live matplotlib animation
├── main.py             # Entry point
├── requirements.txt
└── assets/
    └── demo.gif
```

## How it works

### 1. Simulation (`simulator.py`)
`generate_hotfire_data()` builds a full pressure + temperature timeline:
a 2s startup ramp, steady-state operation with realistic sensor noise and
slow drift, a 3s shutdown ramp, and 1–3 randomly placed fault events:

| Fault | Behavior |
|---|---|
| **Cavitation** | Fast (~0.2s), noisy, oscillating pressure dip |
| **Leak** | Slow (4–8s), accelerating pressure decline with rising noise that doesn't recover |
| **Temp spike** | Rapid Gaussian-shaped temperature spike |

Every sample also carries a ground-truth label purely for scoring — the
detector never sees it.

### 2. Detection (`detector.py`)
`RollingDetector.update(t, pressure, temperature, phase)` runs once per
sample and only ever uses **past** data — exactly what a real-time system
would have. It combines four signals:

1. **dP/dt z-score** — sudden drops/spikes (cavitation)
2. **Slow-EMA baseline deviation** — frozen while a fault is active, so a
   multi-second leak can't drag its own reference value down with it
3. **Local noise ratio** — cavitation is chattery
4. **Temperature z-score** — rapid temperature spikes

All checks only run during **steady-state** operation — a `phase` tag
(`ramp_up` / `steady` / `ramp_down`) is passed alongside each sample, the way
a real test stand would know from its own valve/ignition sequencing. This
matters because a commanded ramp is a large, *intentional* change that would
otherwise either look like a fault or destabilize a z-score-based detector
(a near-noiseless ramp has almost no natural variance, so any tiny wobble
would register as a huge z-score).

Once a fault fires, it's **latched** with hysteresis until the signal looks
clearly normal for a short hold period, instead of flickering on/off with
noise — closer to how a real fault annunciator behaves.

Run `python detector.py` to validate it against the simulator's ground
truth (detection latency, false-positive rate, etc.).

### 3. Visualization (`visualize.py`)
Uses a `matplotlib.collections.LineCollection` so every tiny segment of the
traces can be colored independently — green in nominal operation, red for
the duration of an active fault — plus a shaded red span left behind over
each fault window and a live status readout.

## Configuration

| Setting | Where | What it does |
|---|---|---|
| `SEED` | `visualize.py` | Fix or randomize the simulated test |
| `PLAYBACK_SPEED`, `FPS` | `visualize.py` | Playback speed / animation refresh rate |
| `DetectorConfig` | `detector.py` | All thresholds, window sizes, and latching behavior |

## Extending this

- Swap the hand-tuned detector for an `IsolationForest` or `OneClassSVM`
  (scikit-learn) trained on windows of nominal data, and compare.
- Feed in real DAQ data instead of the simulator (same `RollingDetector`
  interface).
- Replace matplotlib with a Dash/Plotly app for a browser-based dashboard.
- Log flagged events to a CSV/database for post-test review.

## License

MIT — see [LICENSE](LICENSE).
