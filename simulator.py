"""
simulator.py
------------
Simulates a high-frequency (default 500 Hz) live sensor stream from a liquid
rocket engine feedline during a 60-second hot-fire test.

Two channels are generated:
    - pressure_psi    : feedline pressure at the valve/engine inlet
    - temperature_k   : propellant temperature at the same station

The simulator produces a realistic profile:
    1. Startup ramp (0 -> nominal over ~2s)
    2. Steady-state operation with realistic sensor noise + slow drift
    3. Shutdown ramp-down at the end of the burn
    4. Randomly injected anomaly events layered on top:
         - "cavitation": a fast, oscillatory pressure dip (valve cavitation)
         - "leak"      : a sustained, gradually worsening pressure decline
                         with increased noise (structural leak)
         - "temp_spike": a rapid temperature spike (e.g. seal/bearing failure,
                         combustion instability)

Ground-truth anomaly labels are included so the detector's performance can be
checked, but the detector itself never uses these labels.
"""

import numpy as np
import pandas as pd


def generate_hotfire_data(
    duration_s: float = 60.0,
    sample_rate_hz: float = 500.0,
    nominal_pressure_psi: float = 300.0,
    nominal_temp_k: float = 300.0,
    n_anomalies: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a full 60s (or arbitrary duration) hot-fire dataset up front.

    Returns a DataFrame with columns:
        time_s, pressure_psi, temperature_k, true_anomaly, true_anomaly_type
    """
    rng = np.random.default_rng(seed)

    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz

    # ---- 1. Nominal pressure profile: startup ramp -> steady -> shutdown ----
    ramp_up_s = 2.0
    ramp_down_s = 3.0

    pressure = np.full(n, nominal_pressure_psi, dtype=float)
    ramp_up_mask = t < ramp_up_s
    pressure[ramp_up_mask] = nominal_pressure_psi * (t[ramp_up_mask] / ramp_up_s)

    ramp_down_start = duration_s - ramp_down_s
    ramp_down_mask = t >= ramp_down_start
    frac = 1.0 - (t[ramp_down_mask] - ramp_down_start) / ramp_down_s
    pressure[ramp_down_mask] = nominal_pressure_psi * np.clip(frac, 0, 1)

    # slow, low-frequency drift (e.g. tank blowdown / thermal effects)
    drift = 3.0 * np.sin(2 * np.pi * 0.05 * t)
    pressure += drift

    # sensor noise (steady-state operating noise)
    pressure += rng.normal(0, 1.2, size=n)

    # ---- 2. Nominal temperature profile ----
    temperature = np.full(n, nominal_temp_k, dtype=float)
    temperature[ramp_up_mask] = nominal_temp_k * 0.9 + 0.1 * nominal_temp_k * (
        t[ramp_up_mask] / ramp_up_s
    )
    temperature += 2.0 * np.sin(2 * np.pi * 0.03 * t + 1.0)
    temperature += rng.normal(0, 0.5, size=n)

    # commanded operating phase -- a real test stand knows this from valve
    # commands, so a detector is allowed to use it to avoid flagging
    # expected startup/shutdown transients as anomalies.
    phase = np.full(n, "steady", dtype=object)
    phase[ramp_up_mask] = "ramp_up"
    phase[ramp_down_mask] = "ramp_down"

    true_anomaly = np.zeros(n, dtype=bool)
    true_anomaly_type = np.array([""] * n, dtype=object)

    # ---- 3. Inject anomaly events ----
    # Keep events inside the steady operating region, non-overlapping.
    safe_start = ramp_up_s + 2.0
    safe_end = ramp_down_start - 2.0
    event_types = rng.choice(
        ["cavitation", "leak", "temp_spike"], size=n_anomalies, replace=(n_anomalies > 3)
    )

    placed_windows = []

    def _pick_start(min_gap_s):
        for _ in range(200):
            start = rng.uniform(safe_start, safe_end - min_gap_s)
            if all(
                start > (w_end + 1.0) or (start + min_gap_s) < (w_start - 1.0)
                for w_start, w_end in placed_windows
            ):
                return start
        return None  # couldn't place, skip

    for etype in event_types:
        if etype == "cavitation":
            dur = rng.uniform(0.15, 0.35)
            start = _pick_start(dur + 0.5)
            if start is None:
                continue
            i0, i1 = int(start * sample_rate_hz), int((start + dur) * sample_rate_hz)
            local_t = np.linspace(0, dur, i1 - i0)
            # sharp dip with decaying ringing (cavitation chatter), partial recovery
            depth = rng.uniform(40, 90)
            ringing = depth * np.exp(-local_t / (dur * 0.3)) * np.cos(2 * np.pi * 25 * local_t)
            pressure[i0:i1] -= depth * 0.6
            pressure[i0:i1] += ringing
            # elevated high-frequency noise during cavitation
            pressure[i0:i1] += rng.normal(0, 4.0, size=i1 - i0)
            true_anomaly[i0:i1] = True
            true_anomaly_type[i0:i1] = "cavitation"
            placed_windows.append((start, start + dur))

        elif etype == "leak":
            dur = rng.uniform(4.0, 8.0)
            start = _pick_start(dur + 0.5)
            if start is None:
                continue
            i0, i1 = int(start * sample_rate_hz), int((start + dur) * sample_rate_hz)
            local_t = np.linspace(0, 1, i1 - i0)
            total_drop = rng.uniform(30, 70)
            decline = total_drop * (1 - np.exp(-3 * local_t))  # accelerating decline
            pressure[i0:i1] -= decline
            pressure[i0:i1] += rng.normal(0, 2.5, size=i1 - i0)  # noisier as it leaks
            # pressure stays low after the leak starts (doesn't recover)
            pressure[i1:] -= decline[-1] * np.exp(-0.05 * (np.arange(n - i1)))
            true_anomaly[i0:i1] = True
            true_anomaly_type[i0:i1] = "leak"
            placed_windows.append((start, start + dur))

        elif etype == "temp_spike":
            dur = rng.uniform(1.0, 2.5)
            start = _pick_start(dur + 0.5)
            if start is None:
                continue
            i0, i1 = int(start * sample_rate_hz), int((start + dur) * sample_rate_hz)
            local_t = np.linspace(0, dur, i1 - i0)
            spike_mag = rng.uniform(25, 60)
            profile = spike_mag * np.exp(-((local_t - dur * 0.3) ** 2) / (2 * (dur * 0.15) ** 2))
            temperature[i0:i1] += profile
            true_anomaly[i0:i1] = True
            true_anomaly_type[i0:i1] = "temp_spike"
            placed_windows.append((start, start + dur))

    # physical floor: pressure can't go negative
    pressure = np.clip(pressure, 0.0, None)

    df = pd.DataFrame(
        {
            "time_s": t,
            "pressure_psi": pressure,
            "temperature_k": temperature,
            "phase": phase,
            "true_anomaly": true_anomaly,
            "true_anomaly_type": true_anomaly_type,
        }
    )
    return df


if __name__ == "__main__":
    data = generate_hotfire_data(seed=42)
    print(data.describe())
    print("\nInjected anomaly samples:", data["true_anomaly"].sum(), "/", len(data))
    print(data.loc[data["true_anomaly"], "true_anomaly_type"].value_counts())
