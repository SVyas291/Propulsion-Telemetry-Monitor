"""
detector.py
-----------
A causal (real-time-safe) anomaly detector for rocket valve/engine sensor
streams. "Causal" means every computation at time t only uses data strictly
BEFORE t (the current sample is always compared against history that does
NOT include itself) -- exactly what you'd have available on a real test
stand, and it avoids the current data point diluting its own "normal"
statistics.

Three complementary signals are combined:

1. Rate-of-change z-score (dP/dt)
   Catches SUDDEN drops/spikes -- the signature of valve cavitation or a
   structural failure. The point-to-point derivative is compared against
   the *recent history* of derivatives (not including itself).

2. Slow-EMA baseline deviation z-score
   A slow exponential moving average (time constant ~3s) tracks the
   "expected" nominal pressure. Critically, the EMA is FROZEN (not updated)
   while an anomaly is active, so a multi-second leak can't drag its own
   reference value down with it -- the baseline keeps comparing against the
   pressure from before the leak started, which is what makes a sustained,
   gradual decline detectable.

3. Rolling noise (std) ratio
   Cavitation is noisy/chattery. A jump in local standard deviation versus
   its recent history is used as a supporting signal for classification.

4. Temperature z-score
   Same "exclude current sample from its own baseline" logic, applied to
   the temperature channel, to catch rapid spikes.

The four signals are combined into a single `is_anomaly` flag with a simple
classification heuristic (cavitation / leak / temp_spike).
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DetectorConfig:
    sample_rate_hz: float = 500.0

    # rate-of-change (dP/dt) detector
    diff_window_s: float = 0.4          # window of past dP/dt values used for stats
    diff_z_threshold: float = 5.0       # z-score threshold to flag a spike/dip

    # slow-EMA baseline detector (catches slow leaks)
    ema_time_constant_s: float = 3.0    # how slowly the "nominal" baseline adapts
    baseline_std_window_s: float = 2.0  # window used to estimate normal-operation std
    baseline_z_threshold: float = 6.0

    # noise / chatter detector (supports cavitation classification)
    noise_window_s: float = 0.3
    noise_ratio_threshold: float = 2.5

    # temperature spike detector (z-score on the temperature channel)
    temp_window_s: float = 1.0
    temp_z_threshold: float = 5.0

    # alarm latching (hysteresis): once an anomaly fires, keep it latched
    # until conditions look clearly normal for `recovery_hold_s`, instead of
    # flickering on/off sample-to-sample with noise. Mirrors how real fault
    # annunciators behave.
    recovery_hold_s: float = 0.3
    recovery_baseline_z: float = -2.0   # "recovered" band for the leak signal
    recovery_diff_z: float = 2.0        # "recovered" band for the dP/dt signal
    recovery_noise_ratio: float = 1.5   # "recovered" band for the chatter signal
    recovery_temp_z: float = 2.0        # "recovered" band for the temp signal


class RollingDetector:
    """
    Streaming, causal anomaly detector. Call `.update(t, pressure, temperature)`
    once per new sample; it returns a dict describing that sample's anomaly
    status. Internally it only keeps small rolling buffers (deques), so
    memory use is bounded regardless of test duration.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        sr = self.cfg.sample_rate_hz

        self._n_diff = max(10, int(self.cfg.diff_window_s * sr))
        self._n_baseline_std = max(10, int(self.cfg.baseline_std_window_s * sr))
        self._n_noise = max(10, int(self.cfg.noise_window_s * sr))
        self._n_temp = max(10, int(self.cfg.temp_window_s * sr))

        self._diff_buf = deque(maxlen=self._n_diff)
        self._pressure_std_buf = deque(maxlen=self._n_baseline_std)
        self._noise_buf = deque(maxlen=self._n_noise)
        self._noise_hist = deque(maxlen=self._n_noise * 4)
        self._temp_buf = deque(maxlen=self._n_temp)

        # slow EMA baseline state
        self._ema = None
        self._alpha = 1.0 / (self.cfg.ema_time_constant_s * sr)  # per-sample EMA weight

        self._last_pressure = None
        self._last_phase = None

        # alarm-latching state
        self._active_type = None
        self._n_recovery_hold = max(1, int(self.cfg.recovery_hold_s * sr))
        self._recovery_count = 0

    # ------------------------------------------------------------------
    def update(self, t: float, pressure: float, temperature: float, phase: str = "steady") -> dict:
        """
        `phase` is the commanded operating phase ("ramp_up", "steady",
        "ramp_down"). A real test stand knows this from valve/ignition
        commands. ALL statistical anomaly checks here only run during
        "steady" phase: during a commanded ramp, pressure/temperature are
        *expected* to change quickly and smoothly, which breaks z-score-style
        detectors in two ways -- a smooth ramp has almost no point-to-point
        variance, so any tiny fluctuation looks like a huge spike; and a
        legitimately large, intentional change looks identical to a fault.
        A production system would compare ramps against their own expected
        profile; that's out of scope here, so we simply don't judge ramps
        with the steady-state statistics.
        """
        cfg = self.cfg

        # Reset short-term buffers at the moment we transition INTO steady
        # state, so leftover ramp dynamics (very different scale/slope)
        # don't contaminate the first fraction of a second of monitoring.
        if phase == "steady" and self._last_phase != "steady":
            self._diff_buf.clear()
            self._noise_buf.clear()
            self._noise_hist.clear()
            self._temp_buf.clear()
            self._pressure_std_buf.clear()
            self._ema = pressure
            self._last_pressure = None
        self._last_phase = phase

        # ---- current-vs-history rate-of-change z-score ----
        dP = 0.0 if self._last_pressure is None else pressure - self._last_pressure
        diff_z = 0.0
        if len(self._diff_buf) >= 10:
            arr = np.fromiter(self._diff_buf, dtype=float)
            mu, sigma = arr.mean(), arr.std()
            sigma = sigma if sigma > 1e-6 else 1e-6
            diff_z = (dP - mu) / sigma
        self._last_pressure = pressure

        # ---- noise ratio (uses history only) ----
        noise_ratio = 1.0
        if len(self._noise_buf) >= 10:
            arr = np.fromiter(self._noise_buf, dtype=float)
            local_std = arr.std()
            hist = np.fromiter(self._noise_hist, dtype=float) if self._noise_hist else np.array([local_std])
            baseline_noise = np.median(hist) if len(hist) > 5 else local_std
            baseline_noise = baseline_noise if baseline_noise > 1e-6 else 1e-6
            noise_ratio = local_std / baseline_noise

        # ---- temperature z-score (history only) ----
        temp_z = 0.0
        if len(self._temp_buf) >= 10:
            arr = np.fromiter(self._temp_buf, dtype=float)
            mu, sigma = arr.mean(), arr.std()
            sigma = sigma if sigma > 1e-6 else 1e-6
            temp_z = (temperature - mu) / sigma

        # ---- slow-EMA baseline deviation z-score ----
        if self._ema is None:
            self._ema = pressure  # initialize on first sample
        baseline_z = 0.0
        if len(self._pressure_std_buf) >= 10:
            arr = np.fromiter(self._pressure_std_buf, dtype=float)
            base_std = arr.std()
            base_std = base_std if base_std > 1e-6 else 1e-6
            baseline_z = (pressure - self._ema) / base_std

        # ---- entry conditions (strict thresholds; steady-state only) ----
        in_steady = phase == "steady"
        is_sudden_drop = in_steady and (diff_z < -cfg.diff_z_threshold)
        is_sudden_spike = in_steady and (diff_z > cfg.diff_z_threshold)
        is_sustained_dev = in_steady and (baseline_z < -cfg.baseline_z_threshold)
        is_chatter = noise_ratio > cfg.noise_ratio_threshold
        is_temp_spike_flag = in_steady and (temp_z > cfg.temp_z_threshold)

        new_event_type = None
        if (is_sudden_drop or is_sudden_spike) and is_chatter:
            new_event_type = "cavitation"
        elif is_sustained_dev:
            new_event_type = "leak"
        elif is_sudden_drop:
            new_event_type = "cavitation"
        elif is_temp_spike_flag:
            new_event_type = "temp_spike"

        # ---- alarm latch / hysteresis state machine ----
        if self._active_type is None:
            if new_event_type is not None:
                self._active_type = new_event_type
                self._recovery_count = 0
        else:
            # already in an active event -- check if this sample looks "recovered"
            if self._active_type == "leak":
                looks_normal = baseline_z > cfg.recovery_baseline_z
            elif self._active_type == "cavitation":
                looks_normal = (abs(diff_z) < cfg.recovery_diff_z) and (noise_ratio < cfg.recovery_noise_ratio)
            else:  # temp_spike
                looks_normal = temp_z < cfg.recovery_temp_z

            if looks_normal:
                self._recovery_count += 1
                if self._recovery_count >= self._n_recovery_hold:
                    self._active_type = None
                    self._recovery_count = 0
            else:
                self._recovery_count = 0
                # a *different* fault type firing while latched escalates/overrides
                if new_event_type is not None and new_event_type != self._active_type:
                    self._active_type = new_event_type

        anomaly_type = self._active_type
        is_anomaly = anomaly_type is not None

        result = {
            "time_s": t,
            "pressure_psi": pressure,
            "temperature_k": temperature,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "diff_z": diff_z,
            "baseline_z": baseline_z,
            "noise_ratio": noise_ratio,
            "temp_z": temp_z,
            "ema_baseline": self._ema,
        }

        # ---- update rolling buffers/state for NEXT call ----
        self._diff_buf.append(dP)
        self._noise_buf.append(pressure)
        if len(self._noise_buf) == self._noise_buf.maxlen:
            self._noise_hist.append(np.fromiter(self._noise_buf, dtype=float).std())
        self._temp_buf.append(temperature)

        # Only feed the "normal operation" std estimator when not anomalous,
        # so a leak/cavitation event doesn't inflate the std used to judge itself.
        if not is_anomaly:
            self._pressure_std_buf.append(pressure)

        # Freeze the EMA baseline during a steady-state anomaly so a
        # multi-second leak can't drag its own reference value down with it.
        # During commanded ramp phases, snap the EMA to the live value so it
        # doesn't lag and cause a false "leak" the instant steady-state
        # checks resume.
        if phase != "steady":
            self._ema = pressure
        elif not is_anomaly:
            self._ema = (1 - self._alpha) * self._ema + self._alpha * pressure

        return result


def detect_anomalies_batch(df, config: DetectorConfig | None = None):
    """
    Convenience wrapper: run the streaming detector over a whole DataFrame
    (e.g. one produced by simulator.generate_hotfire_data) and return the
    DataFrame with detector columns appended. This just replays the stream
    sample-by-sample for offline evaluation -- the detector itself is
    unchanged from (and identical to) real-time use.
    """
    det = RollingDetector(config)
    rows = []
    has_phase = "phase" in df.columns
    for row in df.itertuples(index=False):
        phase = row.phase if has_phase else "steady"
        rows.append(det.update(row.time_s, row.pressure_psi, row.temperature_k, phase))
    out = df.reset_index(drop=True).copy()
    res_df = pd.DataFrame(rows)
    for col in ["is_anomaly", "anomaly_type", "diff_z", "baseline_z", "noise_ratio", "temp_z", "ema_baseline"]:
        out[col] = res_df[col]
    return out


if __name__ == "__main__":
    from simulator import generate_hotfire_data

    data = generate_hotfire_data(seed=42)
    result = detect_anomalies_batch(data)

    detected = result["is_anomaly"].sum()
    truth = result["true_anomaly"].sum()
    print(f"Ground-truth anomalous samples: {truth}")
    print(f"Detector-flagged samples:       {detected}")

    tp = ((result["is_anomaly"]) & (result["true_anomaly"])).sum()
    fp = ((result["is_anomaly"]) & (~result["true_anomaly"])).sum()
    fn = ((~result["is_anomaly"]) & (result["true_anomaly"])).sum()
    print(f"True positives: {tp}  False positives: {fp}  False negatives: {fn}")
    print("\nDetected anomaly type breakdown:")
    print(result.loc[result["is_anomaly"], "anomaly_type"].value_counts())
