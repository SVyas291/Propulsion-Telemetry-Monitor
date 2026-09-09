"""
visualize.py
------------
Live, updating plot of the simulated hot-fire test. Pressure and temperature
traces stream in "real time" (the 60s test plays out over ~60 real seconds),
and the line turns RED for the duration of any flagged anomaly, with a
shaded red band and an annotation naming the fault type.

Run directly:
    python visualize.py

Controls:
    - Close the window to stop.
    - Edit `SEED` below to get a different random test (or set to None).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from simulator import generate_hotfire_data
from detector import RollingDetector, DetectorConfig

SEED = 7                  # set to None for a different random test each run
SAMPLE_RATE_HZ = 500.0
DURATION_S = 60.0
FPS = 30                  # animation refresh rate
PLAYBACK_SPEED = 1.0      # 1.0 = real time; 2.0 = twice as fast, etc.

ANOMALY_COLORS = {
    "cavitation": "#ff3b30",
    "leak": "#ff9500",
    "temp_spike": "#ff3b30",
    None: "#2ecc71",
}


def build_animation():
    # Pre-generate the full "sensor recording" -- in a real system this would
    # instead be a live socket/DAQ feed, but the detector below only ever
    # looks at data up to the current sample, so it's identical either way.
    data = generate_hotfire_data(
        duration_s=DURATION_S, sample_rate_hz=SAMPLE_RATE_HZ, seed=SEED
    )
    n_total = len(data)
    samples_per_frame = max(1, int((SAMPLE_RATE_HZ / FPS) * PLAYBACK_SPEED))
    n_frames = int(np.ceil(n_total / samples_per_frame))

    detector = RollingDetector(DetectorConfig(sample_rate_hz=SAMPLE_RATE_HZ))

    # rolling "processed so far" buffers
    t_hist, p_hist, temp_hist = [], [], []
    anomaly_type_hist = []

    # ---- figure setup ----
    plt.style.use("dark_background")
    fig, (ax_p, ax_t) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Liquid Rocket Engine Feedline -- Live Hot-Fire Monitor", fontsize=13)

    ax_p.set_ylabel("Pressure (psi)")
    ax_p.set_xlim(0, DURATION_S)
    ax_p.set_ylim(-10, data["pressure_psi"].max() * 1.15)
    ax_p.grid(alpha=0.2)

    ax_t.set_ylabel("Temperature (K)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_xlim(0, DURATION_S)
    ax_t.set_ylim(data["temperature_k"].min() * 0.97, data["temperature_k"].max() * 1.05)
    ax_t.grid(alpha=0.2)

    # LineCollections let us color each tiny segment individually (green/red)
    lc_p = LineCollection([], linewidths=1.6)
    lc_t = LineCollection([], linewidths=1.6)
    ax_p.add_collection(lc_p)
    ax_t.add_collection(lc_t)

    status_text = ax_p.text(
        0.99, 0.95, "", transform=ax_p.transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold", color="#2ecc71",
    )
    time_text = ax_p.text(0.01, 0.95, "", transform=ax_p.transAxes, ha="left", va="top", fontsize=10)

    # keep track of shaded anomaly spans so we don't redraw duplicates every frame
    span_state = {"active": False, "start_t": None, "type": None, "patches": []}

    def _segments(x, y, colors):
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        return segs, colors[1:]  # color segment by its ending point's status

    def update(frame):
        start = frame * samples_per_frame
        end = min(n_total, start + samples_per_frame)
        if start >= n_total:
            return lc_p, lc_t, status_text, time_text

        chunk = data.iloc[start:end]
        for row in chunk.itertuples(index=False):
            result = detector.update(row.time_s, row.pressure_psi, row.temperature_k, row.phase)
            t_hist.append(row.time_s)
            p_hist.append(row.pressure_psi)
            temp_hist.append(row.temperature_k)
            anomaly_type_hist.append(result["anomaly_type"])

            # manage shaded red span for the live anomaly region
            if result["is_anomaly"] and not span_state["active"]:
                span_state["active"] = True
                span_state["start_t"] = row.time_s
                span_state["type"] = result["anomaly_type"]
            elif not result["is_anomaly"] and span_state["active"]:
                span_state["active"] = False
                p1 = ax_p.axvspan(span_state["start_t"], row.time_s, color="red", alpha=0.12, zorder=0)
                p2 = ax_t.axvspan(span_state["start_t"], row.time_s, color="red", alpha=0.12, zorder=0)
                span_state["patches"].extend([p1, p2])

        colors = [ANOMALY_COLORS.get(a, "#ff3b30") for a in anomaly_type_hist]

        if len(t_hist) >= 2:
            segs_p, seg_colors = _segments(t_hist, p_hist, colors)
            lc_p.set_segments(segs_p)
            lc_p.set_color(seg_colors)

            segs_t, _ = _segments(t_hist, temp_hist, colors)
            lc_t.set_segments(segs_t)
            lc_t.set_color(seg_colors)

        current_type = anomaly_type_hist[-1]
        if current_type is not None:
            status_text.set_text(f"\u26a0 ANOMALY: {current_type.upper()}")
            status_text.set_color(ANOMALY_COLORS[current_type])
        else:
            status_text.set_text("NOMINAL")
            status_text.set_color("#2ecc71")

        time_text.set_text(f"t = {t_hist[-1]:5.1f} s")

        return lc_p, lc_t, status_text, time_text

    interval_ms = 1000.0 / FPS
    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms, blit=False, repeat=False
    )
    plt.tight_layout()
    return fig, anim


def main():
    fig, anim = build_animation()
    plt.show()
    return anim  # keep a reference so it isn't garbage-collected


if __name__ == "__main__":
    main()
