# ALBANIA: "Looks like an anomaly!" - a streaming anomaly detector for bunker sensors.
#
# THE TASK IN PLAIN WORDS
#   A harness feeds us one sensor reading at a time (a dict with timestamp, bunker_zone and
#   six numeric sensors). For every reading, update() must return a number in [0, 1].
#   The grader treats score >= 0.5 as "anomaly" and compares that to hidden labels.
#   We may remember the past, but we can never look ahead.
#
# WHAT THE TRAINING DATA SHOWS (train-input.csv / train-output.csv)
#   * Readings arrive once per minute. The two zones come in blocks of 900 minutes
#     (north, south, north, south), so each zone has 15-hour gaps in its own history.
#   * "Normal" is not flat: temperature swings on a ~12 hour cycle and air pressure on a
#     ~20 hour cycle. A detector with one fixed "normal mean" mistakes those swings for
#     drift and raises hundreds of false alarms, which wrecks F1 (80% of the score).
#   * The anomalies, measured against a baseline that follows those cycles:
#       point      - 1 reading; one sensor jumps 14-25 standard deviations (sigma).
#       drift      - ~60 readings; one sensor starts ~3.5 sigma off and keeps walking to ~10.
#       collective - ~30-35 readings; vibration, acoustic and CO2 all sit ~4 sigma off together.
#
# THE IDEA
#   1. For each zone and sensor, keep a moving "expected value" (an exponential moving
#      average) that is quick enough to follow the slow daily cycles, but too slow to
#      follow a drift - so a drift pulls away from it and stands out.
#   2. Each reading's residual (actual - expected) divided by the noise level gives a
#      z-score per sensor: how unusual it is right now.
#   3. Three checks on those z-scores, each turned into a 0..1 score:
#        point      - is any single z-score huge?
#        drift      - has one sensor's short average moved clearly away from 0?
#        collective - is the combined (sum of squares) short average of all sensors big?
#      We return the largest of the three.
#   4. The expected value only learns from readings that look normal. While something
#      looks anomalous it is frozen (it stays where it was), so a drift can't
#      "teach" the detector that the drifted value is normal.
#
# Start reading at Detector.update(); the bottom of the file replays the training data
# and prints the same score the organizers compute.

"""Streaming anomaly detector: point spikes, gradual drifts and collective shifts."""

import math
from datetime import datetime

# The six numeric sensors in every reading, in a fixed order.
SENSORS = (
    "temperature_c",
    "humidity_pct",
    "vibration_g",
    "acoustic_db",
    "air_pressure_kpa",
    "co2_ppm",
)

# ---------------------------------------------------------------------------------------
# Tunable settings. Each one is explained next to its value.
# ---------------------------------------------------------------------------------------

# How many readings a zone needs before we judge anything: we fit a straight line through
# them to get a starting level and noise level. WARMUP is used once per zone;
# REWARMUP is used after a gap in time (the cycle has moved on, so the old level is stale).
WARMUP = 60
REWARMUP = 20

# A time gap (in minutes) bigger than this means "the stream for this zone paused".
GAP_MINUTES = 5

# How strongly each normal reading pulls the expected value towards itself (an EMA).
# This rate has to sit between two speeds:
#   * fast enough to follow the 12-20 hour cycles (they move only ~0.02 sigma per minute),
#   * slow enough NOT to follow a drift (a drift ramps ~0.15 sigma per minute).
# Within that window the expected value hugs the cycle, but a drift runs away from it and
# gets flagged. (Holt's method was tried first: it also learns slopes, so it happily
# learned each drift ramp as "normal" and missed every drift.)
LEVEL_RATE = 0.02

# The noise level (sigma) of each sensor also keeps learning slowly from normal readings.
SIGMA_RATE = 0.01

# Point detector: a single z-score above this is a spike. Real spikes are 14+ sigma and
# ordinary noise basically never passes 5, so there is a wide safe margin.
POINT_Z = 7.0

# Before z-scores enter the short average we cap them at +-CLIP, so a spike can't push
# the average up and cause false alarms on the following readings.
CLIP = 6.0

# Short average (exponential moving average, EMA) of each sensor's z-score. rate 0.4
# means the last ~2-4 readings matter most: noise (+-1 per reading) shrinks to about
# +-0.5, while a lasting shift of 3-4 sigma shows up within one or two readings. A short
# memory also means the alarm switches off quickly once the anomaly is over.
SHORT_RATE = 0.4

# Drift detector: one sensor's short average beyond this many sigma (about 5x the
# averaged noise, so it rarely fires by chance on normal data).
DRIFT_Z = 2.5

# Collective detector: sum over all sensors of (short average)^2 beyond this.
# Collective events move three sensors by ~4 sigma each (total ~50), while normal noise
# gives about 6 x 0.5^2 = 1.5. 15 sits safely between, and still catches three sensors at
# ~2.3 sigma each even when none of them alone would trip the drift check.
COLL_SUM = 15.0

# Values above were chosen from a grid search on the training data, picking a setting in
# the middle of the good region (scores 94-96 around it) rather than the single best one,
# so small differences in the hidden test stream don't push us off a cliff.

# A reading whose score is above this does not update the expected value or sigma.
LEARN_BELOW = 0.3

# Safety valve: if a zone has looked anomalous for this many readings in a row, assume
# "normal" really changed and start learning again (otherwise we'd alarm forever).
MAX_FROZEN = 240

# How sharp the 0..1 conversion is around each threshold.
SHARPNESS = 6.0


def squash(statistic, threshold):
    """Turn a non-negative statistic into a score in (0, 1) that is exactly 0.5 at threshold.

    It is a sigmoid on statistic/threshold: far below the threshold gives ~0, far above
    gives ~1, and the grader's 0.5 cut-off falls exactly on our threshold.
    """
    x = SHARPNESS * (statistic / threshold - 1.0)
    x = max(-50.0, min(50.0, x))  # keep math.exp from overflowing
    return 1.0 / (1.0 + math.exp(-x))


def fit_line(values):
    """Least-squares straight line through values at times 0..n-1.

    Returns (value of the line at the last time, slope per step, standard deviation of the
    points around the line). Used to start the expected value after a warm-up.
    """
    n = len(values)
    t_mean = (n - 1) / 2.0
    v_mean = sum(values) / n
    num = sum((t - t_mean) * (v - v_mean) for t, v in enumerate(values))
    den = sum((t - t_mean) ** 2 for t in range(n)) or 1.0
    slope = num / den
    residuals = [v - (v_mean + slope * (t - t_mean)) for t, v in enumerate(values)]
    sd = math.sqrt(sum(r * r for r in residuals) / max(n - 2, 1))
    return v_mean + slope * (n - 1 - t_mean), slope, sd


def parse_time(value):
    """Timestamp as a datetime, or None if it is missing or unreadable."""
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class ZoneState:
    """Everything we remember about one bunker zone."""

    def __init__(self):
        n = len(SENSORS)
        self.buffer = []           # readings collected during a (re)warm-up
        self.ready = False         # True once level and sigma exist
        self.level = [0.0] * n     # expected value of each sensor right now
        self.sigma = [1.0] * n     # noise level of each sensor around the expected value
        self.short = [0.0] * n     # short EMA of z-scores
        self.last_time = None      # timestamp of the previous reading of this zone
        self.frozen = 0            # how many readings in a row we skipped learning


class Detector:
    def __init__(self):
        # Each zone has its own sensors and its own idea of normal, so state is per zone.
        self.zones = {}

    def update(self, reading: dict) -> float:
        try:
            return self._update(reading)
        except Exception:
            # Never crash the harness: an unexpected input simply counts as "normal".
            return 0.0

    def _update(self, reading):
        zone = self.zones.setdefault(reading.get("bunker_zone", "default"), ZoneState())

        # Read the six sensor values. A missing or broken value is replaced by what we
        # expected, so it neither raises nor lowers the score.
        values = []
        for i, name in enumerate(SENSORS):
            try:
                v = float(reading[name])
                if not math.isfinite(v):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                v = zone.level[i]
            values.append(v)

        # A pause in this zone's stream: the daily cycles moved on while we weren't looking,
        # so the old level is stale. Keep sigma (noise doesn't change), but re-learn
        # the level from a short new warm-up.
        now = parse_time(reading.get("timestamp"))
        if now is not None and zone.last_time is not None:
            gap = (now - zone.last_time).total_seconds() / 60.0
            if gap > GAP_MINUTES and zone.ready:
                zone.ready = False
                zone.buffer = []
                zone.short = [0.0] * len(SENSORS)
        if now is not None:
            zone.last_time = now

        # ---- Warm-up: collect readings, then fit a line per sensor to start the level. ----
        if not zone.ready:
            # Even while warming up again, a huge spike is easy to see with the old sigma.
            spike = 0.0
            if zone.buffer and any(s != 1.0 for s in zone.sigma):
                ref = zone.buffer[-1]
                spike = max(abs(v - r) / s for v, r, s in zip(values, ref, zone.sigma))
                if spike > POINT_Z * 1.5:
                    return squash(spike, POINT_Z)  # don't let the spike into the buffer
            zone.buffer.append(values)
            first_time = all(s == 1.0 for s in zone.sigma)
            if len(zone.buffer) >= (WARMUP if first_time else REWARMUP):
                for i in range(len(SENSORS)):
                    # Start the expected value where the fitted line ends (the slope is
                    # only used here, to place that end point correctly).
                    level, _slope, sd = fit_line([row[i] for row in zone.buffer])
                    zone.level[i] = level
                    if first_time:
                        zone.sigma[i] = max(sd, 1e-9)
                zone.ready = True
                zone.buffer = []
            return 0.0

        # ---- Residuals and z-scores against the expected value. ----
        residual = [v - l for v, l in zip(values, zone.level)]
        z = [r / s for r, s in zip(residual, zone.sigma)]

        # Point detector: the biggest single z-score right now.
        point_stat = max(abs(zi) for zi in z)
        is_spike = point_stat > POINT_Z

        # Short averages, fed with capped z-scores. A spike is skipped completely: one
        # wild reading says nothing about a drift or a lasting shift.
        if not is_spike:
            for i, zi in enumerate(z):
                zc = max(-CLIP, min(CLIP, zi))
                zone.short[i] += SHORT_RATE * (zc - zone.short[i])

        drift_stat = max(abs(e) for e in zone.short)       # one sensor clearly off
        coll_stat = sum(e * e for e in zone.short)          # many sensors moderately off

        score = max(
            squash(point_stat, POINT_Z),
            squash(drift_stat, DRIFT_Z),
            squash(coll_stat, COLL_SUM),
        )

        # ---- Learn from this reading only if it looks normal (or we've been stuck). ----
        learn = score < LEARN_BELOW or zone.frozen >= MAX_FROZEN
        if learn:
            zone.frozen = 0
            for i in range(len(SENSORS)):
                # Move the expected value a small step (LEVEL_RATE) towards the reading.
                zone.level[i] += LEVEL_RATE * residual[i]
                # Noise level: slow EMA of the squared residual (capped so a borderline
                # reading can't inflate it much).
                r = max(-3 * zone.sigma[i], min(3 * zone.sigma[i], residual[i]))
                zone.sigma[i] = math.sqrt((1 - SIGMA_RATE) * zone.sigma[i] ** 2 + SIGMA_RATE * r * r)
        else:
            # Frozen: keep the expected value where it was before the anomaly started.
            zone.frozen += 1

        # squash() already returns a finite value in (0, 1); clamp again to be safe.
        return float(min(1.0, max(0.0, score)))


# ---------------------------------------------------------------------------------------
# Local check: replay the training stream and print the organizer's scoring formula.
# Run:  python detector.py   (from inside the Albania folder)
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    import pandas as pd

    # Which file to replay:
    #   python detector.py some-input.csv   -> that file
    #   python detector.py                  -> test-input.csv if it exists, else train-input.csv
    # Paths are looked up next to this script, so it works from any working directory.
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    elif os.path.exists(os.path.join(here, "test-input.csv")):
        input_path = os.path.join(here, "test-input.csv")
    else:
        input_path = os.path.join(here, "train-input.csv")

    # Which labels file to score against, chosen the same way:
    #   python detector.py some-input.csv some-output.csv  -> that labels file
    #   otherwise                                           -> test-output.csv if it exists,
    #                                                          else train-output.csv
    folder, name = os.path.split(input_path)
    if len(sys.argv) > 2:
        labels_path = sys.argv[2]
    elif os.path.exists(os.path.join(here, "test-output.csv")):
        labels_path = os.path.join(here, "test-output.csv")
    else:
        labels_path = os.path.join(here, "train-output.csv")
    print(f"reading {input_path}")

    inputs = pd.read_csv(input_path)

    # Feed the readings one by one, exactly like the organizer harness does.
    detector = Detector()
    scores = [detector.update(row) for row in inputs.to_dict("records")]
    predicted = [s >= 0.5 for s in scores]

    # Load the labels, but only use them if they really belong to this input: they must
    # have an is_anomaly column and one row per reading. (This stops us from, say, scoring
    # test-input.csv against train-output.csv by accident.)
    labels = None
    if labels_path != input_path and os.path.exists(labels_path):
        labels = pd.read_csv(labels_path)
        if "is_anomaly" not in labels.columns or len(labels) != len(inputs):
            print(f"ignoring {labels_path}: it does not match {name}")
            labels = None
        else:
            print(f"scoring against {labels_path}")

    # Without usable labels there is nothing to score, so we save our predictions
    # instead - into a separate file, so no labels file is ever overwritten.
    if labels is None:
        out_path = os.path.join(folder, name.replace(".csv", "") + "-predictions.csv")
        pd.DataFrame({
            "timestamp": inputs.get("timestamp", pd.Series(range(len(inputs)))),
            "is_anomaly": [int(p) for p in predicted],
            "anomaly_score": scores,
        }).to_csv(out_path, index=False)
        print(f"no labels found; flagged {sum(predicted)} of {len(predicted)} readings")
        print(f"predictions written to {out_path}")
        sys.exit(0)

    actual = labels["is_anomaly"].astype(bool).tolist()
    kinds = labels["anomaly_type"].tolist()

    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(a and not p for p, a in zip(predicted, actual))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    def recall(kind):
        hits = [p for p, k in zip(predicted, kinds) if k == kind]
        return sum(hits) / len(hits) if hits else 0.0

    final = 100 * (0.80 * f1 + 0.05 * recall("point") + 0.075 * recall("drift") + 0.075 * recall("collective"))
    print(f"TP={tp} FP={fp} FN={fn}  F1={f1:.3f}")
    print(f"recall point={recall('point'):.3f} drift={recall('drift'):.3f} collective={recall('collective'):.3f}")
    print(f"final score = {min(100, max(1, final)):.2f} / 100")
