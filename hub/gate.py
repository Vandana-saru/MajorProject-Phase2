"""
PRAHARI — Layer 4: trust gating.

The whole argument of the project lives here. Everything upstream proves the
system can tell a degraded window from a clean one; this module answers the
question a reviewer will actually ask, which is "so what?"

Gating policy
-------------
    STI >= release_threshold   -> RELEASE  (show the heart rate)
    STI >= flag_threshold      -> FLAG     (show it, marked unreliable)
    otherwise                  -> SUPPRESS (show nothing)

Clinically misleading is defined as |HR_observed - HR_true| exceeding
MISLEADING_BPM. The default of 5 bpm is a placeholder that WP4 must replace
with a cited tolerance: check the ANSI/AAMI EC13 heart-rate meter accuracy
clause and quote the actual figure rather than a round number. Leaving this
uncited is the easiest thing for a reviewer to attack.
"""

from dataclasses import dataclass

import numpy as np

MISLEADING_BPM = 5.0        # TODO(WP4): replace with the cited EC13 tolerance

RELEASE = "release"
FLAG = "flag"
SUPPRESS = "suppress"


@dataclass
class GatePolicy:
    release_threshold: int = 60
    flag_threshold: int = 30

    def decide(self, sti: int) -> str:
        if sti >= self.release_threshold:
            return RELEASE
        if sti >= self.flag_threshold:
            return FLAG
        return SUPPRESS


DEFAULT_POLICY = GatePolicy()


def apply_gate(hr_bpm, sti, policy=DEFAULT_POLICY):
    """
    Returns (decision, displayed_hr).
    displayed_hr is None when the reading is suppressed.
    """
    decision = policy.decide(int(sti))
    return decision, (None if decision == SUPPRESS else hr_bpm)


# ------------------------------------------------------------ threshold sweep
def sweep(sti_values, hr_errors, thresholds=None, misleading_bpm=MISLEADING_BPM):
    """
    THE HEADLINE FIGURE.

    For each candidate release threshold, compute:
        availability          fraction of windows where an HR is released
        misleading_released   fraction of ALL windows that release a reading
                              whose error exceeds the tolerance
        precision             of released readings, the fraction that are good

    Plot misleading_released against availability and the trade-off curve is
    the paper's central claim. The ungated baseline is threshold 0.

    sti_values : (n,) Signal Trust Index per window
    hr_errors  : (n,) |HR_obs - HR_true| per window, NaN allowed
    """
    sti_values = np.asarray(sti_values, dtype=np.float64)
    hr_errors = np.asarray(hr_errors, dtype=np.float64)
    ok = np.isfinite(hr_errors)

    if thresholds is None:
        thresholds = np.arange(0, 101, 2)

    rows = []
    n = int(ok.sum())
    for t in thresholds:
        released = ok & (sti_values >= t)
        n_rel = int(released.sum())
        misleading = int((released & (hr_errors > misleading_bpm)).sum())
        rows.append(dict(
            threshold=int(t),
            availability=n_rel / n if n else 0.0,
            misleading_released=misleading / n if n else 0.0,
            n_released=n_rel,
            n_misleading=misleading,
            precision=(1.0 - misleading / n_rel) if n_rel else float("nan"),
        ))
    return rows


def summarise(rows, max_misleading=0.01, target_availability=0.50):
    """
    Pick an operating point and report it beside the ungated baseline, so the
    reader sees what gating bought and what it cost.

    The principled rule for a SAFETY gate is safety-first, not availability-
    first: take the LOWEST threshold whose misleading-release rate is within
    tolerance, which is the same as keeping as much signal as possible subject
    to not lying to anyone.

        safe_point        lowest threshold with misleading_released <= tolerance
        constrained_point highest threshold still meeting target_availability
                          (the availability-first view, for contrast)

    CAVEAT FOR WP4: the synthetic corpus is ~80% degraded by construction,
    which is nothing like a real session where most windows are clean. Any
    availability figure computed on it is therefore pessimistic — a gate that
    releases "only the clean windows" scores ~20% here and would score ~90% on
    a well-mounted real recording. Re-run this sweep on the real data before
    quoting an operating point, and state the class prior you computed under.
    """
    baseline = rows[0]

    safe = next((r for r in rows if r["misleading_released"] <= max_misleading),
                min(rows, key=lambda r: r["misleading_released"]))

    feasible = [r for r in rows if r["availability"] >= target_availability]
    constrained = max(feasible, key=lambda r: r["threshold"]) if feasible else baseline

    def reduction(r):
        if baseline["misleading_released"] <= 0:
            return float("nan")
        return 1.0 - r["misleading_released"] / baseline["misleading_released"]

    return dict(
        baseline=baseline,
        operating_point=safe,
        constrained_point=constrained,
        max_misleading=max_misleading,
        target_availability=target_availability,
        misleading_reduction=reduction(safe),
        availability_cost=baseline["availability"] - safe["availability"],
        constrained_reduction=reduction(constrained),
    )
