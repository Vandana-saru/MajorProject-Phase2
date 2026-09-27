"""
PRAHARI — build the synthetic corpus.

Differs from Phase 1 in two ways that matter:

  1. Severity is kept as a continuous target (WP3's regression layer).
  2. The CLEAN SOURCE WINDOW is stored alongside every degraded window.
     This is the independent metric that stops WP3's restoration evaluation
     being circular, and it gives WP4 a ground-truth heart rate to measure
     error against. Phase 1 discarded it; that was the expensive mistake.

Usage
-----
    python -m ml.build_dataset --source bidmc      # needs network
    python -m ml.build_dataset --source synthetic  # works offline

Writes data/synthetic/corpus.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hub.contracts import FS, WINDOW_SAMPLES, CLASSES
from hub.preprocess import preprocess, window_signal, resample_to
from hub import degrade

BIDMC_RECORDS = ["bidmc01", "bidmc02", "bidmc03", "bidmc04", "bidmc05"]
FAULTS = ["Noisy", "Drifted", "Attenuated", "Spiked"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "synthetic", "corpus.npz")


# --------------------------------------------------------------- source: BIDMC
def load_bidmc(n_records=5):
    """Pull the PPG channel from PhysioNet. 125 Hz -> 100 Hz."""
    import wfdb
    sigs = []
    for rec in BIDMC_RECORDS[:n_records]:
        print(f"  fetching {rec} ...", flush=True)
        r = wfdb.rdrecord(rec, pn_dir="bidmc")
        names = [n.upper() for n in r.sig_name]
        ppg_idx = next((i for i, n in enumerate(names) if "PLETH" in n or "PPG" in n), None)
        if ppg_idx is None:
            print(f"    no PPG channel in {rec}, skipping")
            continue
        x = np.asarray(r.p_signal[:, ppg_idx], dtype=np.float64)
        x = x[np.isfinite(x)]
        sigs.append(resample_to(x, r.fs, FS))
    if not sigs:
        raise RuntimeError("no BIDMC records loaded")
    return sigs


# ----------------------------------------------------- source: offline fallback
def synth_ppg(duration_s=480, fs=FS, hr_bpm=72.0, seed=0):
    """
    A physiologically shaped PPG surrogate: systolic peak + dicrotic notch,
    heart-rate variability, respiratory amplitude modulation.

    This is NOT a substitute for BIDMC in the paper. It exists so that four
    people can build and test the whole pipeline on a day when PhysioNet is
    unreachable or the college wifi is being the college wifi.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # heart rate wanders a little
    hr = hr_bpm + 4.0 * np.sin(2 * np.pi * 0.04 * t) + rng.normal(0, 0.4, n).cumsum() * 0.002
    phase = np.cumsum(2 * np.pi * hr / 60.0 / fs)

    systolic = np.exp(-((np.mod(phase, 2 * np.pi) - 1.0) ** 2) / 0.18)
    dicrotic = 0.35 * np.exp(-((np.mod(phase, 2 * np.pi) - 2.6) ** 2) / 0.30)
    beat = systolic + dicrotic

    resp = 1.0 + 0.09 * np.sin(2 * np.pi * 0.25 * t)        # ~15 breaths/min
    x = beat * resp
    x += rng.normal(0, 0.012, n)                             # sensor floor noise
    return x


# --------------------------------------------------------------- corpus build
def build(sources, seed=42, max_clean_windows=None):
    """
    ORDER MATTERS. Normalise the CLEAN source first, then window, then inject
    degradation, and extract features from the degraded window without
    re-normalising.

    Doing it the other way round (degrade -> z-score) makes the Attenuated
    class bit-for-bit identical to Clean and erases most of Drifted. See the
    docstring in hub/preprocess.py.
    """
    rng = np.random.default_rng(seed)

    # 1. band-pass + z-score the clean source signal, THEN window it
    clean_windows = []
    for sig in sources:
        norm = preprocess(sig)                     # clean source -> normalised
        for w in window_signal(norm):
            clean_windows.append(w)
    clean_windows = np.asarray(clean_windows, dtype=np.float64)
    if max_clean_windows:
        clean_windows = clean_windows[:max_clean_windows]
    print(f"  {len(clean_windows)} clean source windows (normalised before windowing)")

    X, X_ref, y, sev = [], [], [], []

    # 2. Clean class — the normalised window, undegraded
    for w in clean_windows:
        X.append(w)
        X_ref.append(w)
        y.append("Clean")
        sev.append(0.0)

    # 3. Each fault, each severity, injected into the NORMALISED window.
    #    X_ref keeps the pre-degradation window — WP3's independent
    #    restoration metric and WP4's ground-truth heart rate.
    for fault in FAULTS:
        for level in (1, 2, 3):
            for w in clean_windows:
                d, s = degrade.apply(w, fault, level, rng)
                X.append(d)
                X_ref.append(w)
                y.append(fault)
                sev.append(s)

    X = np.asarray(X, dtype=np.float32)
    X_ref = np.asarray(X_ref, dtype=np.float32)
    y = np.asarray(y)
    sev = np.asarray(sev, dtype=np.float32)

    return dict(X=X, X_ref=X_ref, y=y, severity=sev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bidmc", "synthetic"], default="bidmc")
    ap.add_argument("--records", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-clean", type=int, default=None,
                    help="cap clean source windows (use ~200 for a fast smoke test)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.source == "bidmc":
        try:
            sources = load_bidmc(args.records)
        except Exception as e:
            print(f"  BIDMC unavailable ({e.__class__.__name__}: {e})")
            print("  falling back to the offline surrogate — DO NOT use these "
                  "numbers in the paper, only for pipeline testing")
            sources = [synth_ppg(480, seed=i) for i in range(args.records)]
            args.source = "synthetic-fallback"
    else:
        sources = [synth_ppg(480, seed=i) for i in range(args.records)]

    d = build(sources, seed=args.seed, max_clean_windows=args.max_clean)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, source=args.source, fs=FS, **d)

    print(f"\nwrote {args.out}")
    print(f"  {len(d['y'])} windows, {d['X'].shape[1]} samples each")
    for c in CLASSES:
        print(f"    {c:<12} {int((d['y'] == c).sum())}")


if __name__ == "__main__":
    main()
