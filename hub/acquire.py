"""
PRAHARI — hub-side acquisition.

Reads JSON frames from the ESP32 node, or replays a recorded session so that
the three work packages without hardware can run the identical code path.

Usage
-----
    # live, recording to disk
    python -m hub.acquire --port /dev/ttyUSB0 --record data/raw/ramaa_s1.jsonl

    # sampling validation (WP1's 100 Hz proof)
    python -m hub.acquire --port /dev/ttyUSB0 --validate 60

    # replay for everyone without a board
    python -m hub.acquire --replay data/raw/ramaa_s1.jsonl
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hub.contracts import SERIAL_BAUD, FRAME_REQUIRED_KEYS, FS, WINDOW_SAMPLES


# --------------------------------------------------------------- frame parsing
def parse_line(line):
    """Returns a frame dict, or None for comments / malformed lines."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        f = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not all(k in f for k in FRAME_REQUIRED_KEYS):
        return None
    if len(f["samples"]) != f["n"]:
        return None
    return f


# ------------------------------------------------------------------ live serial
def stream_serial(port, baud=SERIAL_BAUD, record=None, timeout=1.0):
    """Yields frames from the ESP32. Optionally appends raw lines to `record`."""
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial not installed:  pip install pyserial")

    ser = serial.Serial(port, baud, timeout=timeout)
    sink = open(record, "a") if record else None
    print(f"# listening on {port} @ {baud}", file=sys.stderr)
    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="replace")
            if not raw:
                continue
            if raw.startswith("#"):
                print(raw.rstrip(), file=sys.stderr)
                continue
            f = parse_line(raw)
            if f is None:
                continue
            if sink:
                sink.write(raw if raw.endswith("\n") else raw + "\n")
                sink.flush()
            yield f
    finally:
        ser.close()
        if sink:
            sink.close()


# ---------------------------------------------------------------------- replay
def stream_replay(path, realtime=False, hop_s=2.5):
    """Yields frames from a recorded .jsonl session."""
    with open(path) as fh:
        for raw in fh:
            f = parse_line(raw)
            if f is None:
                continue
            if realtime:
                time.sleep(hop_s)
            yield f


def frame_to_window(frame):
    return np.asarray(frame["samples"], dtype=np.float64)


# ------------------------------------------------------- WP1 sampling validation
def validate_sampling(port, seconds=60, baud=SERIAL_BAUD, out=None):
    """
    The 100 Hz proof. Collects frames, reconstructs the sample timeline from
    frame timestamps, and reports effective rate and jitter.

    Produces the inter-sample-interval histogram that goes in the paper's
    Methods section, so run this before collecting any real data — if the rate
    is wrong, every frequency feature downstream is wrong and nothing later
    will tell you.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    t_ms, seqs, jitters = [], [], []
    for f in stream_serial(port, baud):
        t_ms.append(f["t_ms"])
        seqs.append(f["seq"])
        jitters.append(f.get("jit_us", 0))
        if time.time() - t0 > seconds:
            break

    if len(t_ms) < 3:
        raise SystemExit("not enough frames — is the node streaming?")

    t_ms = np.asarray(t_ms, dtype=np.float64)
    seqs = np.asarray(seqs)
    dt_frame = np.diff(t_ms) / 1000.0
    hop_s = WINDOW_SAMPLES / 2 / FS
    eff_fs = (WINDOW_SAMPLES / 2) / np.median(dt_frame)

    dropped = int(np.sum(np.diff(seqs) != 1))

    print(f"\n  frames            : {len(t_ms)}")
    print(f"  dropped/out-of-seq: {dropped}")
    print(f"  median frame gap  : {np.median(dt_frame)*1000:.2f} ms "
          f"(expected {hop_s*1000:.1f} ms)")
    print(f"  effective rate    : {eff_fs:.3f} Hz (target {FS} Hz)")
    print(f"  rate error        : {(eff_fs-FS)/FS*100:+.3f} %")
    print(f"  worst ISR jitter  : {max(abs(j) for j in jitters)} us")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(dt_frame * 1000, bins=40, color="#0B6E5F")
    ax[0].axvline(hop_s * 1000, color="#9B2A2A", ls="--", label="expected")
    ax[0].set_xlabel("inter-frame interval (ms)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"Frame timing — effective {eff_fs:.2f} Hz")
    ax[0].legend()

    ax[1].plot(jitters, lw=0.8, color="#2F4C77")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("worst sample jitter (us)")
    ax[1].set_title("Per-frame sampling jitter")
    fig.tight_layout()

    out = out or "docs/figures/wp1_sampling_validation.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\n  wrote {out}   <- this is a paper figure, keep it")
    return dict(effective_fs=float(eff_fs), dropped=dropped, n_frames=len(t_ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="e.g. /dev/ttyUSB0 or COM5")
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--record", help="append raw frames to this .jsonl")
    ap.add_argument("--replay", help="replay a recorded .jsonl instead")
    ap.add_argument("--validate", type=int, metavar="SECONDS",
                    help="run the sampling-rate validation for N seconds")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.validate:
        if not args.port:
            raise SystemExit("--validate needs --port")
        validate_sampling(args.port, args.validate, args.baud)
        return

    src = (stream_replay(args.replay) if args.replay
           else stream_serial(args.port, args.baud, args.record))

    for i, f in enumerate(src, 1):
        w = frame_to_window(f)
        print(f"seq={f['seq']:>5}  n={f['n']}  "
              f"mean={w.mean():>10.2f}  p2p={np.ptp(w):>10.2f}  "
              f"jit={f.get('jit_us', 0):>5} us")
        if args.limit and i >= args.limit:
            break


if __name__ == "__main__":
    main()
