# PRAHARI — frozen interface contracts

**Frozen 21 Sep 2026.** Do not change anything here without telling all four
work packages. Every boundary in the system is defined by this document, and
it is the only reason four people on four laptops can work without blocking
each other.

The executable version is `hub/contracts.py`. This file explains it.

---

## 1. Serial frame — WP1 → WP2 → hub

One JSON object per line, one line per 5-second window, 50% overlap.

```json
{"seq":41,"t_ms":205300,"fs":100,"n":500,"jit_us":12,"samples":[2047.0, ...]}
```

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | monotonic window counter — gaps mean dropped frames |
| `t_ms` | int | node `millis()` at the moment the window was emitted |
| `fs` | int | sampling rate, always 100 |
| `n` | int | sample count, always 500 |
| `jit_us` | int | worst per-sample timing error in this window, µs |
| `samples` | float[] | raw sensor units, oldest first |

Once on-board inference is live the node also sends `class`, `proba`, `sev`
and `sti`. The hub tolerates their absence.

Lines beginning with `#` are human-readable status and are ignored by the
parser. Anything that is not valid JSON with all required keys is dropped
silently — a node that resets mid-line must not crash the hub.

**Baud: 921600.** At 100 Hz, 500 floats every 2.5 s, 115200 will not keep up.

---

## 2. Window — everyone

```
float32, shape (500,), 100 Hz, 50% overlap
```

Normalised. Which normalisation depends on the path:

| Path | Normalisation |
|---|---|
| Synthetic corpus | band-pass → per-window z-score of the **clean source**, *then* degradation injected |
| Real recordings | band-pass → scale by the **fixed session calibration** from that subject's clean baseline |

### Why this matters more than it looks

Per-window z-scoring destroys the Attenuated class completely. Not
approximately — exactly:

```
max | zscore(x) - zscore(0.15 * x) |  ==  0   (to float precision)
```

A degraded-then-normalised window is indistinguishable from a clean one. The
same applies, less absolutely, to Drifted, because the 0.5 Hz high-pass removes
most of the baseline wander it is meant to detect.

So the **order is part of the contract**:

```
SYNTHETIC   raw → band-pass → z-score → window → INJECT → features
REAL        raw → band-pass → calibrated scale → window → features
```

`tests/test_pipeline.py::test_zscore_destroys_attenuation` pins this. If it
fails, stop and find out who changed what.

---

## 3. Feature vector — WP2 ↔ WP3

`float32, shape (13,)`. **Order is frozen.** The ESP32 C implementation must
emit exactly this order, index for index, or WP2's parity report is
meaningless.

| # | Name | Domain |
|---|---|---|
| 0 | `mean` | time |
| 1 | `variance` | time |
| 2 | `rms` | time |
| 3 | `kurtosis` | time (Fisher / excess) |
| 4 | `zcr` | time (zero-crossing rate) |
| 5 | `p2p` | time (peak-to-peak) |
| 6 | `skewness` | time |
| 7 | `mad` | time (mean absolute deviation) |
| 8 | `dom_freq` | frequency |
| 9 | `spec_entropy` | frequency (base-2) |
| 10 | `ppg_band_power` | frequency (0.5–4 Hz) |
| 11 | `power_ratio` | frequency (band / total) |
| 12 | `spec_centroid` | frequency |

Spectrum convention, which WP2 must match exactly in C: **Hann window**,
real FFT, one-sided magnitude, PSD = magnitude². Getting the window function
or the normalisation wrong is the usual cause of a failed parity check.

---

## 4. Verdict — WP3 → WP4

```json
{"class":"Noisy","proba":[0.05,0.80,0.05,0.05,0.05],"severity":0.62,"sti":31}
```

- `class` — one of `Clean, Noisy, Drifted, Attenuated, Spiked`
- `proba` — length 5, **in that class order**, not the classifier's internal order
- `severity` — float 0.0 (clean) to 1.0 (worst)
- `sti` — int 0–100

### Signal Trust Index

```
STI = 100 × P(Clean) × (1 − severity)
```

Either signal alone can pull trust down: a confident degraded classification,
or a high severity. **This definition is frozen** because WP4's threshold
sweep is calibrated against it — change the formula and every threshold
already swept becomes meaningless.

---

## 5. Restoration — WP3 internal, consumed by WP4

```python
restore(window, verdict) -> window     # same shape, same normalisation
```

Restoration never changes the shape or the units. A `Clean` verdict returns
the window unchanged.

**Evaluation rule (contractual, not advisory):** restoration is never reported
on the basis of "the classifier now says Clean". That is circular — the
classifier is the thing being satisfied. Every restoration claim must lead
with a classifier-independent metric:

- correlation against the pre-degradation window (synthetic; `X_ref` in the corpus)
- SNR improvement in dB against the same reference
- heart-rate error reduction (real data; WP4 supplies it)

`hub/restore.py::evaluate` returns the independent metrics by default and
labels the circular one explicitly.

---

## 6. Corpus file — `data/synthetic/corpus.npz`

| Key | Shape | Meaning |
|---|---|---|
| `X` | (n, 500) | the window as the model sees it, degraded |
| `X_ref` | (n, 500) | the **pre-degradation** window — the independent reference |
| `y` | (n,) | class label string |
| `severity` | (n,) | 0.0 / 0.33 / 0.66 / 1.00 |

`X_ref` is the thing Phase 1 threw away. It is what lets WP3 evaluate
restoration honestly and WP4 compute a ground-truth heart rate. Never drop it.
