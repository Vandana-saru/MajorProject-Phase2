# PRAHARI — data collection protocol

Version 1.0, 21 Sep 2026. WP1 owns this document.

Real degradation has no ground truth. Nobody can look at window 847 and say
"that is Attenuated rather than Noisy" with any authority. So we do not label
by eye — we **induce** each degradation deliberately and label by timestamp.
That is the difference between a defensible dataset and one a reviewer
dismisses in a sentence.

---

## Before anyone records anything

### 1. Ethics and consent — ask the guides first

We are recording physiological signals from four named people. Ask TVS
Adinarayana and Shasikiran S at Review 1 whether the department requires a
written consent declaration or an ethics note, **even for self-recording among
team members**. Many do. Discovering this in week five is far worse than
asking now. Until they answer, record only team members who have verbally
agreed, and keep a dated note of who agreed and when.

### 2. Sampling validation — non-negotiable

```bash
python -m hub.acquire --port /dev/ttyUSB0 --validate 60
```

Effective rate must be within **±1%** of 100 Hz and dropped frames must be 0.
If the rate is wrong, every frequency-domain feature is computed on the wrong
axis — and three of the top four features by Gini importance are frequency
features. Nothing downstream will warn you. The histogram this writes to
`docs/figures/wp1_sampling_validation.png` is a paper figure; keep it.

### 3. Mounting

Use the velcro strap, hair tie or medical tape. **Not** held by hand. Mount
pressure is the single largest source of uncontrolled variance in PPG, and a
hand-held sensor produces data that cannot be reproduced tomorrow.

Same finger every session (index finger, non-dominant hand, unless the subject
has a reason to differ — record the reason). Forearm supported on the table.

---

## Per-session procedure

Every session, every subject, in this order. Total ≈ 8 minutes of recording.

### Step 0 — clean baseline (120 s) — THE CALIBRATION, NOT A WARM-UP

Seated, still, forearm supported, strap firm. Record 120 seconds.

This baseline is what makes the Attenuated class detectable at all. Per-window
z-scoring mathematically erases amplitude information — a window scaled by
0.15 and then z-scored is *bit-for-bit identical* to the clean one. So the
real-data path normalises every later window against the **fixed** mean and
scale taken from this baseline, not against itself:

```python
from hub.preprocess import Calibration
cal = Calibration.from_baseline(baseline_raw, subject="ramaa")
cal.save("data/raw/ramaa_s1_calibration.json")
```

Skip this step and the Attenuated class quietly becomes undetectable in
deployment while still scoring perfectly in your offline evaluation. That is
the worst kind of bug.

### Steps 1-4 — induced degradation, 90 s each

Rest 20 s between blocks. Announce the start and end of each block out loud so
it lands in the log; the operator marks timestamps.

| Block | Class | Manoeuvre | Duration |
|---|---|---|---|
| 1 | **Noisy** | Deliberate finger tremor, tapping, rubbing against the sensor face. Vary intensity across the block: 30 s light, 30 s moderate, 30 s vigorous. | 90 s |
| 2 | **Attenuated** | Progressive loss of contact: 30 s strap loosened one notch, 30 s loosened two notches, 30 s finger resting with light contact only. | 90 s |
| 3 | **Drifted** | Slow sustained pressure change: press gradually harder over ~10 s, release over ~10 s, repeat. Then hand lowered below heart level and raised, ~10 s cycles. | 90 s |
| 4 | **Spiked** | Sharp cable jolt or a single knock to the sensor body, at marked intervals of roughly 10 s. Keep the hand still between knocks. | 90 s |

The three intensity sub-blocks within each 90 s are what give WP3's severity
regressor a graded target on real data, mirroring the three synthetic levels.
Mark the sub-block boundaries too.

### Step 5 — clean closing baseline (60 s)

Same as step 0. If it looks materially different from the opening baseline,
the mount shifted during the session and the session is suspect. Note it.

---

## Labelling

1. Labels come from the **protocol timestamps**, not from looking at waveforms.
2. Discard the 3 seconds either side of every block boundary — the transition
   is neither class.
3. **Second-rater pass.** One other team member independently reviews the
   labelled blocks against the session log and flags anything that looks
   mislabelled. Record agreement.
4. Windows the two raters disagree on are **discarded**, and the count is
   published in the paper. Do not adjudicate them into one class; the honest
   number is more useful than the extra data.

---

## Session log — fill this in every time

```
session_id      :
date / time     :
subject         :
sensor part no. :
mount           : strap / tape / other
finger          :
operator        :
calibration file:
sampling check  : effective ___ Hz, dropped ___ frames
block timestamps:
  baseline open   ____ - ____
  noisy           ____ - ____   (sub-blocks ____ / ____ / ____)
  attenuated      ____ - ____   (sub-blocks ____ / ____ / ____)
  drifted         ____ - ____   (sub-blocks ____ / ____ / ____)
  spiked          ____ - ____
  baseline close  ____ - ____
anomalies       :
second rater    :            agreement ____%
```

---

## Target

Four members × ~8 min ≈ 32 minutes of recording per round. At 5 s windows with
50% overlap that is roughly **750 windows per round**. Two or three rounds
across different days gives 1,500–2,300 real windows, which is ample for the
cross-domain evaluation and for fine-tuning.

Record on **different days**, not all in one sitting. Same-day recordings share
mount, lighting and skin condition, and a model that generalises across four
people on one afternoon has not been shown to generalise at all.
