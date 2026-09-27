# PRAHARI — closed-loop sensor integrity guardian

Phase 2 of the Edge-Based Medical Device Self-Diagnosis System.
Dept. of ECE, New Horizon College of Engineering.

Prahari does not diagnose a patient. It diagnoses the **sensor** — deciding
what is wrong with a PPG signal, how badly, repairing what it can, and
refusing to release a heart rate it does not trust.

```
        ESP32 node                          Raspberry Pi 4 hub
 ┌────────────────────────┐          ┌──────────────────────────────┐
 │ PPG  → ADC/I²C @100 Hz │  USB     │ Layer 1  detect the fault    │
 │ 500-sample windows     │ ───────► │ Layer 2  grade the severity  │
 │ 13 features            │  JSON    │ Layer 3  matched restoration │
 │ compressed forest (C)  │  lines   │ Layer 4  trust-gate the HR   │
 └────────────────────────┘          └──────────────────────────────┘
```

---

## Setup — every laptop, once

```bash
git clone <repo-url> && cd prahari
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # 11 tests, all must pass
```

If `pytest` fails on a fresh clone, stop and fix it before writing code. The
tests pin the contracts, and a broken contract is the one bug that costs four
people a week instead of one person an hour.

## First run — no hardware needed

```bash
python -m ml.build_dataset --source bidmc     # falls back offline if PhysioNet is unreachable
python -m ml.train --cv
python -m ml.evaluate
streamlit run hub/dashboard.py                # pick "Corpus" in the sidebar
```

`ml.evaluate` writes five figures and `results.json` to `docs/figures/`.

> **On the offline fallback:** if PhysioNet is unreachable the dataset builder
> generates a physiologically-shaped PPG surrogate so the pipeline still runs.
> Those numbers are for pipeline testing only. **Nothing from the surrogate
> goes in the paper** — the surrogate is cleaner and more regular than real
> PPG, so it inflates accuracy substantially. Only BIDMC numbers are reportable.

---

## Hardware setup

### ESP32 node

1. Arduino IDE → Preferences → Additional Board URLs:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
2. Boards Manager → install **esp32** by Espressif
3. Board: **ESP32 Dev Module**
4. Open `firmware/esp32_node/esp32_node.ino`, set `SENSOR_MODE`, flash
5. Serial Monitor at **921600**

**Wiring — analog sensor** (round Pulse Sensor, TCRT5000 module, most lab stock)

| Sensor | ESP32 |
|---|---|
| VCC | **3V3** — not 5V, the ADC is 3.3 V tolerant only |
| GND | GND |
| OUT | **GPIO34** |

GPIO34 is on ADC1. Do not move it to an ADC2 pin (GPIO 0/2/4/12–15/25–27):
ADC2 stops working the moment Wi-Fi is enabled, and it fails *silently*.

**Wiring — MAX30102 / MAX30105 over I²C**

| Sensor | ESP32 |
|---|---|
| VIN | 3V3 |
| GND | GND |
| SCL | GPIO22 |
| SDA | GPIO21 |

Install **SparkFun MAX3010x** from Library Manager.

### Raspberry Pi hub

```bash
# flash Raspberry Pi OS Lite with SSH pre-enabled from Raspberry Pi Imager
sudo apt update && sudo apt install -y python3-venv python3-pip git
git clone <repo-url> && cd prahari
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo usermod -a -G dialout $USER      # serial access; log out and back in

# remote access for the whole team
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```

---

## Daily commands

```bash
# WP1 — validate sampling BEFORE collecting anything
python -m hub.acquire --port /dev/ttyUSB0 --validate 60

# WP1 — record a session
python -m hub.acquire --port /dev/ttyUSB0 --record data/raw/ramaa_s1.jsonl

# WP3 / WP4 — same code path, no hardware
python -m hub.acquire --replay data/raw/ramaa_s1.jsonl
streamlit run hub/dashboard.py

# everyone, before pushing
pytest -q
```

---

## Layout

```
firmware/esp32_node/    acquisition + (later) on-board inference
hub/contracts.py        FROZEN interfaces — read docs/contracts.md first
hub/preprocess.py       band-pass, normalisation, session calibration
hub/features.py         the 13 features, reference implementation
hub/degrade.py          synthetic faults, severity retained
hub/restore.py          Layer 3 + the non-circular evaluation metrics
hub/hr.py               heart-rate estimation
hub/gate.py             Layer 4 trust gating and the threshold sweep
hub/pipeline.py         the whole chain, one window in, full verdict out
hub/acquire.py          serial reader, replay, sampling validation
hub/dashboard.py        Streamlit live view
ml/build_dataset.py     corpus builder (severity + pre-degradation reference)
ml/train.py             Layer 1 detector + Layer 2 severity regressors
ml/evaluate.py          the five review figures + results.json
docs/contracts.md       what every module promises every other module
docs/protocol.md        data collection protocol — read before recording
```

---

## Three things that will silently ruin the project

1. **Sampling rate drift.** Three of the top four features are frequency
   features. If the node is not at a true 100 Hz, they are computed on the
   wrong axis and nothing downstream warns you. Run `--validate` first, always.

2. **Per-window z-scoring on real data.** It erases the Attenuated class
   exactly, not approximately. The real path must use the session calibration
   from the subject's clean baseline (`hub.preprocess.Calibration`). This is
   why step 0 of the protocol is not a warm-up.

3. **Circular restoration evaluation.** "The classifier now says Clean" proves
   nothing when the classifier is the judge. Always lead with correlation
   against the pre-degradation window, SNR gain, or HR-error reduction.

## Work packages

| WP | Owner | Owns |
|---|---|---|
| WP1 | Ramaa | acquisition, Pi, protocol, data collection |
| WP2 | Member B | model compression, C generation, parity, footprint |
| WP3 | Member C | severity regression, trust index, restoration |
| WP4 | Member D | heart rate, gating, dashboard, figure pack |
