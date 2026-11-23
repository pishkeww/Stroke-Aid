# Prehospital Stroke Triage Aid

An offline, camera-based decision-support prototype for ambulances. It analyses a patient's face and eyes in real time, combines the result with manually entered RACE items, and suggests whether to route to a CT or thrombectomy-capable centre based on last known well (LKW) time.

> **Research prototype. Not a medical device. Not clinically validated.**
> All thresholds are engineering starting points. It has not been tested on real patients and must not be used to make clinical decisions. Advisory output only.

## What it does

- **Facial asymmetry:** resting mouth droop, smile asymmetry, brow raise and eye closure, measured in a roll- and scale-invariant face frame.
- **Central vs peripheral pattern:** lower-face weakness with upper-face sparing (central, stroke-like) vs whole-hemiface weakness (peripheral, Bell's-palsy-like). A "central" call is only made after the guided brow and eye tasks.
- **Head pose and gaze:** head turn, signed gaze deviation, eyelid opening.
- **Face dynamics:** ptosis asymmetry, blink rate, blink lag between eyes, smile onset lag. Reported only, not yet scored.
- **RACE fusion:** the camera supplies facial palsy and head/gaze deviation. The paramedic enters arm, leg and speech. Output is a LOW, ELEVATED, HIGH or INSUFFICIENT_DATA level.
- **LKW and routing:** treatment-window arithmetic, direct-to-thrombectomy vs drip-and-ship vs nearest CT, with configurable destinations and travel times.
- **Camera sources:** built-in or USB webcam, phone camera (via a phone-as-webcam app or an IP-camera stream URL), or a video file. Rotate and mirror options for phones.
- **Fully offline:** MediaPipe FaceMesh runs locally. No network needed.

## Install

Requires Python 3.11 on Windows (other platforms are untested).

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Or just double-click `run.bat`, which creates the environment on first run.

## Run

```
run.bat
```

or

```
python main_app.py
python main_app.py --source 1
python main_app.py --source http://192.168.42.129:8080/video --rotate 90
python main_app.py --video clip.mp4
python main_app.py --lkw-min-ago 45 --lat 25.6 --lon 85.1
python main_app.py --list-cameras
python main_app.py --headless
```

### Using the app

1. Seat the patient facing the camera, well lit, head roughly frontal.
2. Press the facial exam button (key **E** in headless mode). It guides rest, smile, brow raise and eye closure.
3. Enter arm, leg and speech scores in the dropdowns (keys **A / L / V** in headless mode).
4. Set last known well, either a clock time or minutes ago.
5. Read the LVO level and routing suggestion. **R** resets, **Q** quits.

### Phone as a camera

- **USB:** use a phone-as-webcam app. It then appears as a normal camera index.
- **Network:** use an IP-camera app and pass its URL with `--source`. USB tethering or a phone hotspot gives a local network with no internet.
- Bluetooth does not carry live video at usable quality.
- If left and right look swapped, add `--mirror`. For a phone held upright, add `--rotate 90` or `270`.

## Configure destinations

Edit `destinations.json` with real facilities: location, whether CT, thrombolysis and thrombectomy are available, and travel assumptions. The shipped entries are **examples** and the app warns about that.

## Healthy-face self-test

Measure how much each metric varies on healthy faces before trusting a threshold:

```
selftest.bat run --subject H01 --condition normal --reps 3
selftest.bat report
```

The report shows metric spread, threshold headroom, false-positive rates with confidence intervals, repeat-test variation, and resting droop against head yaw.

## Tests

```
pip install -r requirements-dev.txt
pytest -q
```

The tests use synthetic landmarks with known ground truth, so no camera is needed.

## Build a Windows exe

```
build_exe.bat
```

Runs the tests, then builds `dist\StrokeTriage\StrokeTriage.exe` with PyInstaller.

## Project layout

| File | Role |
|---|---|
| `main_app.py` | PyQt5 GUI and command-line entry point |
| `tracker.py` | OpenCV-window mode (headless) |
| `capture.py` | Camera, stream and file sources with reconnect, rotate and mirror |
| `feature_extractor.py` | Landmarks to head pose, gaze, eye openness |
| `facial_analysis.py` | Guided facial exam, asymmetry, central vs peripheral pattern |
| `face_dynamics.py` | Blink, ptosis and smile-onset measures |
| `scoring.py` | Smoothing, event detection, NIHSS-style item scores |
| `fusion.py` | RACE-aligned LVO level |
| `routing.py` | LKW clock and destination recommendation |
| `visualization.py` | On-screen overlays |
| `selftest.py`, `metrics.py` | Healthy-face noise-floor tool and statistics helpers |
| `version.py` | Version and source fingerprint |

## Known limitations

- Not validated on patients. Thresholds are unvalidated and need tuning on labelled data.
- Cannot detect all strokes. A normal result does not exclude stroke.
- Sensitive to lighting, camera angle, glasses, facial hair and vehicle motion.
- Arm, leg and speech are manual inputs.
- Routing uses straight-line distance unless you enter travel times.
