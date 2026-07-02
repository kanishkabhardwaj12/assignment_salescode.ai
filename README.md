# Spot the Fake Photo — screen-recapture detector

Given one image, output a score in **[0, 1]**: `0` = real photo, `1` = photo of
a screen ("recapture"). No deep learning — 31 hand-crafted physics-based
features + a small scikit-learn classifier. Small, fast, phone-friendly.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Collect data (~30–60 min)

Take photos with your phone and drop them here:

```
data/
  real/     ~50 normal photos of real things
  screen/   ~50 photos of a phone/laptop/TV screen (and a few printouts)
```

Tips for a robust model: vary lighting (bright/dim), distance (fill the frame
vs. include bezel), angle, screen brightness, and use at least two different
screens. For `real/`, include flat textured things (posters, documents, walls)
— those are the hard negatives.

Transfer HEIC iPhone photos as JPEG (Mail/WhatsApp/AirDrop "Most Compatible"),
or convert: `magick mogrify -format jpg *.heic`.

## 2. Train

```bash
python train.py
```

Extracts features (cached in `features_cache.npz`), reports honest 5-fold
cross-validated accuracy for three small models, picks the best, and saves
`model.pkl` (~a few hundred KB).

## 3. Predict

```bash
python predict.py some_image.jpg
# -> 0.93
```

## 4. Benchmark latency

```bash
python benchmark.py data/real
```

## 5. Optional live demo

```bash
python demo/server.py     # open http://localhost:5001, allow camera
```

Point your webcam at a real object, then at a phone showing a photo, and watch
the score flip.

## How it works

A recaptured image carries physical fingerprints a real photo doesn't:

| Cue | Feature(s) |
|---|---|
| Moiré (sensor grid × pixel grid interference) | isolated peaks in the 2D FFT, radial-spectrum peakiness, spectral flatness — at native resolution and global scale |
| Refresh / rolling-shutter banding | FFT peaks of detrended row/column means |
| Emissive backlight colour | desaturation, blue shift, brightness lift |
| Glass glare | clipped low-saturation blobs |
| Bezel / dark room around the screen | centre-vs-border brightness |
| Double optical path softness | Laplacian variance, Tenengrad, high-pass noise |

`features.py` turns these into a 31-dim vector; `train.py` fits logistic
regression / random forest / gradient boosting and keeps the CV winner.

## Files

```
features.py   feature extraction (the core idea)
train.py      trains + cross-validates + saves model.pkl
predict.py    the one-line predictor (deliverable)
benchmark.py  latency measurement
demo/         live webcam demo (Flask + one HTML page)
NOTE.md       submission note (approach, accuracy, latency, cost)
```
