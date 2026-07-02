# Submission note — Spot the Fake Photo

## Approach

A photo of a screen differs from a real photo in *physics*, not content, so I
skipped deep learning and extracted 31 hand-crafted features that measure
those physical differences directly: **moiré** (the camera sensor grid beating
against the screen pixel grid shows up as sharp isolated peaks in the 2D
Fourier spectrum — computed at native resolution, where moiré lives, and at
global scale), **refresh/rolling-shutter banding** (FFT peaks of row/column
means), **backlight colour** (desaturation, blue shift, lifted brightness),
**glare** (clipped low-saturation blobs), **bezel context** (bright centre vs.
dark border), and **double-optical-path softness** (Laplacian variance,
high-pass noise). A small scikit-learn classifier (best of logistic
regression / random forest / gradient boosting by 5-fold CV) maps the vector
to a probability. The whole model is a few hundred KB.

## Accuracy (honest)

- 5-fold cross-validated accuracy on my ~100 phone photos: **__._%**
  <!-- fill in from train.py output -->
- Caveat: my data is one phone + a few screens; held-out data from other
  devices will likely score a few points lower. The features are
  device-agnostic by design (frequency + colour physics), which should limit
  the drop.

## Latency

- **~__ ms per image** (decode + features + inference), measured on
  <!-- fill from benchmark.py, e.g. "Apple M2 laptop CPU" -->. Single-threaded,
  no GPU. Model load is a one-off ~__ ms.

## Cost per image

- **On-device: $0.** The model is ~KB-scale and the features are one FFT +
  a few filters — well within a phone CPU (est. <100 ms on a mid-range
  Android via OpenCV; no network, works offline, no image leaves the device).
- **Cloud (if server-side)**: one vCPU at ~__ ms/image ⇒ ~__ images/sec/core.
  A $30/month 2-vCPU instance ⇒ roughly **$0.01–0.03 per 1,000 images**
  (≈ $10–30 per million) at moderate utilisation, dominated by instance cost,
  not compute. Recommendation: run on-device, send only low-confidence cases
  (score near the threshold) to the server for a second opinion.

## With more time

1. More data: many phones × many screens (OLED/LCD, 60–120 Hz), printouts,
   and hard negatives (textiles, window screens, halftone-printed posters —
   they alias too).
2. Patch-level voting: score several crops and aggregate, so a screen filling
   only part of the frame is still caught.
3. A tiny CNN (~50 KB, quantised) trained on high-pass residuals as a second
   ensemble member — catches recaptures where moiré is defeated.

## The experienced-candidate questions

- **As cheaters adapt** (defocus, distance, high-DPI screens to kill moiré):
  no single cue is load-bearing — colour, banding, glare and softness still
  fire when moiré doesn't. Operationally: log scores server-side, review the
  borderline band weekly, and retrain on confirmed fraud — the feature
  pipeline stays, only the ~100 KB classifier weights ship as an update. The
  defence is a moving target, not one release.
- **Tiny & fast on a phone**: the features are OpenCV/NEON-friendly (one
  512×512 FFT, Sobel/Laplacian, colour stats); port the extractor to C++ or
  reimplement in ~200 lines with vDSP/RenderScript, export the classifier as
  pure if-else trees (no runtime dependency). Budget: <5 MB, <50 ms.
- **Choosing the cut-off**: it's a cost trade-off, not a fixed 0.5. Estimate
  the base rate of fraud, assign costs (false accusation ≫ missed cheat for
  UX; reversed for payouts), and pick the threshold minimising expected cost
  on a validation ROC. Practically: two thresholds — auto-flag above ~0.9,
  silent-log/second-check in the 0.5–0.9 band — keeps false accusations rare
  while still collecting retraining signal.
