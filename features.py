"""Hand-crafted features for screen-recapture detection.

A photo OF a screen differs from a real photo in physically measurable ways:

1. Moire: the camera's sensor grid beats against the screen's pixel grid,
   producing quasi-periodic interference patterns -> sharp, isolated peaks
   in the 2D Fourier spectrum that natural images almost never have.
2. Banding: screen refresh / PWM backlight + rolling shutter produce
   horizontal or vertical intensity bands -> peaks in the FFT of row/column
   means.
3. Colour: an emissive backlit panel gives a slightly desaturated,
   blue-shifted, brightness-lifted image compared to reflected light.
4. Glare: specular reflections on glass show up as clipped, low-saturation
   bright blobs.
5. Context: recaptures often include the screen bezel / dark surroundings
   at the image border while the centre is bright.
6. Detail: recaptures are re-imaged through two optical systems -> softer
   fine detail relative to their resolution.

extract_features(img_bgr) returns a fixed-length float vector of these cues.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

_FFT_SIZE = 512  # patch side used for spectral analysis


def _hann2d(n):
    w = np.hanning(n).astype(np.float32)
    return np.outer(w, w)


_HANN = _hann2d(_FFT_SIZE)


def _center_crop(gray, size):
    h, w = gray.shape
    if h < size or w < size:
        gray = cv2.copyMakeBorder(
            gray,
            max(0, (size - h) // 2), max(0, (size - h + 1) // 2),
            max(0, (size - w) // 2), max(0, (size - w + 1) // 2),
            cv2.BORDER_REFLECT,
        )
        h, w = gray.shape
    y0, x0 = (h - size) // 2, (w - size) // 2
    return gray[y0:y0 + size, x0:x0 + size]


def _radial_profile(power):
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    total = np.bincount(r.ravel(), power.ravel())
    count = np.bincount(r.ravel())
    return total / np.maximum(count, 1)


def _fft_features(gray_patch, prefix):
    """Six spectral cues from one grayscale patch (values, names)."""
    n = gray_patch.shape[0]
    patch = gray_patch.astype(np.float32)
    patch -= patch.mean()
    spec = np.fft.fftshift(np.fft.fft2(patch * _HANN[:n, :n]))
    power = np.abs(spec) ** 2
    logmag = np.log1p(np.abs(spec)).astype(np.float32)

    cy = cx = n // 2
    yy, xx = np.indices(power.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    prof = _radial_profile(power)
    rmax = n // 2
    mid = prof[int(0.08 * rmax):int(0.80 * rmax)]
    mid = mid[mid > 0]
    if mid.size == 0:
        mid = np.array([1e-12], dtype=np.float32)

    # 1. peakiness of the radial spectrum (moire rings/spokes)
    peak_ratio = float(np.log1p(mid.max() / (np.median(mid) + 1e-12)))

    # 2. high-frequency energy share
    total_e = power[r > 4].sum() + 1e-12
    hf_ratio = float(power[r > 0.5 * rmax].sum() / total_e)

    # 3. spectral flatness of the mid band (peaks -> low flatness)
    flatness = float(np.exp(np.mean(np.log(mid + 1e-12))) / (mid.mean() + 1e-12))

    # 4. energy concentrated on the frequency axes (pixel-grid aliasing)
    axis_mask = ((np.abs(xx - cx) <= 2) | (np.abs(yy - cy) <= 2)) & (r > 8)
    axis_ratio = float(power[axis_mask].sum() / total_e)

    # 5 & 6. isolated 2D outlier peaks in the log spectrum
    smooth = cv2.medianBlur(logmag, 5)
    resid = logmag - smooth
    band = (r > 0.08 * rmax) & (r < 0.9 * rmax)
    rb = resid[band]
    thr = rb.mean() + 4.0 * rb.std()
    n_peaks = float(np.log1p((rb > thr).sum()))
    max_peak = float((rb.max() - rb.mean()) / (rb.std() + 1e-12))

    names = [f"{prefix}_{s}" for s in
             ("peak_ratio", "hf_ratio", "flatness", "axis_ratio",
              "n_peaks", "max_peak")]
    return [peak_ratio, hf_ratio, flatness, axis_ratio, n_peaks, max_peak], names


def _multi_patch_fft_agg(gray_full):
    """Aggregate FFT cues across a 3x3 grid to catch local moire patches."""
    h, w = gray_full.shape
    ys = [0, max(0, h // 2 - _FFT_SIZE // 2), max(0, h - _FFT_SIZE)]
    xs = [0, max(0, w // 2 - _FFT_SIZE // 2), max(0, w - _FFT_SIZE)]

    patch_vecs = []
    for y0 in ys:
        for x0 in xs:
            p = gray_full[y0:y0 + _FFT_SIZE, x0:x0 + _FFT_SIZE]
            if p.shape[0] != _FFT_SIZE or p.shape[1] != _FFT_SIZE:
                p = _center_crop(p, _FFT_SIZE)
            v, _ = _fft_features(p, "tmp")
            patch_vecs.append(v)

    arr = np.asarray(patch_vecs, dtype=np.float32)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    maxs = arr.max(axis=0)
    vals = np.concatenate([means, stds, maxs]).tolist()

    base = ["peak_ratio", "hf_ratio", "flatness", "axis_ratio", "n_peaks", "max_peak"]
    names = [f"fft_grid_mean_{b}" for b in base]
    names += [f"fft_grid_std_{b}" for b in base]
    names += [f"fft_grid_max_{b}" for b in base]
    return vals, names


def _banding_features(gray):
    """Refresh/rolling-shutter banding: periodicity of row & column means."""
    vals, names = [], []
    for axis, tag in ((1, "row"), (0, "col")):
        m = gray.mean(axis=axis).astype(np.float32)
        if m.size > 1024:
            m = cv2.resize(m[None, :], (1024, 1), cv2.INTER_AREA).ravel()
        trend = cv2.GaussianBlur(m[None, :], (31, 1), 0).ravel()
        d = (m - trend) * np.hanning(m.size).astype(np.float32)
        p = np.abs(np.fft.rfft(d)) ** 2
        band = p[3:len(p) // 2]
        ratio = float(np.log1p(band.max() / (np.median(band) + 1e-12))) if band.size else 0.0
        vals.append(ratio)
        names.append(f"band_{tag}_peak")
    return vals, names


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def extract_features(img_bgr):
    """Return (vector, names) of recapture cues for one BGR image."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("empty image")

    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # keep spectral analysis at native resolution (resizing destroys moire)
    vals, names = [], []
    v, n = _fft_features(_center_crop(gray_full, _FFT_SIZE), "fft_c")
    vals += v; names += n

    # second look at global scale: whole image squeezed to the FFT size
    small_gray = cv2.resize(gray_full, (_FFT_SIZE, _FFT_SIZE),
                            interpolation=cv2.INTER_AREA)
    v, n = _fft_features(small_gray, "fft_g")
    vals += v; names += n

    # Local moire may appear in only part of the frame; aggregate over a grid.
    v, n = _multi_patch_fft_agg(gray_full)
    vals += v; names += n

    v, n = _banding_features(gray_full)
    vals += v; names += n

    # ---- colour statistics on a small copy --------------------------------
    small = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    b, g, rch = [small[..., i].astype(np.float32).mean() for i in range(3)]
    lum = (b + g + rch) / 3 + 1e-6

    vals += [
        float(s.mean() / 255), float(s.std() / 255),
        float(((s < 30) & (val > 60)).mean()),
        float(val.mean() / 255), float(val.std() / 255),
        float((b - rch) / lum),                      # blue-shift of backlight
        float(np.std([b, g, rch]) / lum),            # global colour cast
    ]
    names += ["sat_mean", "sat_std", "low_sat_frac", "val_mean", "val_std",
              "blue_cast", "cast_strength"]

    # ---- glare -------------------------------------------------------------
    clip = ((val > 250) & (s < 40)).astype(np.uint8)
    clip_frac = float(clip.mean())
    if clip_frac > 0:
        n_lbl, lbl = cv2.connectedComponents(clip)
        counts = np.bincount(lbl.ravel(), minlength=n_lbl)
        blob = float(counts[1:].max()) / clip.size if n_lbl > 1 else 0.0
    else:
        blob = 0.0
    vals += [clip_frac, float(blob)]
    names += ["clip_frac", "glare_blob"]

    # ---- border vs centre brightness (bezel / dark room around screen) ----
    vs = cv2.resize(gray_full, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    border = np.concatenate([vs[:12].ravel(), vs[-12:].ravel(),
                             vs[12:-12, :12].ravel(), vs[12:-12, -12:].ravel()])
    center = vs[32:-32, 32:-32]
    vals += [
        float((center.mean() - border.mean()) / 255),
        float((border < 40).mean()),
    ]
    names += ["center_minus_border", "border_dark_frac"]

    # ---- sharpness / fine detail ------------------------------------------
    crop = _center_crop(gray_full, _FFT_SIZE).astype(np.float32)
    lap = cv2.Laplacian(crop, cv2.CV_32F)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1)
    hp = crop - cv2.GaussianBlur(crop, (7, 7), 0)
    vals += [
        float(np.log1p(lap.var())),
        float(np.log1p((gx ** 2 + gy ** 2).mean())),
        float(np.log1p(hp.std())),
    ]
    names += ["lap_var", "tenengrad", "noise_std"]

    return np.asarray(vals, dtype=np.float32), names
