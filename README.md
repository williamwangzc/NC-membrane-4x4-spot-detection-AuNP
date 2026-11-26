# Automated Spot Readout Algorithm for Paper-Based Multiplex Immunoassays

Rule-based image analysis pipeline for quantitative readout of colorimetric
(colloidal gold) spots on wax-printed nitrocellulose membranes. It detects a
4×4 spot array, refines each spot center and radius, extracts RGB channel
intensity, and reduces the triplicate signal spots against the corner reference
spots.

## Dependencies

- Python 3.8+
- OpenCV (opencv-python >= 4.5)
- NumPy

## Files

- `auto_spot_readout.py` — Detection: coarse grid localization, per-spot
  refinement, neighbor-trend rescue, and RGB intensity extraction.
- `summarize_g_channel.py` — Collects the G-channel values from all spot CSV
  files into one summary table.
- `postprocess_intensity.py` — Reference subtraction, triplicate averaging, and
  final analyte intensity table.

## Usage

### 1. Detect spots

Single image:

```bash
python auto_spot_readout.py path/to/image.jpg
```

Batch folder:

```bash
python auto_spot_readout.py path/to/image_folder/
```

Running the script with no path opens a file dialog.

### 2. Summarize the readout

After the detection step, run:

```bash
python summarize_g_channel.py
```

This collects every `*_spot_rgb.csv` in the script folder into
`_summary_g_channel.csv`.

### 3. Post-process intensities

```bash
python postprocess_intensity.py
```

This averages the four reference spots and the triplicate signal spots per
analyte, and writes `_analyte_intensity.csv`. Intensity is defined as
`ref_avg - signal_avg`, because the G-channel value decreases as the colorimetric
signal strengthens.

## Algorithm Overview

1. **Coarse localization (Stage A):** Hough circle detection on a
   bilateral-filtered grayscale image, followed by a 4×4 grid fit over the
   candidate pool.

2. **Per-spot refinement (Stage B):** Each grid anchor is refined independently:
   - Local Hough circle re-detection to correct the center position.
   - Radial edge profiling along 24 rays to measure the spot radius (brightness
     method first, redness channel `R - (G+B)/2` as fallback for fully
     purple-red spots).
   - Least-squares circle fitting (Kasa method) for robust center estimation.

3. **Neighbor-trend rescue (Stage C, v3):** Spots that fail Stage B are
   corrected from the anchor-to-refined offset trend of their refined neighbors
   on the same row and column. Row and column predictions are weighted by the
   number of neighbors, the shift is capped at 1.5× the median grid radius, and
   the corrected anchor is refined once more. Spots that already refined
   successfully are never recomputed.

4. **Readout:** Green-channel mean within a 10% inward-shrunk ROI, so wax edge
   and background texture do not contaminate the value.

Spots are numbered automatically from the top-left corner, row-major order.

## Spot Layout (4×4 Array)

- Reference spots: positions 1, 4, 13, 16 (corners)
- FER: positions 2, 5, 6
- CRP: positions 3, 7, 8 (competitive assay format)
- D-D: positions 9, 10, 14
- cTnI: positions 11, 12, 15

## Output

For each input image, the detection script generates:

- `*_v3_result.png` — Result image with green ROI circles and spot numbering.
- `*_v3_compare.png` — Side-by-side view: red circles (uniform grid radius) vs
  green circles (per-spot refined, shrunk ROI).
- `*_hough_candidate_preview.png` — Raw Hough candidate detections.
- `*_v3_spot_rgb.csv` — Per-spot RGB channel values.

The summary and post-processing steps produce:

- `_summary_g_channel.csv` — G-channel table, rows = spots, columns = images.
- `_analyte_intensity.csv` — Reference and signal averages, and final analyte
  intensities.

### Per-spot CSV fields

| Field     | Meaning |
|-----------|---------|
| `id`      | Spot number, top-left row-major order. |
| `x`, `y`  | Refined center coordinates (px). |
| `used_r`  | Refined spot radius (px). |
| `refined` | `1` = center/radius from edge refinement; `0` = grid anchor fallback. |
| `rescue`  | `0` = refined directly; `1` = rescued from neighbor trend and refined; `2` = rescued anchor kept, second refinement still failed. |
| `brightness` | Mean of R, G, B. |
| `R`, `G`, `B` | Per-channel mean within the readout ROI. |
