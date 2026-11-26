# -*- coding: utf-8 -*-
"""
postprocess_intensity.py
=========================
Post-processing for calibration curve data:
  1. Average the reference spots for each image.
  2. Average the triplicate signal spots per analyte.
  3. Subtract signal from reference to get the final analyte intensity.

The G-channel value decreases as the colorimetric signal strengthens, so
intensity = ref_avg - signal_avg (positive = stronger signal).

Input : _summary_g_channel.csv  (rows = spot 1-16, columns = image names)
Output: _analyte_intensity.csv

Image names are "<concentration>-<batch>"; columns are reordered by batch,
with concentrations ascending within each batch.
"""

import csv
from pathlib import Path

# Folder holding the summary table and the output intensity table.
# Change this to your data folder if the scripts are kept elsewhere.
BASE = Path(__file__).resolve().parent
input_csv = BASE / "_summary_g_channel.csv"
output_csv = BASE / "_analyte_intensity.csv"

# Spot-to-analyte mapping for a 4x4 array
REF_SPOTS = [1, 4, 13, 16]  # Corner reference spots
ANALYTES = {
    "FER": [2, 5, 6],
    "CRP": [3, 7, 8],          # Competitive assay format
    "D-D": [9, 10, 14],
    "cTnI": [11, 12, 15],
}

# Read the G-channel summary
data = {}
with open(input_csv, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    img_cols = headers[1:]  # Skip spot_id column
    for row in reader:
        spot = int(row["spot_id"].replace("spot_", ""))
        data[spot] = {col: float(row[col]) if row[col] else None for col in img_cols}

# Compute reference average, signal average, and intensity per image
results = {}
for col in img_cols:
    ref_vals = [data[s][col] for s in REF_SPOTS if data[s][col] is not None]
    ref_avg = sum(ref_vals) / len(ref_vals) if ref_vals else None
    results[col] = {"ref_avg": ref_avg}
    for analyte, spots in ANALYTES.items():
        sig_vals = [data[s][col] for s in spots if data[s][col] is not None]
        sig_avg = sum(sig_vals) / len(sig_vals) if sig_vals else None
        intensity = ref_avg - sig_avg if (sig_avg is not None and ref_avg is not None) else None
        results[col][f"{analyte}_sig_avg"] = round(sig_avg, 2) if sig_avg else None
        results[col][f"{analyte}_intensity"] = round(intensity, 2) if intensity else None


def parse_col(name):
    conc, batch = name.split("-")
    return int(conc), int(batch)


# Reorder columns: batch ascending, concentration ascending within batch
batches = {}
for col in img_cols:
    conc, batch = parse_col(col)
    batches.setdefault(batch, []).append((conc, col))

ordered_cols = []
for batch in sorted(batches.keys()):
    batches[batch].sort(key=lambda x: x[0])
    ordered_cols.extend([col for _, col in batches[batch]])

# Write output table
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Analyte", "Metric"] + ordered_cols)
    writer.writerow(["Ref", "ref_avg"] + [results[col]["ref_avg"] for col in ordered_cols])
    for analyte in ANALYTES:
        writer.writerow([analyte, "signal_avg"] +
                        [results[col][f"{analyte}_sig_avg"] for col in ordered_cols])
        writer.writerow([analyte, "intensity"] +
                        [results[col][f"{analyte}_intensity"] for col in ordered_cols])

print(f"Output: {output_csv}")
print(f"  {len(img_cols)} images, {len(ANALYTES)} analytes")
