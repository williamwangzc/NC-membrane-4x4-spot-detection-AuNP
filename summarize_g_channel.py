# -*- coding: utf-8 -*-
"""
summarize_g_channel.py
======================
Collect the per-spot G-channel values from every *_spot_rgb.csv produced by
auto_spot_readout.py and write one summary table.

Input : individual spot CSV files in BASE
Output: _summary_g_channel.csv  (rows = spot 1-16, columns = image names)

Image names are expected as "<concentration>-<batch>", e.g. "10-2", and are
sorted by concentration first, then batch number.
"""

import csv
import glob
import os
import re
from pathlib import Path

# Folder holding the spot CSV files and the output summary.
# Change this to your data folder if the scripts are kept elsewhere.
BASE = Path(__file__).resolve().parent

CONC_ORDER = {"0": 0, "1": 1, "2": 2, "5": 5, "10": 10,
              "20": 20, "50": 50, "100": 100}


def clean_name(stem):
    """Strip the readout suffix, keeping the leading conc-batch prefix."""
    stem = re.sub(r"_v3_spot_rgb$", "", stem)
    stem = re.sub(r"_spot_rgb$", "", stem)
    return stem


def parse_prefix(name):
    """Return (concentration, batch) from a '<conc>-<batch>' name."""
    m = re.match(r"(\d+)-(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 999, 999


def sort_key(file_path):
    name = clean_name(os.path.splitext(os.path.basename(file_path))[0])
    conc, batch = parse_prefix(name)
    return CONC_ORDER.get(str(conc), 999), batch


def main():
    csv_files = sorted(
        glob.glob(os.path.join(BASE, "*_spot_rgb.csv")),
        key=sort_key
    )
    if not csv_files:
        print("No *_spot_rgb.csv files found in", BASE)
        return

    data = {}
    image_names = []

    for fp in csv_files:
        name = clean_name(os.path.splitext(os.path.basename(fp))[0])
        image_names.append(name)
        with open(fp, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spot = int(row["id"])
                data.setdefault(spot, {})[name] = float(row["G"])

    out_path = BASE / "_summary_g_channel.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["spot_id"] + image_names)
        for spot in sorted(data.keys()):
            writer.writerow([f"spot_{spot}"] +
                            [data[spot].get(img, "") for img in image_names])

    print(f"Output: {out_path}")
    print(f"  {len(image_names)} images, {len(data)} spots")


if __name__ == "__main__":
    main()
