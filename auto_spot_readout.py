# -*- coding: utf-8 -*-
"""
auto_spot_readout.py
====================
Automated spot detection and RGB intensity extraction for 4x4 multiplex
colorimetric (colloidal gold) immunoassay arrays on nitrocellulose membranes.

Pipeline
  Stage A  Coarse localization: Hough circle detection on a bilateral-filtered
           grayscale image, followed by a 4x4 grid fit over the candidate pool.
  Stage B  Per-spot refinement: local Hough re-detection for center correction,
           radial edge profiling for the spot radius, and a least-squares circle
           fit for robust center estimation.
  Stage C  Neighbor-trend rescue (v3): spots that fail Stage B are corrected
           from the row/column offset trend of their refined neighbors, then
           refined once more.
  Readout  Green-channel mean within a 10% inward-shrunk ROI, excluding wax
           edge and background artifacts.

Usage
  Single image (command line):
      python auto_spot_readout.py path/to/image.jpg
  Batch folder:
      python auto_spot_readout.py path/to/image_folder/
  Interactive: run directly; a file dialog opens when no path is given.
"""

import cv2
import numpy as np
import csv
import os
import sys

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except Exception:
    HAS_TK = False

# ===================== Parameters =====================

# ---- Stage A: coarse Hough circle detection ----
bilateral_d = 9
bilateral_sigmaColor = 75
bilateral_sigmaSpace = 75

dp = 1.2
minDist = 72
param1 = 60
param2 = 15
minRadius = 32
maxRadius = 68

grid_tol = 30
grid_min_match = 12

# ---- Stage B: per-spot refinement ----
refine_enabled = True         # Master switch; set False to use grid anchors only
num_rays = 24                 # Number of radial rays for edge profiling
ray_scan_max = 1.35           # Ray scan end = this factor x grid radius
local_hough_param2 = 20       # Local Hough param2 (stricter than global)
refine_center_iters = 3       # Center refinement iterations
refine_center_max_shift = 20  # Max single center move (px), prevents jumping to an adjacent spot
refine_r_ratio_lo = 0.80      # Min refined radius, relative to grid median
refine_r_ratio_hi = 1.20      # Max refined radius, relative to grid median
refine_min_valid_rays = 0.5   # Fall back to grid radius if valid rays are below this fraction
edge_shrink = 0.90            # Readout ROI shrink factor (1.0 = no shrink; 0.90 = 10% inward)

# ---- Stage C: neighbor-trend rescue for failed spots (v3) ----
# When a spot fails Stage B, the anchor position is re-estimated from the
# anchor-to-refined offset trend of already-refined neighbors on the same row
# and column. The corrected anchor is refined once more.
neighbor_fix_enabled = True
neighbor_fix_max_shift = 1.5  # Predicted offset cap = this factor x median grid radius
# =======================================================


def imread_unicode(path):
    """Image read that supports non-ASCII (Chinese) paths on Windows."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    """Image write that supports non-ASCII (Chinese) paths on Windows."""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def measure_edge_distances(gray_img, red_img, cx, cy, r_ref):
    """
    Scan radial profiles from (cx, cy) along num_rays rays and locate the
    spot-to-background boundary on each ray.

    Two-channel strategy:
      1. Brightness method (primary): find the brightness peak (white ring or
         white center), then scan outward to where it drops to the midpoint
         between peak and background. Works as long as a white ring surrounds
         the purple-red signal center.
      2. Redness method (fallback): if the brightness method fails (spot fully
         red-purple with no white ring), use the redness profile
         R - (G+B)/2 and find where redness drops from peak to background.

    Returns a list of (angle, boundary_distance) pairs.
    """
    h, w = gray_img.shape
    smooth_g = cv2.boxFilter(gray_img, ddepth=-1, ksize=(3, 3))
    smooth_r = cv2.boxFilter(red_img, ddepth=-1, ksize=(3, 3))

    results = []
    for k in range(num_rays):
        ang = 2.0 * np.pi * k / num_rays
        dx = np.cos(ang)
        dy = np.sin(ang)
        prof = []
        t = r_ref * 0.40
        t_end = r_ref * ray_scan_max
        step = 1.0
        while t <= t_end:
            xi = int(round(cx + t * dx))
            yi = int(round(cy + t * dy))
            if 0 <= xi < w and 0 <= yi < h:
                prof.append((t, float(smooth_g[yi, xi]), float(smooth_r[yi, xi])))
            t += step
        if len(prof) < 8:
            continue
        vals_g = [p[1] for p in prof]
        vals_r = [p[2] for p in prof]

        # ---- Brightness method ----
        cross_t = -1.0
        peak_idx = int(np.argmax(vals_g))
        peak_val = vals_g[peak_idx]
        bg_val = vals_g[-1]
        thr = (peak_val + bg_val) * 0.5
        if peak_val - bg_val >= 12.0:
            for i in range(peak_idx, len(prof)):
                if vals_g[i] < thr:
                    cross_t = prof[i][0]
                    break

        # ---- Redness method (fallback) ----
        if cross_t <= 0:
            r_peak_idx = int(np.argmax(vals_r))
            r_peak = vals_r[r_peak_idx]
            r_bg = vals_r[-1]
            thr_r = (r_peak + r_bg) * 0.5
            if r_peak - r_bg >= 15.0:
                for i in range(r_peak_idx, len(prof)):
                    if vals_r[i] < thr_r:
                        cross_t = prof[i][0]
                        break

        if cross_t > 0:
            results.append((ang, cross_t))
    return results


def refine_spot(gray_img, red_img, cx, cy, r_grid):
    """
    Refine a single grid anchor point.
    Returns (refined_cx, refined_cy, refined_r, ok_flag).

    Step 1: local Hough circle re-detection in a neighborhood window to correct
            the grid-extrapolated center.
    Step 2: radial edge profiling to measure the spot radius, with a
            least-squares circle fit for robust center refinement.

    Falls back to the grid values if refinement fails.
    """
    # ---------- Step 1: local Hough center correction ----------
    h, w = gray_img.shape
    win = int(round(1.6 * r_grid))
    x0 = max(0, int(cx) - win)
    y0 = max(0, int(cy) - win)
    x1 = min(w, int(cx) + win)
    y1 = min(h, int(cy) + win)
    patch = gray_img[y0:y1, x0:x1]
    local_circles = None
    if patch.size > 0:
        local_circles = cv2.HoughCircles(
            patch, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=r_grid,
            param1=60, param2=local_hough_param2,
            minRadius=int(round(0.7 * r_grid)),
            maxRadius=int(round(1.3 * r_grid))
        )
    if local_circles is not None:
        # Keep the candidate closest to the anchor (add back the crop offset)
        best_c = None
        best_d = 1e9
        for c in local_circles[0]:
            gcx = c[0] + x0
            gcy = c[1] + y0
            d = np.hypot(gcx - cx, gcy - cy)
            if d < best_d:
                best_d = d
                best_c = (gcx, gcy)
        if best_c is not None and best_d < grid_tol * 0.8:
            cxx, cyy = float(best_c[0]), float(best_c[1])
        else:
            cxx, cyy = float(cx), float(cy)
    else:
        cxx, cyy = float(cx), float(cy)

    # ---------- Step 2: radial profiling + iterative center refinement ----------
    r_ref = float(r_grid)
    ok = False
    for _ in range(refine_center_iters):
        samples = measure_edge_distances(gray_img, red_img, cxx, cyy, r_ref)
        if len(samples) < num_rays * refine_min_valid_rays:
            break
        d_arr = np.array([s[1] for s in samples])
        ang_arr = np.array([s[0] for s in samples])
        r_new = float(np.median(d_arr))
        r_new = max(r_grid * refine_r_ratio_lo, min(r_grid * refine_r_ratio_hi, r_new))
        # Least-squares circle fit (Kasa method) for a robust center estimate
        ex = cxx + d_arr * np.cos(ang_arr)
        ey = cyy + d_arr * np.sin(ang_arr)
        if len(ex) >= 8:
            A = np.column_stack([ex, ey, np.ones_like(ex)])
            b = -(ex ** 2 + ey ** 2)
            try:
                D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
                new_cx = -D / 2.0
                new_cy = -E / 2.0
            except np.linalg.LinAlgError:
                new_cx, new_cy = float(ex.mean()), float(ey.mean())
        else:
            new_cx, new_cy = float(ex.mean()), float(ey.mean())
        shift = np.hypot(new_cx - cxx, new_cy - cyy)
        if shift > refine_center_max_shift:
            scale = refine_center_max_shift / shift
            new_cx = cxx + (new_cx - cxx) * scale
            new_cy = cyy + (new_cy - cyy) * scale
        cxx, cyy = new_cx, new_cy
        r_ref = r_new
        ok = True
        if shift < 0.5:
            break

    # Final radius measurement after convergence
    if ok:
        samples = measure_edge_distances(gray_img, red_img, cxx, cyy, r_ref)
        if len(samples) >= num_rays * refine_min_valid_rays:
            r_final = float(np.median([s[1] for s in samples]))
            r_final = max(r_grid * refine_r_ratio_lo, min(r_grid * refine_r_ratio_hi, r_final))
            return cxx, cyy, r_final, True
    return float(cx), float(cy), float(r_grid), False


def predict_offset_along(refine_results, grid_points, l0, k0, axis):
    """
    Estimate the anchor-to-refined offset of a target spot along its row or
    column, using only neighbors that refined successfully.

    axis='row': fixed row l0, vary column index k.
    axis='col': fixed column k0, vary row index l.

    With two or more neighbors, their offsets are fit by a 1-D linear trend
    and evaluated at the target index; with a single neighbor, the mean
    offset is used. Returns (dx, dy, n_neighbors) or None.
    """
    samples = []
    for idx in range(len(grid_points)):
        if not refine_results[idx]["ok"]:
            continue
        l = idx // 4
        k = idx % 4
        if axis == "row":
            if l != l0:
                continue
            t = k
        else:
            if k != k0:
                continue
            t = l
        gx, gy = grid_points[idx]
        dx = refine_results[idx]["rx"] - gx
        dy = refine_results[idx]["ry"] - gy
        samples.append((t, dx, dy))

    if not samples:
        return None

    tgt = k0 if axis == "row" else l0
    t_arr = np.array([s[0] for s in samples], dtype=np.float64)
    dx_arr = np.array([s[1] for s in samples], dtype=np.float64)
    dy_arr = np.array([s[2] for s in samples], dtype=np.float64)

    if len(samples) >= 2 and (t_arr.max() - t_arr.min()) >= 1:
        kx = np.polyfit(t_arr, dx_arr, 1)
        ky = np.polyfit(t_arr, dy_arr, 1)
        return float(np.polyval(kx, tgt)), float(np.polyval(ky, tgt)), len(samples)

    return float(dx_arr.mean()), float(dy_arr.mean()), 1


def process_image(img_path):
    """Run the full detection pipeline on one image and write its outputs."""
    img_bgr = imread_unicode(img_path)
    if img_bgr is None:
        print(f"  Error: cannot read {img_path}")
        return

    img_dir = os.path.dirname(os.path.abspath(img_path))
    img_name, _ = os.path.splitext(os.path.basename(img_path))

    out_img = os.path.join(img_dir, f"{img_name}_v3_result.png")
    candidate_preview = os.path.join(img_dir, f"{img_name}_hough_candidate_preview.png")
    compare_img = os.path.join(img_dir, f"{img_name}_v3_compare.png")
    out_csv = os.path.join(img_dir, f"{img_name}_v3_spot_rgb.csv")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)

    # Redness channel, fallback for fully purple-red spots
    R_ch = img_rgb[:, :, 0].astype(np.float32)
    G_ch = img_rgb[:, :, 1].astype(np.float32)
    B_ch = img_rgb[:, :, 2].astype(np.float32)
    red_img = R_ch - 0.5 * (G_ch + B_ch)

    # ---- Stage A: Hough circles + 4x4 grid fitting ----
    circles = cv2.HoughCircles(
        gray_filtered,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    candidates = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cx, cy, r = c
            candidates.append([float(cx), float(cy), float(r)])
    print(f"  Hough candidate pool size = {len(candidates)}")

    candidate_canvas = img_bgr.copy()
    for (cx, cy, r) in candidates:
        cv2.circle(candidate_canvas, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    imwrite_unicode(candidate_preview, candidate_canvas)

    # Adaptive grid search over every candidate pair
    best_match_count = 0
    best_grid_param = None
    pts_arr = np.array(candidates, dtype=np.float32)
    xy_pts = pts_arr[:, :2]

    if len(pts_arr) >= 2:
        n = len(pts_arr)
        for i in range(n):
            x1, y1, _ = pts_arr[i]
            for j in range(i + 1, n):
                x2, y2, _ = pts_arr[j]
                dx0 = x2 - x1
                dy0 = y2 - y1
                dist0 = np.hypot(dx0, dy0)
                if dist0 < 40:
                    continue
                u = np.array([dx0, dy0])
                v = np.array([dy0, -dx0])
                mat = np.stack([u, v], axis=1)
                try:
                    mat_inv = np.linalg.inv(mat)
                except np.linalg.LinAlgError:
                    continue
                delta = xy_pts - np.array([x1, y1])
                g_coords = delta @ mat_inv
                gl_all = g_coords[:, 0]
                gk_all = g_coords[:, 1]
                L_min = int(np.floor(np.min(gl_all)))
                L_max = int(np.ceil(np.max(gl_all)))
                K_min = int(np.floor(np.min(gk_all)))
                K_max = int(np.ceil(np.max(gk_all)))
                for L0 in range(L_min, L_max - 2):
                    for K0 in range(K_min, K_max - 2):
                        match = 0
                        for l in range(L0, L0 + 4):
                            for k in range(K0, K0 + 4):
                                t_pt = np.array([x1, y1]) + l * u + k * v
                                d_all = np.hypot(xy_pts[:, 0] - t_pt[0], xy_pts[:, 1] - t_pt[1])
                                if np.min(d_all) < grid_tol:
                                    match += 1
                        if match > best_match_count:
                            best_match_count = match
                            best_grid_param = (np.array([x1, y1]), u, v, L0, K0)

    final_spots = []
    draw_img = img_bgr.copy()
    compare_canvas = img_bgr.copy()

    if best_grid_param is None or best_match_count < grid_min_match:
        print(f"  Grid match below threshold ({grid_min_match}); no valid output.")
        imwrite_unicode(out_img, draw_img)
        return

    origin, u, v, L0, K0 = best_grid_param
    grid_points = []
    for l in range(L0, L0 + 4):
        for k in range(K0, K0 + 4):
            grid_points.append(origin + l * u + k * v)
    grid_points = np.array(grid_points)

    r_list = [p[2] for p in candidates]
    median_r = int(np.median(r_list))
    print(f"  Grid solved, median radius = {median_r}")

    # Red circles: old uniform-radius placement, kept on the comparison image
    for (gx, gy) in grid_points:
        cv2.circle(compare_canvas, (int(gx), int(gy)), median_r, (0, 0, 255), 2)

    # First pass: refine every spot once, record results without drawing
    refine_results = []
    refined_count = 0
    for idx, (gx, gy) in enumerate(grid_points):
        if refine_enabled:
            rx, ry, rr, ok = refine_spot(gray_filtered, red_img, gx, gy, median_r)
            # Safety: reject a refinement that jumps too far from the anchor
            if ok and np.hypot(rx - gx, ry - gy) > grid_tol * 0.8:
                rx, ry, rr, ok = gx, gy, median_r, False
        else:
            rx, ry, rr, ok = gx, gy, median_r, False
        refine_results.append({"ok": bool(ok), "rx": float(rx), "ry": float(ry),
                               "rr": float(rr), "rescue": 0})
        if ok:
            refined_count += 1

    # Stage C: rescue failed spots from the row/column offset trend
    rescued = 0
    if neighbor_fix_enabled and refine_enabled:
        for idx in range(len(grid_points)):
            if refine_results[idx]["ok"]:
                continue
            l0, k0 = idx // 4, idx % 4
            gx, gy = grid_points[idx]
            pred_row = predict_offset_along(refine_results, grid_points, l0, k0, "row")
            pred_col = predict_offset_along(refine_results, grid_points, l0, k0, "col")

            if pred_row and pred_col:
                w1, w2 = pred_row[2], pred_col[2]
                ox = (pred_row[0] * w1 + pred_col[0] * w2) / (w1 + w2)
                oy = (pred_row[1] * w1 + pred_col[1] * w2) / (w1 + w2)
            elif pred_row:
                ox, oy = pred_row[0], pred_row[1]
            elif pred_col:
                ox, oy = pred_col[0], pred_col[1]
            else:
                continue  # No usable neighbor; keep the grid anchor

            # Cap the predicted shift so an extrapolation cannot run away
            cap = neighbor_fix_max_shift * median_r
            mag = float(np.hypot(ox, oy))
            if mag > cap and mag > 0:
                ox, oy = ox * cap / mag, oy * cap / mag
            mx, my = gx + ox, gy + oy

            rx2, ry2, rr2, ok2 = refine_spot(gray_filtered, red_img, mx, my, median_r)
            if ok2 and np.hypot(rx2 - mx, ry2 - my) <= grid_tol * 0.8:
                refine_results[idx] = {"ok": True, "rx": rx2, "ry": ry2,
                                        "rr": rr2, "rescue": 1}
            else:
                refine_results[idx] = {"ok": False, "rx": mx, "ry": my,
                                      "rr": median_r, "rescue": 2}
            refined_count += 1
            rescued += 1
            print(f"  [rescue] spot near ({int(gx)}, {int(gy)}) corrected to "
                  f"({mx:.1f}, {my:.1f}), rescue={refine_results[idx]['rescue']}")

    # Second pass: read out and draw
    for idx, (gx, gy) in enumerate(grid_points):
        res = refine_results[idx]
        rx, ry, rr = res["rx"], res["ry"], res["rr"]
        ok = res["ok"]
        rescue = res["rescue"]

        read_r = rr * edge_shrink
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(rx), int(ry)), int(round(read_r)), 255, -1)
        mean_rgb = cv2.mean(img_rgb, mask=mask)[:3]
        R, G, B = mean_rgb
        brightness = (R + G + B) / 3
        final_spots.append({
            "id": idx + 1,
            "x": round(rx, 2),
            "y": round(ry, 2),
            "used_r": round(rr, 2),
            "refined": 1 if ok else 0,
            "rescue": rescue,
            "brightness": round(brightness, 2),
            "R": round(R, 2),
            "G": round(G, 2),
            "B": round(B, 2)
        })
        cv2.circle(draw_img, (int(rx), int(ry)), int(round(read_r)), (0, 255, 0), 2)
        cv2.circle(compare_canvas, (int(rx), int(ry)), int(round(read_r)), (0, 255, 0), 2)

    # Consistent numbering: top-left origin, row-major order
    final_spots.sort(key=lambda s: s["y"])
    for row in range(4):
        row_spots = final_spots[row * 4:(row + 1) * 4]
        row_spots.sort(key=lambda s: s["x"])
        final_spots[row * 4:(row + 1) * 4] = row_spots
    for i, s in enumerate(final_spots):
        s["id"] = i + 1
        cv2.putText(draw_img, f"{s['id']}", (int(s["x"]) - 12, int(s["y"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print(f"  Refined spots = {refined_count} / 16" +
          (f", rescued = {rescued}" if rescued > 0 else ""))

    imwrite_unicode(out_img, draw_img)
    imwrite_unicode(compare_img, compare_canvas)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y", "used_r", "refined",
                                               "rescue", "brightness", "R", "G", "B"])
        writer.writeheader()
        writer.writerows(final_spots)

    print(f"  Output: {out_img}")
    print(f"          {compare_img}")
    print(f"          {out_csv}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if target and os.path.isdir(target):
        images = [os.path.join(target, f) for f in sorted(os.listdir(target))
                  if os.path.splitext(f)[1].lower() in image_exts]
        print(f"Batch mode: {len(images)} image(s) in {target}")
        for img in images:
            print(f"\nProcessing: {img}")
            process_image(img)
    elif target and os.path.isfile(target):
        print(f"Processing: {target}")
        process_image(target)
    elif HAS_TK:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select membrane image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if selected:
            process_image(selected)
        else:
            print("No image selected.")
    else:
        print("Usage: python auto_spot_readout.py <image_path | image_folder>")


if __name__ == "__main__":
    main()
