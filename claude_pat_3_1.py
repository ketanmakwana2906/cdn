"""
UNIVERSAL chart-data extraction script for TI-vs-wind-speed report charts.
Unlike earlier versions, this makes NO assumptions about image resolution,
plot pixel geometry, or axis min/max -- every one of those is detected
fresh from each image. This was specifically built and validated against
two different chart families with different image sizes AND different
axis ranges (KAD-xx: 1520x847px, y-axis 0-31; MAG-PN-xx: ~1179x605px,
y-axis 0-32), including a chart variant with a shaded "IEC demand
exceeded" overlay region.

PIPELINE PER IMAGE
  1. detect_plot_box       - find the axis border box (contiguity-based,
                             robust to text/gridlines outweighing the
                             border in raw pixel count on some renders)
  2. manual_axis_range     - Prompts the user via CMD for the specific
                             X and Y bounds for the current image.
  3. calibrate             - fit pixel<->value mapping via least-squares
                             regression across ALL detected gridlines
                             (local-baseline gridline detection, robust to
                             shaded overlay regions), anchored by the
                             manual axis range
  4. detect_legend_colors  - OCR the legend to find the exact "Effective
                             TI" (and "ambient TI") swatch colors fresh
                             per image, not hard-coded
  5. extract + clean curve - color-cluster extraction with red/pink
                             hard-lightness-split disambiguation (same
                             hue, different lightness -- a loose color
                             match lets one bleed into the other at
                             crossing points) + continuity-based outlier
                             rejection
  6. calibration_confidence - validates the fitted fit quality; if it's
                             bad, the image is flagged rather than
                             silently trusted
"""

import cv2
import numpy as np
import pandas as pd
import os
import glob
from scipy.ndimage import median_filter

# ============================================================
# ====================  USER CONFIG  ========================
# ============================================================

TESSERACT_CMD = r"C:\Users\30140541\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Folder containing the chart images to process.
INPUT_FOLDER = "bkp/sam"

# Which image file extensions to pick up from INPUT_FOLDER.
IMAGE_PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')

# The wind speeds (x-axis values) to extract at.
TARGET_X = [3, 4, 5, 6, 7, 8, 9, 9.3, 10, 11, 11.3, 12, 13, 14, 15, 16,
        17, 18, 19, 20] 

# Name of the output Excel file (written inside INPUT_FOLDER unless
# you give a full path).
OUTPUT_XLSX = "master_output_3.xlsx"

# --- Resolution handling ---
CANONICAL_WIDTH_PX = 1400
MIN_SOURCE_WIDTH_PX = 700
NATIVE_BAND_PX = (1000, 1700)

# ============================================================

def normalize_image(img):
    """Resize toward CANONICAL_WIDTH_PX ONLY when the source is below the
    validated native band. Refuses (returns None) below
    MIN_SOURCE_WIDTH_PX, where even upscaling can't recover detail that
    was never captured in the source image -- genuine information loss,
    not something more processing can fix."""
    h, w = img.shape[:2]
    if w < MIN_SOURCE_WIDTH_PX:
        return None, f"source width {w}px below MIN_SOURCE_WIDTH_PX={MIN_SOURCE_WIDTH_PX}"
    if NATIVE_BAND_PX[0] <= w <= NATIVE_BAND_PX[1]:
        return img, None
    scale = CANONICAL_WIDTH_PX / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp), None

try:
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ============================================================
# 1. PLOT BORDER BOX DETECTION
# ============================================================
def _longest_run(bool_arr):
    if not bool_arr.any():
        return 0
    diff = np.diff(np.concatenate(([0], bool_arr.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).max() if len(starts) else 0

def detect_plot_box(gray, thresh=150):
    dark = gray < thresh
    H, W = gray.shape
    row_runs = np.array([_longest_run(dark[r, :]) for r in range(H)])
    col_runs = np.array([_longest_run(dark[:, c]) for c in range(W)])
    row_thresh = 0.5 * row_runs.max()
    col_thresh = 0.5 * col_runs.max()
    rows = np.where(row_runs > row_thresh)[0]
    cols = np.where(col_runs > col_thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Could not detect a plot border box in this image")
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    return top, bottom, left, right


# ============================================================
# 3. GRIDLINE DETECTION + CALIBRATION REGRESSION
# ============================================================
def _detect_gridlines_local(line, window=9, min_dev=3):
    baseline = median_filter(line, size=window, mode='nearest')
    dev = baseline - line
    idx = np.where(dev > min_dev)[0]
    if len(idx) == 0:
        return np.array([])
    groups, cur = [], [idx[0]]
    for x in idx[1:]:
        if x - cur[-1] <= 2:
            cur.append(x)
        else:
            groups.append(cur); cur = [x]
    groups.append(cur)
    return np.array([np.average(g, weights=dev[g]) for g in groups])

def _robust_axis_fit(centers, n_expected, tol_px=0.4):
    centers = np.sort(centers)
    if len(centers) < 3:
        return None
    slope0 = (centers[-1] - centers[0]) / max(n_expected - 1, 1)
    intercept0 = centers[0]
    for _ in range(6):
        idx_est = (centers - intercept0) / slope0
        idx_round = np.round(idx_est)
        resid = np.abs(idx_est - idx_round)
        keep = (idx_round >= 0) & (idx_round <= n_expected - 1) & (resid < tol_px)
        if keep.sum() < 3:
            tol_px *= 1.6
            continue
        A = np.vstack([idx_round[keep], np.ones(keep.sum())]).T
        slope0, intercept0 = np.linalg.lstsq(A, centers[keep], rcond=None)[0]
    resid_final = centers - (slope0 * np.round((centers - intercept0) / slope0) + intercept0)
    return slope0, intercept0, np.abs(resid_final).mean()

def _best_fit_n_ticks(centers, n_candidates):
    best = None
    for n in n_candidates:
        fit = _robust_axis_fit(centers, n)
        if fit is None:
            continue
        slope, intercept, resid = fit
        idx_est = (np.sort(centers) - intercept) / slope
        kept = ((np.round(idx_est) >= 0) & (np.round(idx_est) <= n - 1) &
                (np.abs(idx_est - np.round(idx_est)) < 0.4)).sum()
        score = (kept, -resid)
        if best is None or score > best[0]:
            best = (score, n, slope, intercept)
    return best

def _infer_tick_spacing(axis_span, n_gridlines_detected):
    if n_gridlines_detected < 2 or axis_span <= 0:
        return 1
    raw_spacing = axis_span / (n_gridlines_detected - 1)
    nice_values = [1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
    return min(nice_values, key=lambda v: abs(np.log(v) - np.log(raw_spacing)))

def calibrate(img, box, x_range=None, y_range=None):
    top, bottom, left, right = box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    x_min, x_max = x_range
    y_min, y_max = y_range
    row_probe = bottom - max(3, int((bottom - top) * 0.01))
    xg = _detect_gridlines_local(gray[row_probe, left:right + 1]) + left
    col_probe = left + max(8, int((right - left) * 0.02))
    yg = _detect_gridlines_local(gray[top:bottom + 1, col_probe]) + top
    
    xg = np.unique(np.concatenate([xg, [left, right]]))
    yg = np.unique(np.concatenate([yg, [top, bottom]]))
    if x_max is not None:
        x_spacing = _infer_tick_spacing(x_max - x_min, len(xg))
        n_x = round((x_max - x_min) / x_spacing) + 1
        xfit = _robust_axis_fit(xg, n_x)
        x_slope, x_intercept = (xfit[0], xfit[1]) if xfit else ((right - left) / (x_max - x_min), left)
    else:
        best = _best_fit_n_ticks(xg, range(15, 60))
        n_x, x_slope, x_intercept = best[1], best[2], best[3]
        x_max = x_min + n_x - 1
    if y_max is not None:
        y_spacing = _infer_tick_spacing(y_max - y_min, len(yg))
        n_y = round((y_max - y_min) / y_spacing) + 1
        yfit = _robust_axis_fit(yg, n_y)
        y_slope_idx, y_intercept = (yfit[0], yfit[1]) if yfit else ((bottom - top) / (y_max - y_min), top)
    else:
        best = _best_fit_n_ticks(yg, range(15, 60))
        n_y, y_slope_idx, y_intercept = best[1], best[2], best[3]
        y_max = y_min + n_y - 1
        
    def x_to_col(x): return x_slope * (x - x_min) + x_intercept
    def col_to_x(c): return (c - x_intercept) / x_slope + x_min
    def y_to_row(y): return y_slope_idx * (y_max - y) + y_intercept
    def row_to_y(row): return y_max - (row - y_intercept) / y_slope_idx
    
    return {
        'plot_box': box, 'x_range': (x_min, x_max), 'y_range': (y_min, y_max),
        'x_to_col': x_to_col, 'col_to_x': col_to_x,
        'y_to_row': y_to_row, 'row_to_y': row_to_y,
        'n_x_gridlines_found': len(xg), 'n_y_gridlines_found': len(yg),
    }

def calibration_confidence(calib, tol_frac=0.03, expected_x_max_range=None, expected_y_max_range=None):
    top, bottom, left, right = calib['plot_box']
    x_min, x_max = calib['x_range']
    y_min, y_max = calib['y_range']
    expected_w = calib['x_to_col'](x_max) - calib['x_to_col'](x_min)
    actual_w = right - left
    expected_h = abs(calib['y_to_row'](y_min) - calib['y_to_row'](y_max))
    actual_h = bottom - top
    w_err = abs(expected_w - actual_w) / max(actual_w, 1)
    h_err = abs(expected_h - actual_h) / max(actual_h, 1)
    ok = (w_err < tol_frac) and (h_err < tol_frac) and calib['n_x_gridlines_found'] >= 10 and calib['n_y_gridlines_found'] >= 10
    info = {'x_span_error_pct': round(w_err * 100, 2), 'y_span_error_pct': round(h_err * 100, 2),
            'x_range': calib['x_range'], 'y_range': calib['y_range']}
    if expected_x_max_range is not None:
        info['x_max_plausible'] = expected_x_max_range[0] <= x_max <= expected_x_max_range[1]
    if expected_y_max_range is not None:
        info['y_max_plausible'] = expected_y_max_range[0] <= y_max <= expected_y_max_range[1]
    return ok, info

# ============================================================
# 4. LEGEND COLOR AUTO-DETECTION (OCR)
# ============================================================
def _ocr_legend_lines(img):
    H, W = img.shape[:2]
    x0, x1 = int(0.80 * W), int(0.995 * W)
    y0, y1 = int(0.08 * H), int(0.30 * H)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or not HAS_OCR:
        return [], x0
    scale = 3
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    lines = {}
    for i, txt in enumerate(data['text']):
        if txt.strip():
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            lines.setdefault(key, []).append(
                (txt, data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
    out = []
    for k, v in sorted(lines.items(), key=lambda kv: min(t[2] for t in kv[1])):
        text = " ".join(t[0] for t in v)
        top = min(t[2] for t in v)
        height = max(t[2] + t[4] for t in v) - top
        left = min(t[1] for t in v)
        out.append({'text': text, 'row_c': y0 + (top + height / 2) / scale,
                     'left_px': x0 + left / scale})
    return out, x0

def _swatch_color(img, row_c, left_edge_x, x0):
    y0i, y1i = int(row_c) - 4, int(row_c) + 5
    x_start, x_end = x0, int(left_edge_x) + 15
    strip = img[y0i:y1i, x_start:x_end].astype(int)
    if strip.size == 0:
        return None
    chroma = strip.max(axis=2) - strip.min(axis=2)
    if chroma.max() < 15:
        return None
    idx = np.unravel_index(np.argmax(chroma), chroma.shape)
    return strip[idx].astype(float)

def detect_legend_colors(img):
    lines, x0 = _ocr_legend_lines(img)
    result = {'effective': None, 'ambient': None}
    for line in lines:
        low = line['text'].lower()
        if 'effective' in low and 'ambient' not in low and result['effective'] is None:
            c = _swatch_color(img, line['row_c'], line['left_px'], x0)
            if c is not None:
                result['effective'] = c
        elif 'ambient' in low and result['ambient'] is None:
            c = _swatch_color(img, line['row_c'], line['left_px'], x0)
            if c is not None:
                result['ambient'] = c
    if result['ambient'] is not None and result['effective'] is not None:
        red_g = result['effective'][1]
        pink_g = result['ambient'][1]
        pink_r = result['ambient'][2]
        if not (pink_g > red_g and pink_r > 200):
            result['ambient'] = None
    return result

FALLBACK_RED = np.array([114, 114, 255], dtype=float)
FALLBACK_PINK_G_OFFSET = 85

# ============================================================
# 5. CURVE EXTRACTION (red/pink disambiguation + continuity cleaning)
# ============================================================
def estimate_pink_from_image(img, box, red_bgr):
    top, bottom, left, right = box
    seg = img[top:bottom + 1, left:right + 1].reshape(-1, 3).astype(float)
    chroma = seg.max(axis=1) - seg.min(axis=1)
    red_g = red_bgr[1]
    mask = ((seg[:, 2] > 200) & (seg[:, 1] > red_g + 20) & (seg[:, 1] < 235) &
            (chroma > 20) & (np.abs(seg[:, 0] - seg[:, 1]) < 15))
    candidates = seg[mask]
    if len(candidates) < 30:
        return None
    return np.median(candidates, axis=0)

def build_red_mask_fn(red_bgr, pink_bgr=None, tol=80):
    red_g = red_bgr[1]
    split_g = (red_g + pink_bgr[1]) / 2.0 if pink_bgr is not None else red_g + FALLBACK_PINK_G_OFFSET / 2.0
    pure_marker = np.array([255, 0, 0], dtype=float)
    def weight_fn(seg):
        r, g, b = seg[:, 2], seg[:, 1], seg[:, 0]
        chroma = seg.max(axis=1) - seg.min(axis=1)
        gb_gap = np.abs(g - b)
        hue_ok = (r > 200) & (gb_gap < 20)
        w = np.where(
            g <= red_g,
            1.0,
            np.clip(1 - (g - red_g) / max(split_g - red_g, 1), 0, 1)
        )
        w = np.where(hue_ok, w, 0.0)
        w *= np.clip(chroma / 25, 0, 1)
        w = np.where(g >= split_g, 0, w)
        pure_d = np.linalg.norm(seg - pure_marker, axis=1)
        w = np.where(pure_d < 40, 0, w)
        return w
    return weight_fn

def _collect_column_candidates(img, calib, weight_fn):
    top, bottom, left, right = calib['plot_box']
    per_col = {}
    for col in range(left, right + 1):
        seg = img[top:bottom + 1, col, :].astype(float)
        w = weight_fn(seg)
        idx = np.where(w > 0.05)[0]
        if len(idx) == 0:
            continue
        runs, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= 2:
                cur.append(i)
            else:
                runs.append(cur); cur = [i]
        runs.append(cur)
        run_stats = []
        for run in runs:
            weight_sum = w[run].sum()
            if max(w[run]) < 0.12:
                continue
            mean_g = np.mean(seg[run, 1])
            rows_local = np.array(run)
            centroid = np.average(rows_local, weights=w[run]) + top
            y_val = calib['row_to_y'](centroid)
            run_stats.append((run, weight_sum, mean_g, centroid, y_val))
        if run_stats:
            per_col[col] = run_stats
    return per_col

def extract_raw_curve(img, calib, weight_fn):
    candidates_by_col = _collect_column_candidates(img, calib, weight_fn)
    if not candidates_by_col:
        return {}
    cols_sorted = sorted(candidates_by_col)
    rough_y = {}
    for col in cols_sorted:
        stats = candidates_by_col[col]
        max_weight = max(r[1] for r in stats)
        cands = [r for r in stats if r[1] >= 0.35 * max_weight]
        chosen = min(cands, key=lambda r: r[2])
        rough_y[col] = chosen[4]
    cols_arr = np.array(cols_sorted)
    y_arr = np.array([rough_y[c] for c in cols_sorted])
    ref_y = np.empty_like(y_arr)
    HALF_WIN_PX = 12
    for i, c in enumerate(cols_arr):
        m = (cols_arr >= c - HALF_WIN_PX) & (cols_arr <= c + HALF_WIN_PX)
        ref_y[i] = np.median(y_arr[m])
    ref_map = dict(zip(cols_sorted, ref_y))
    raw = {}
    MAX_DEV = 1.5
    for col in cols_sorted:
        stats = candidates_by_col[col]
        ref = ref_map[col]
        chosen = min(stats, key=lambda r: abs(r[4] - ref))
        if abs(chosen[4] - ref) > MAX_DEV:
            continue
        raw[col] = chosen[3]
    return raw

def clean_curve(raw_cols, calib, window=9, max_dev_data_units=0.45, edge_protect=3):
    if not raw_cols:
        return {}
    cols = np.array(sorted(raw_cols))
    rows = np.array([raw_cols[c] for c in cols])
    ys = np.array([calib['row_to_y'](r) for r in rows])
    half = window // 2
    keep = np.ones(len(cols), dtype=bool)
    n = len(cols)
    for i in range(n):
        if i < edge_protect or i >= n - edge_protect:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        nb = np.delete(ys[lo:hi], i - lo)
        if len(nb) == 0:
            continue
        if abs(ys[i] - np.median(nb)) > max_dev_data_units:
            keep[i] = False
    cols_c, ys_c = cols[keep], ys[keep]
    if len(cols_c) < 2:
        return raw_cols
    full_cols = np.arange(cols_c[0], cols_c[-1] + 1)
    ys_interp = np.interp(full_cols, cols_c, ys_c)
    return {c: y for c, y in zip(full_cols, ys_interp)}

def sample_at_x(cleaned_col_to_y, calib, x_query, half_window_px=1.5):
    if not cleaned_col_to_y:
        return None
    c_center = calib['x_to_col'](x_query)
    c0, c1 = int(np.floor(c_center - half_window_px)), int(np.ceil(c_center + half_window_px))
    vals, wts = [], []
    for c in range(c0, c1 + 1):
        if c in cleaned_col_to_y:
            wcol = max(0.01, 1 - abs(c - c_center) / (half_window_px + 1))
            vals.append(cleaned_col_to_y[c]); wts.append(wcol)
    if vals:
        return float(np.average(vals, weights=wts))
    cols_sorted = np.array(sorted(cleaned_col_to_y))
    MAX_EXTRAPOLATE_PX = 20
    if c_center < cols_sorted[0] and cols_sorted[0] - c_center <= MAX_EXTRAPOLATE_PX:
        near = cols_sorted[:3]
        ys = np.array([cleaned_col_to_y[c] for c in near])
        slope = np.polyfit(near, ys, 1)[0] if len(near) >= 2 else 0
        return float(cleaned_col_to_y[cols_sorted[0]] + slope * (c_center - cols_sorted[0]))
    if c_center > cols_sorted[-1] and c_center - cols_sorted[-1] <= MAX_EXTRAPOLATE_PX:
        near = cols_sorted[-3:]
        ys = np.array([cleaned_col_to_y[c] for c in near])
        slope = np.polyfit(near, ys, 1)[0] if len(near) >= 2 else 0
        return float(cleaned_col_to_y[cols_sorted[-1]] + slope * (c_center - cols_sorted[-1]))
    return None

# ============================================================
# 6. TITLE DETECTION (with filename cross-validation)
# ============================================================
def detect_title(img):
    if not HAS_OCR:
        return None
    H, W = img.shape[:2]
    crop = img[0:int(0.08 * H), int(0.25 * W):int(0.75 * W)]
    gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for fx in (4, 3, 5, 6, 2):
        big = cv2.resize(gray0, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        for cfg in ('', '--psm 7', '--psm 6'):
            txt = pytesseract.image_to_string(big, config=cfg).strip().replace('\n', ' ').strip()
            if txt and len(txt) <= 40:
                return txt
    return None

def resolve_graph_name(img, filename_stem):
    ocr_name = detect_title(img)
    if not ocr_name:
        return filename_stem, ocr_name, "filename (OCR found nothing)", False
    norm_ocr = ocr_name.replace(' ', '').upper()
    norm_file = filename_stem.replace(' ', '').upper()
    agrees = norm_file in norm_ocr or norm_ocr in norm_file or norm_ocr == norm_file
    if agrees:
        return ocr_name, ocr_name, "OCR (matches filename)", False
    return filename_stem, ocr_name, "filename (OCR disagreed)", True


# ============================================================
# 8. BATCH RUNNER
# ============================================================
def find_image_files(folder='.', patterns=('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files))

def process_folder(folder, target_x, out_xlsx, patterns=('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')):
    files = find_image_files(folder, patterns)
    df = pd.DataFrame({"Wind speed (m/s)": target_x})
    audit_rows = []
    
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        print(f"\n==========================================")
        print(f"--- Processing Image: {os.path.basename(f)} ---")
        
        img = cv2.imread(f)
        if img is None:
            print("  FAILED: unreadable image file")
            df[stem] = [np.nan] * len(target_x)
            audit_rows.append({"file": stem, "status": "FAILED: unreadable image file",
                                "calibration_ok": False})
            continue
            
        img, norm_reason = normalize_image(img)
        if img is None:
            print(f"  FAILED: {norm_reason}")
            df[stem] = [np.nan] * len(target_x)
            audit_rows.append({"file": stem, "status": f"FAILED: {norm_reason}",
                                "calibration_ok": False})
            continue
            
        # Per-image user input
        print(f"Please provide the manual axis ranges for this image:")
        while True:
            try:
                xmin = float(input("  Enter X-axis min (e.g., 0): "))
                xmax = float(input("  Enter X-axis max (e.g., 30): "))
                ymin = float(input("  Enter Y-axis min (e.g., 0): "))
                ymax = float(input("  Enter Y-axis max (e.g., 32): "))
                break # Exit loop if inputs are valid numbers
            except ValueError:
                print("  Invalid input. Please enter valid numeric values only.")
                
        user_x_range = (xmin, xmax)
        user_y_range = (ymin, ymax)
        print(f"  -> Continuing extraction using X: {user_x_range}, Y: {user_y_range} ...")

        try:
            box = detect_plot_box(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            
            # Using user-provided ranges directly for this image
            calib = calibrate(img, box, user_x_range, user_y_range)
            conf_ok, conf_info = calibration_confidence(calib)
            
            if not conf_ok:
                print(f"  WARNING: calibration low-confidence -- {conf_info}")
                df[stem] = [np.nan] * len(target_x)
                audit_rows.append({"file": stem, "status": "FAILED: calibration low-confidence",
                                    "calibration_ok": False, **conf_info})
                continue
                
            colors = detect_legend_colors(img)
            red_bgr = colors['effective'] if colors['effective'] is not None else FALLBACK_RED
            pink_bgr = colors['ambient']
            if pink_bgr is None:
                pink_bgr = estimate_pink_from_image(img, box, red_bgr)
            weight_fn = build_red_mask_fn(red_bgr, pink_bgr)
            raw = extract_raw_curve(img, calib, weight_fn)
            cleaned = clean_curve(raw, calib)
            vals = [sample_at_x(cleaned, calib, x) for x in target_x]
            df[stem] = [round(v, 3) if v is not None else np.nan for v in vals]
            graph_name, ocr_raw, name_source, mismatch = resolve_graph_name(img, stem)
            n_unresolved = sum(1 for v in vals if v is None)
            status = "OK" if n_unresolved == 0 else f"OK with {n_unresolved} unresolved point(s)"
            print(f"  graph_name={graph_name} [{name_source}]  x_range={conf_info['x_range']} "
                  f"y_range={conf_info['y_range']}  status={status}")
            audit_rows.append({"file": stem, "graph_name": graph_name, "status": status,
                                "calibration_ok": True, "unresolved_points": n_unresolved,
                                **conf_info})
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            df[stem] = [np.nan] * len(target_x)
            audit_rows.append({"file": stem, "status": f"FAILED: {type(e).__name__}: {e}",
                                "calibration_ok": False})
                                
    df_audit = pd.DataFrame(audit_rows)
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        print(df)
        df.to.excel(writer, sheet_name='Effective_TI', index=False)
        df_audit.to.excel(writer, sheet_name='QA_Audit_Log', index=False)
    print(f"\nSaved {out_xlsx}")
    return df, df_audit

def _check_tesseract_ready():
    if not HAS_OCR:
        print("WARNING: pytesseract is not installed (pip install pytesseract). "
              "Legend-color detection will be SKIPPED for all images -- results "
              "will rely entirely on structural fallbacks, which is less reliable. "
              "Install pytesseract and the Tesseract OCR engine for full accuracy.")
        return
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print(f"Tesseract OK -- using: {pytesseract.pytesseract.tesseract_cmd}")
    except Exception as e:
        print(f"WARNING: pytesseract is installed but could not reach the Tesseract "
              f"engine ({e}). Set TESSERACT_CMD at the top of this script to the full "
              f"path of your tesseract executable. Proceeding with reduced accuracy "
              f"(structural fallbacks only).")

if __name__ == '__main__':
    _check_tesseract_ready()
    out_path = OUTPUT_XLSX if os.path.isabs(OUTPUT_XLSX) else os.path.join(INPUT_FOLDER, OUTPUT_XLSX)
    process_folder(INPUT_FOLDER, TARGET_X, out_path, IMAGE_PATTERNS)
