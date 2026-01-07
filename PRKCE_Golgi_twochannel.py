# Golgi two-channel analysis for PRKCE experiment
# Analyzes red and white channels separately based on ROIs

import os
import cv2
import numpy as np
import pandas as pd
import zipfile
import re
import sys
from PIL import Image
from read_roi import read_roi_zip
from roifile import ImagejRoi
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from scalebar_detection_dist import get_scale_info

LOG_PATH = "unsupported_rois.log"
LAST_FOLDER_FILE = "last_folder.txt"

# --- ROI to mask ---
def roi_to_mask(roi, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    roi_type = roi.get("type", "").lower()
    if roi_type in ("polygon", "freehand") and "x" in roi and "y" in roi:
        points = np.array(list(zip(roi["x"], roi["y"])), np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif roi_type in ("rect", "rectangle") and all(k in roi for k in ["left", "top", "width", "height"]):
        x, y, w, h = int(roi["left"]), int(roi["top"]), int(roi["width"]), int(roi["height"])
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    elif roi_type == "oval" and all(k in roi for k in ["left", "top", "width", "height"]):
        x, y, w, h = int(roi["left"]), int(roi["top"]), int(roi["width"]), int(roi["height"])
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    else:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"unsupported ROI type: {roi_type}\n")
        return None
    return mask

def create_imagej_roi_bytes(contour, name="roi"):
    points = contour.squeeze()
    roi = ImagejRoi.frompoints(points, name=name)
    return roi.tobytes()

# --- Single ROI processing ---
def analyze_single_roi(photo_name, gray, mask, scale_ratio, roi_name, channel_code, idx, hull_zip_bytes, particle_zip_bytes):
    roi_img = cv2.bitwise_and(gray, gray, mask=mask)
    _, bin_img = cv2.threshold(roi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_PARTICLE_AREA_UM2 = 0.0325
    min_area_px = MIN_PARTICLE_AREA_UM2 / (scale_ratio ** 2)

    particle_areas = []
    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        area_um2 = area_px * (scale_ratio ** 2)
        if area_um2 >= MIN_PARTICLE_AREA_UM2:
            particle_areas.append(area_um2)
            roi_bytes = create_imagej_roi_bytes(cnt, name=f"{roi_name}_particle_{i+1}")
            particle_zip_bytes[f"{roi_name}_particle_{i+1}.roi"] = roi_bytes

    num_particles = len(particle_areas)
    sum_particle_area = sum(particle_areas)
    avg_particle_area = sum_particle_area / num_particles if num_particles > 0 else 0
    particle_sizes_str = ";".join([f"{a:.4f}" for a in particle_areas])

    # Convex hull (using only filtered contours)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) * (scale_ratio ** 2) >= MIN_PARTICLE_AREA_UM2]
    if filtered_contours:
        all_points = np.vstack(filtered_contours)
        hull = cv2.convexHull(all_points)
        hull_area_px = cv2.contourArea(hull)
        hull_area_um2 = hull_area_px * (scale_ratio ** 2)
        hull_bytes = create_imagej_roi_bytes(hull, name=f"{roi_name}_hull")
        hull_zip_bytes[f"{roi_name}_hull.roi"] = hull_bytes
    else:
        hull_area_um2 = 0.0

    return {
        "Image": photo_name,
        "ROI Index": idx,
        "ROI Name": roi_name,
        "Channel": channel_code,
        "Particle Number": num_particles,
        "Particle Sizes (µm²)": particle_sizes_str,
        "Sum of Particle Area (µm²)": sum_particle_area,
        "Average Particle Size (µm²)": avg_particle_area,
        "Convex Hull Area (µm²)": hull_area_um2
    }

# --- Analyze one image ---
def analyze_image(image_path, roi_paths, output_dir, channel_code, scale_ratio):
    os.makedirs(output_dir, exist_ok=True)
    photo_name = os.path.splitext(os.path.basename(image_path))[0]
    image = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    results = []
    hull_zip_bytes = {}
    particle_zip_bytes = {}
    idx = 1

    if isinstance(roi_paths, str):
        roi_paths = [roi_paths]

    for roi_path in roi_paths:
        if not roi_path:
            continue
        if roi_path.lower().endswith(".zip"):
            roi_dict = read_roi_zip(roi_path)
            for name, roi in roi_dict.items():
                mask = roi_to_mask(roi, gray.shape)
                if mask is None:
                    continue
                result = analyze_single_roi(photo_name, gray, mask, scale_ratio, name, channel_code, idx, hull_zip_bytes, particle_zip_bytes)
                results.append(result)
                idx += 1
        elif roi_path.lower().endswith(".roi"):
            roi = ImagejRoi.fromfile(roi_path)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            points = roi.coordinates()
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)
            name = os.path.splitext(os.path.basename(roi_path))[0]
            result = analyze_single_roi(photo_name, gray, mask, scale_ratio, name, channel_code, idx, hull_zip_bytes, particle_zip_bytes)
            results.append(result)
            idx += 1

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{photo_name}_{channel_code}_analysis.csv")
    df.to_csv(csv_path, index=False)

    with zipfile.ZipFile(os.path.join(output_dir, f"{photo_name}_{channel_code}_hulls.zip"), "w") as zf:
        for k, v in hull_zip_bytes.items():
            zf.writestr(k, v)
    with zipfile.ZipFile(os.path.join(output_dir, f"{photo_name}_{channel_code}_particles.zip"), "w") as zf:
        for k, v in particle_zip_bytes.items():
            zf.writestr(k, v)

    print(f"[SAVED] {csv_path} ({len(results)} ROIs)")
    return df

# --- ROI file finder ---
def find_roi_files(folder, channel_code):
    for f in os.listdir(folder):
        fl = f.lower()
        if fl.endswith("roiset.zip") and channel_code.lower() in fl:
            return os.path.join(folder, f)
    pattern = re.compile(rf'(^|[^a-zA-Z]){channel_code.lower()}([^a-zA-Z]|$)')
    rois = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".roi") and pattern.search(f.lower())]
    return rois if rois else None

# --- Folder selection ---
def select_parent_folder():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    start_dir = os.getcwd()
    if os.path.exists(LAST_FOLDER_FILE):
        with open(LAST_FOLDER_FILE, "r", encoding="utf-8") as f:
            last = f.read().strip()
            if os.path.exists(last):
                start_dir = last
    folder = QFileDialog.getExistingDirectory(None, "Select Parent folder:", start_dir)
    if folder:
        with open(LAST_FOLDER_FILE, "w", encoding="utf-8") as f:
            f.write(folder)
        return [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    return []

# --- Batch process ---
def batch_process():
    folders = select_parent_folder()
    if not folders:
        QMessageBox.critical(None, "Error", "No folders selected.")
        return

    sample_image = None
    for folder in folders:
        for file in os.listdir(folder):
            if file.lower().endswith("_ch02.tif") or file.lower().endswith("_ch03.tif"):
                sample_image = os.path.join(folder, file)
                break
        if sample_image:
            break

    if not sample_image:
        QMessageBox.critical(None, "Error", "Sample image does not exist.")
        return

    image = np.array(Image.open(sample_image).convert("RGB"))
    pixels_per_um, _, _ = get_scale_info(image)
    scale_ratio = 1 / pixels_per_um
    print(f"[INFO] scale: {pixels_per_um:.6f} px/µm")

    all_results = []

    for folder in folders:
        print(f"[INFO] Processing: {folder}")
        for file in os.listdir(folder):
            fl = file.lower()
            if fl.endswith("_ch02.tif"):
                roi_files = find_roi_files(folder, "R")
                if roi_files:
                    df = analyze_image(os.path.join(folder, file), roi_files, folder, "R", scale_ratio)
                    df.insert(0, "Folder", os.path.basename(folder))
                    all_results.append(df)
            elif fl.endswith("_ch03.tif"):
                roi_files = find_roi_files(folder, "W")
                if roi_files:
                    df = analyze_image(os.path.join(folder, file), roi_files, folder, "W", scale_ratio)
                    df.insert(0, "Folder", os.path.basename(folder))
                    all_results.append(df)

    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        parent = os.path.dirname(folders[0])
        all_df.to_csv(os.path.join(parent, "twochannel_golgi_all_results.csv"), index=False)
        print(f"[SAVED] twochannel_golgi_all_results.csv ({len(all_df)} rows)")

    QMessageBox.information(None, "Batch process finished.")

if __name__ == "__main__":
    batch_process()
