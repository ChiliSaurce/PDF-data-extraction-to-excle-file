import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
from pytesseract import Output
from tqdm import tqdm

# === CONFIG ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # download  tesseract.exe and install it
image_folder = r"D:\PythonProject\CoreDescription\picture_1A" # Your image folder
excel_path = "CoreScan_List.xlsx" # your excel file, including core scan name and depth infor
output_excel = "all_tectonic_descriptions.xlsx" # Output file

# === Load Excel once ===
df_excel = pd.read_excel(excel_path)
filename_column = df_excel.columns[2]  # File names are in the 3rd column
top_col = [col for col in df_excel.columns if 'top' in col.lower()]
if not top_col:
    raise ValueError("No Top Depth column found.")
top_col = top_col[0]

# === Column definitions ===
column_names = [
    "Tectonic element type", "Alpha angle", "Beta angle", "Filling", "Spacing",
    "Roughness", "Aperture", "Clast size", "Clast shape", "Matrix", "Colour", "Comments"
]
column_fractions = [
    (0.31429, 0.35714), (0.35714, 0.40476), (0.40476, 0.45), (0.45, 0.49571),
    (0.49571, 0.54476), (0.54476, 0.58952), (0.58952, 0.63714), (0.63714, 0.68286),
    (0.68286, 0.73190), (0.73190, 0.77762), (0.77762, 0.82), (0.82, 1.0)
] 

def clean_word(word):
    bad_symbols = {'|', '!', '{', '}', '[', ']', '—', '-', '~', '(', ')', '.', '"', "'", '\\', '/'}
    word = word.strip()
    if not word or word in bad_symbols:
        return ""
    return word.strip("|!{}[]()<>\"'\\/")

# === Process All Images ===
all_rows = []

for filename in tqdm(os.listdir(image_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Skipping unreadable: {filename}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        column_ranges = [(int(start * img_width), int(end * img_width)) for start, end in column_fractions]

        # === Match image name to CoreScan_List ===
        core_match = re.search(r"(\d+_\d+_A_)(\d{3})_(\d)", filename)
        if not core_match:
            print(f"⚠️ Skipping unmatched filename: {filename}")
            continue
        match_key = f"{core_match.group(1)}{int(core_match.group(2))}_{core_match.group(3)}"
        matched_row = df_excel[df_excel[filename_column].astype(str).str.contains(match_key, na=False)]
        if matched_row.empty:
            print(f"⚠️ No match in Excel for: {filename}")
            continue
        top_depth_value = float(matched_row.iloc[0][top_col])

        # === Crop image areas ===
        y_frac_start = 0.201
        y_frac_end = 0.9427
        depth_x_start_frac = 0.0
        depth_x_end_frac = 0.071428571
        y_start = int(y_frac_start * img_height)
        y_end = int(y_frac_end * img_height)
        x_start = column_ranges[0][0]
        x_end = column_ranges[-1][1]
        depth_x1 = int(depth_x_start_frac * img_width)
        depth_x2 = int(depth_x_end_frac * img_width)
        tectonic_crop = image_rgb[y_start:y_end, x_start:x_end]
        depth_crop = image_rgb[y_start:y_end, depth_x1:depth_x2]

        # === OCR Tectonic Description ===
        tectonic_ocr = pytesseract.image_to_data(tectonic_crop, config="--oem 3 --psm 6", output_type=Output.DICT)
        lines, current_line, last_y = [], [], -25
        for i, word in enumerate(tectonic_ocr['text']):
            word = clean_word(word)
            if not word:
                continue
            x = tectonic_ocr['left'][i] + x_start
            y = tectonic_ocr['top'][i] + y_start
            if abs(y - last_y) > 25 and current_line:
                lines.append((int(np.mean([item[1] for item in current_line])), current_line))
                current_line = []
            current_line.append((x, y, word))
            last_y = y
        if current_line:
            lines.append((int(np.mean([item[1] for item in current_line])), current_line))

        tectonic_rows = []
        for y, items in lines:
            row = [""] * len(column_names)
            for x, _, word in items:
                for i, (x_min, x_max) in enumerate(column_ranges):
                    if x_min <= x < x_max:
                        row[i] += word + " "
                        break
            tectonic_rows.append([y] + [r.strip() for r in row])

        tectonic_df = pd.DataFrame(tectonic_rows, columns=["Y-pixel"] + column_names)

        # === OCR Depth Markers ===
        depth_ocr = pytesseract.image_to_data(depth_crop, config="--oem 3 --psm 6 outputbase digits", output_type=Output.DICT)
        depth_positions = {}
        for i, word in enumerate(depth_ocr['text']):
            word = word.strip()
            if word.isdigit():
                val = int(word)
                if 10 <= val <= 110 and val % 10 == 0:
                    y = depth_ocr['top'][i] + y_start
                    depth_positions[val] = y
        if not depth_positions:
            print(f"⚠️ No depth markers found for: {filename}")
            continue

        depth_df = pd.DataFrame(sorted(depth_positions.items()), columns=["Depth (cm)", "Y-pixel"])
        depth_df = depth_df.sort_values("Depth (cm)")
        y_pixels = depth_df["Y-pixel"].values
        depths_cm = depth_df["Depth (cm)"].values
        dy = np.diff(y_pixels).mean()
        adjusted_y = y_pixels + (dy / 5.0)
        a, b = np.polyfit(adjusted_y, depths_cm, 1)
        tectonic_df["Section Depth (cm)"] = tectonic_df["Y-pixel"].apply(lambda y: max(0, round(a * y + b, 1)))
        tectonic_df["Geology Depth (m)"] = tectonic_df["Section Depth (cm)"].apply(lambda d: round(top_depth_value + d / 100, 3))

        core_id = re.search(r"\d+_\d+_A_\d+_\d", filename).group()
        tectonic_df["Core ID"] = core_id
        all_rows.append(tectonic_df)

    except Exception as e:
        print(f"❌ Error in {filename}: {e}")

# === Save All to Excel ===
if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
    final_cols = ["Core ID", "Section Depth (cm)", "Geology Depth (m)"] + column_names
    final_df[final_cols].to_excel(output_excel, index=False)
    print(f"\n✅ Saved all results to: {output_excel}")
else:
    print("⚠️ No valid rows to save.")
