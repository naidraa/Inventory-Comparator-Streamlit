import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

st.set_page_config(page_title="Inventory Comparator", layout="wide")

# === Fungsi deteksi item ===
def find_item_counts(img, template_dict, threshold=0.7, extra_w=40, extra_h=20, dedup_dist=10):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    for item_name, template_path in template_dict.items():
        template = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        found_points = []
        jumlah_total = 0

        # coba multi-scale
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            resized = cv2.resize(template_gray, None, fx=scale, fy=scale)
            w, h = resized.shape[::-1]

            if img_gray.shape[0] < h or img_gray.shape[1] < w:
                continue

            res = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                if any(abs(pt[0]-sp[0]) < dedup_dist and abs(pt[1]-sp[1]) < dedup_dist for sp in found_points):
                    continue
                found_points.append(pt)

                # crop area angka jumlah
                slot_x, slot_y = pt[0], pt[1]
                slot = img[slot_y:slot_y+h+extra_h, slot_x:slot_x+w+extra_w]
                gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                text = pytesseract.image_to_string(
                    thresh,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789"
                ).strip()

                if text.isdigit():
                    jumlah_total += int(text)

        results[item_name] = jumlah_total

    return results

# === Sidebar ===
st.sidebar.header("Upload Gambar Inventory")
img1_file = st.sidebar.file_uploader("Gambar Inventory 1", type=["png","jpg","jpeg"])
img2_file = st.sidebar.file_uploader("Gambar Inventory 2", type=["png","jpg","jpeg"])

threshold = st.sidebar.slider("Match Threshold", 0.3, 1.0, 0.7, 0.05)

# === Load templates ===
template_dir = "templates"
template_dict = {}
if os.path.exists(template_dir):
    for fname in os.listdir(template_dir):
        if fname.lower().endswith((".png",".jpg",".jpeg")):
            template_dict[fname.split(".")[0]] = os.path.join(template_dir, fname)

if not template_dict:
    st.warning("⚠️ Belum ada template di folder `templates/`. Upload crop ikon item dulu ke repo.")
else:
    st.sidebar.success(f"{len(template_dict)} template ditemukan")

# === Main App ===
if img1_file and img2_file and template_dict:
    img1 = np.array(Image.open(img1_file).convert("RGB"))[..., ::-1].copy()
    img2 = np.array(Image.open(img2_file).convert("RGB"))[..., ::-1].copy()

    st.image([img1[..., ::-1], img2[..., ::-1]], caption=["Inventory 1", "Inventory 2"], use_column_width=True)

    with st.spinner("🔎 Menganalisa..."):
        counts1 = find_item_counts(img1, template_dict, threshold=threshold)
        counts2 = find_item_counts(img2, template_dict, threshold=threshold)

    # tampilkan tabel perbandingan
    st.subheader("📊 Perbandingan Inventory")
    table_data = []
    for item in template_dict.keys():
        v1 = counts1.get(item, 0)
        v2 = counts2.get(item, 0)
        diff = v2 - v1
        table_data.append([item, v1, v2, diff])

    st.table(table_data)
