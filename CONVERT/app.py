import streamlit as st
import cv2
import pytesseract
import numpy as np
import pandas as pd
import os

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # default di cloud

def find_item_counts(img, template_dict, threshold=0.8, extra_w=40, extra_h=20, dedup_dist=10):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    for item_name, template_path in template_dict.items():
        template = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]

        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) == 0:
            results[item_name] = 0
            continue

        jumlah_total = 0
        seen_points = []
        for pt in zip(*loc[::-1]):
            # Deduplication (hindari hitungan dobel di slot sama)
            if any(abs(pt[0]-sp[0]) < dedup_dist and abs(pt[1]-sp[1]) < dedup_dist for sp in seen_points):
                continue
            seen_points.append(pt)

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


st.set_page_config(page_title="Inventory Comparator", layout="wide")
st.title("📦 Inventory Comparator (Template Matching + OCR)")

uploaded_before = st.file_uploader("Upload BEFORE screenshot", type=["png", "jpg", "jpeg"])
uploaded_after = st.file_uploader("Upload AFTER screenshot", type=["png", "jpg", "jpeg"])

# Sidebar options
st.sidebar.header("⚙️ OCR Settings")
threshold = st.sidebar.slider("Min Match Score", 0.5, 1.0, 0.80, 0.01)
extra_w = st.sidebar.slider("Extra Width for Number Area", 0, 100, 40, 5)
extra_h = st.sidebar.slider("Extra Height for Number Area", 0, 100, 20, 5)
dedup_dist = st.sidebar.slider("Deduplicate Distance", 1, 50, 10, 1)

# Load template images
templates = {}
if os.path.exists("templates"):
    for fname in os.listdir("templates"):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            item_name = os.path.splitext(fname)[0]
            templates[item_name] = os.path.join("templates", fname)

if uploaded_before and uploaded_after and templates:
    file_bytes_before = np.asarray(bytearray(uploaded_before.read()), dtype=np.uint8)
    before_img = cv2.imdecode(file_bytes_before, 1)

    file_bytes_after = np.asarray(bytearray(uploaded_after.read()), dtype=np.uint8)
    after_img = cv2.imdecode(file_bytes_after, 1)

    before = find_item_counts(before_img, templates, threshold, extra_w, extra_h, dedup_dist)
    after = find_item_counts(after_img, templates, threshold, extra_w, extra_h, dedup_dist)

    items = list(templates.keys())
    data = []
    for item in items:
        b = before.get(item, 0)
        a = after.get(item, 0)
        data.append([item, b, a, a - b])

    df = pd.DataFrame(data, columns=["Item", "Before", "After", "Selisih"])
    st.subheader("📊 Hasil Perbandingan")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", csv, "comparison.csv", "text/csv")
else:
    st.info("Upload 2 screenshot + pastikan folder `templates/` berisi ikon item.")


