def find_item_counts(img, template_dict, threshold=0.8, extra_w=40, extra_h=20, dedup_dist=10):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    for item_name, template_path in template_dict.items():
        template = cv2.imread(template_path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        found_points = []
        jumlah_total = 0

        # coba beberapa skala biar lebih fleksibel
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            resized = cv2.resize(template_gray, None, fx=scale, fy=scale)
            w, h = resized.shape[::-1]

            if img_gray.shape[0] < h or img_gray.shape[1] < w:
                continue  # skip kalau template lebih besar dari gambar

            res = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                # Deduplication biar ga hitung ganda
                if any(abs(pt[0]-sp[0]) < dedup_dist and abs(pt[1]-sp[1]) < dedup_dist for sp in found_points):
                    continue
                found_points.append(pt)

                # crop area jumlah
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
