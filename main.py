import cv2
import os
import json


def extract_ui_elements(image_path, output_dir="Element", padding=5, scale_factor=2):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    # Work on a downscaled image for detection (biggest performance win)
    small_w, small_h = w // scale_factor, h // scale_factor
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # Light Gaussian blur is much cheaper than bilateral filter and still kills noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu thresholding: single global pass, much faster than adaptive Gaussian
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing on small image (4x fewer pixels = much faster)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours on small image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area thresholds are scaled down to match half-resolution
    min_area = 50
    max_area = small_w * small_h * 0.9
    extracted = 0
    coordinates = {}

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < min_area or area > max_area:
            continue
        # Reject overly thin fragments (scaled dimensions)
        if bw < 5 or bh < 5:
            continue

        # Padding on small-image coordinates
        sx1 = max(0, x - padding)
        sy1 = max(0, y - padding)
        sx2 = min(small_w, x + bw + padding)
        sy2 = min(small_h, y + bh + padding)

        # Upscale to original full-resolution coordinates
        x1 = sx1 * scale_factor
        y1 = sy1 * scale_factor
        x2 = min(w, sx2 * scale_factor)
        y2 = min(h, sy2 * scale_factor)

        # Safety: skip empty crops (shouldn't happen, but cheap to guard)
        if x1 >= x2 or y1 >= y2:
            continue

        element_name = f"el_{extracted:03d}"

        # Crop from the ORIGINAL high-res image so quality stays crisp
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(output_dir, f"{element_name}.jpg")
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Store full-res coordinates
        coordinates[element_name] = [x1, y1, x2, y2]
        extracted += 1

    # Compact JSON (no whitespace) to keep it tiny for downstream models
    json_path = os.path.join(output_dir, "coordinatesElement.json")
    with open(json_path, "w") as f:
        json.dump({"el": coordinates}, f, separators=(',', ':'))

    print(f"Extracted {extracted} UI elements to '{output_dir}'")
    print(f"Saved compact coordinates to '{json_path}'")
    return extracted


if __name__ == "__main__":
    IMAGE = "UI-edited.jpg"
    if not os.path.exists(IMAGE):
        print(f"Error: '{IMAGE}' not found in working directory.")
    else:
        extract_ui_elements(IMAGE)
