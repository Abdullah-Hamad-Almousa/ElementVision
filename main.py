import cv2
import numpy as np
import os
import json


def extract_ui_elements(image_path, output_dir="Element", padding=5):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use bilateral filter to smooth while keeping edges
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold to handle different backgrounds
    thresh = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to merge broken parts of same element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    min_area = 200        # ignore very tiny noise
    max_area = w * h * 0.9  # ignore the whole image
    extracted = 0
    coordinates = {}

    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < min_area or area > max_area:
            continue
        # Ignore overly narrow or flat lines (often separators)
        if bw < 10 or bh < 10:
            continue

        # Add padding and clamp to image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        element_name = f"element_{extracted:03d}"
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(output_dir, f"{element_name}.png")
        cv2.imwrite(out_path, crop)

        # Store coordinates: [x1, y1, x2, y2]
        coordinates[element_name] = [int(x1), int(y1), int(x2), int(y2)]
        extracted += 1

    # Write coordinates to JSON
    json_path = os.path.join(output_dir, "coordinatesElement.json")
    with open(json_path, "w") as f:
        json.dump({"elements": coordinates}, f, indent=2)

    print(f"Extracted {extracted} UI elements to '{output_dir}'")
    print(f"Saved coordinates to '{json_path}'")
    return extracted


if __name__ == "__main__":
    IMAGE = "UI-edited.jpg"
    if not os.path.exists(IMAGE):
        print(f"Error: '{IMAGE}' not found in working directory.")
    else:
        extract_ui_elements(IMAGE)
