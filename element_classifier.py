import os
import json
import cv2
import numpy as np
from pathlib import Path


class ElementClassifier:
    """
    Model B — Heuristic classifier for UI element crops.
    Uses hard geometric gates + targeted heuristics for high accuracy
    on standard mobile/web UI screenshots.
    """

    TOP_CATEGORIES = ["element", "text", "image", "other"]
    ELEMENT_SUBTYPES = [
        "button", "icon", "input", "checkbox",
        "toggle", "dropdown", "slider", "link"
    ]

    def __init__(self, element_dir="Element",
                 coords_file="Element/coordinatesElement.json",
                 output_path="elements.json"):
        self.element_dir = Path(element_dir)
        self.coords_file = Path(coords_file)
        self.output_path = Path(output_path)
        self.coordinates = {}
        self.results = {cat: {} for cat in self.TOP_CATEGORIES}

    # ------------------------------------------------------------------
    # Coordinate loading
    # ------------------------------------------------------------------
    def load_coordinates(self):
        with open(self.coords_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.coordinates = data.get("els", {})

    # ------------------------------------------------------------------
    # Feature extraction (pure OpenCV)
    # ------------------------------------------------------------------
    def extract_features(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        area = h * w
        aspect = w / max(h, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # colour / texture
        mean_c = cv2.mean(img)[:3]
        std_c = np.std(img, axis=(0, 1))
        mean_std = float(np.mean(std_c))
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # edges
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.count_nonzero(edges)
        edge_density = edge_count / area if area else 0.0

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        h_edges = np.count_nonzero(np.abs(sobelx) > 30)
        v_edges = np.count_nonzero(np.abs(sobely) > 30)
        h_edge_ratio = h_edges / area
        v_edge_ratio = v_edges / area

        # contour geometry
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cnt_area = cv2.contourArea(largest)
            x, y, bw, bh = cv2.boundingRect(largest)
            bbox_area = bw * bh
            fill_ratio = cnt_area / bbox_area if bbox_area else 1.0
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = cnt_area / hull_area if hull_area else 1.0
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
            is_rectangular = float(len(approx) == 4)
            child_count = (
                sum(1 for h in hierarchy[0] if h[3] != -1)
                if hierarchy is not None else 0
            )
        else:
            fill_ratio = 1.0
            solidity = 1.0
            is_rectangular = 0.0
            child_count = 0

        # dominant colours
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixels, 3, None, criteria, 5,
                                    cv2.KMEANS_PP_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dom_color_count = sum(1 for c in counts if c / len(labels) > 0.10)

        # MSER text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        mser_count = sum(
            1 for r in regions
            if 10 < cv2.contourArea(r) < area * 0.6
        )

        # Connected components
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cc_count = sum(
            1 for i in range(1, num_labels)
            if 5 < stats[i, cv2.CC_STAT_AREA] < area * 0.5
        )

        return {
            "width": w,
            "height": h,
            "area": area,
            "aspect": aspect,
            "mean_std": mean_std,
            "laplacian_var": laplacian_var,
            "edge_density": edge_density,
            "h_edge_ratio": h_edge_ratio,
            "v_edge_ratio": v_edge_ratio,
            "fill_ratio": fill_ratio,
            "solidity": solidity,
            "is_rectangular": is_rectangular,
            "child_count": child_count,
            "dom_color_count": dom_color_count,
            "mser_count": mser_count,
            "cc_count": cc_count,
        }

    # ------------------------------------------------------------------
    # Hard-gate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gate_pass(val, lo, hi):
        """Returns 1.0 if inside [lo, hi], else 0.0."""
        return 1.0 if lo <= val <= hi else 0.0

    @staticmethod
    def _soft_score(val, lo, hi, weight=1.0):
        """Linear score inside [lo, hi], decays outside with half-width band."""
        centre = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        if half <= 0:
            return weight if abs(val - centre) < 0.001 else 0.0
        dist = abs(val - centre)
        if dist <= half:
            return weight
        decay = (dist - half) / half
        return max(0.0, weight * (1.0 - decay))

    # ------------------------------------------------------------------
    # Top-level classification with hard gates
    # ------------------------------------------------------------------
    def classify(self, features):
        s = features
        a = s["aspect"]
        area = s["area"]
        h = s["height"]
        w = s["width"]
        ed = s["edge_density"]
        hed = s["h_edge_ratio"]
        rec = s["is_rectangular"]
        sol = s["solidity"]
        fill = s["fill_ratio"]

        scores = {"text": 0.0, "image": 0.0, "element": 0.0, "other": 0.0}

        # ==============================
        # OTHER (catch-all / noise)
        # ==============================
        # Extreme aspect (scrollbar, divider, line)
        if a < 0.1 or a > 12:
            scores["other"] += 5.0
        # Very large with very little edge detail
        if area > 60000 and ed < 0.05:
            scores["other"] += 4.0
        # Very tiny noise
        if area < 100:
            scores["other"] += 3.0
        # Low solidity blob
        if sol < 0.5:
            scores["other"] += 1.5

        # ==============================
        # TEXT
        # ==============================
        # Hard gate: must be reasonably wide and not huge
        if a >= 1.2 and h <= 120 and area <= 40000:
            # Wide aspect strongly boosts text
            scores["text"] += self._soft_score(a, 1.5, 10.0, 3.0)
            # Height sweet-spot for typography
            scores["text"] += self._soft_score(h, 14, 80, 2.0)
            # Horizontal edge density (text lines)
            scores["text"] += self._soft_score(hed, 0.04, 0.6, 2.5)
            # Moderate overall edges
            scores["text"] += self._soft_score(ed, 0.03, 0.5, 1.0)
            # Few child contours (single text block, not icon/button)
            if s["child_count"] <= 3:
                scores["text"] += 1.5
            else:
                scores["text"] -= 1.0
            # Low colour variance = monochrome text
            if s["mean_std"] < 45:
                scores["text"] += 1.5
            else:
                scores["text"] -= 0.5
            # MSER text regions
            scores["text"] += self._soft_score(s["mser_count"], 2, 30, 1.5)
            # CC count moderate (letters)
            if s["cc_count"] >= 2:
                scores["text"] += 1.0
            # Penalty for too-large area
            if area > 25000:
                scores["text"] -= 2.5

        # ==============================
        # IMAGE
        # ==============================
        # Must have some size and colour/texture
        if area >= 1000 and 0.3 <= a <= 5.0:
            scores["image"] += self._soft_score(area, 2000, 200000, 1.5)
            scores["image"] += self._soft_score(s["laplacian_var"], 30, 3000, 1.5)
            # Rich colours
            if s["dom_color_count"] >= 3:
                scores["image"] += 1.5
            else:
                scores["image"] -= 0.5
            # Texture / edges
            scores["image"] += self._soft_score(ed, 0.04, 0.6, 1.0)
            # Multiple internal regions = photo/illustration
            if s["child_count"] > 2:
                scores["image"] += 1.0
            # Mean std = colour vibrancy
            if s["mean_std"] > 25:
                scores["image"] += 1.0
            # Penalty for very thin strips (not images)
            if a > 8 or a < 0.15:
                scores["image"] -= 3.0
            # Penalty for tiny edge-less blobs
            if ed < 0.02:
                scores["image"] -= 2.0

        # ==============================
        # ELEMENT (interactive UI control)
        # ==============================
        # Size gate
        if 100 <= area <= 60000:
            # Aspect: buttons, icons, inputs are usually 0.5–6.0
            scores["element"] += self._soft_score(a, 0.5, 6.0, 2.0)
            # Rectangular = UI control
            if rec and sol > 0.80:
                scores["element"] += 2.5
            elif rec:
                scores["element"] += 1.0
            # Moderate edge density = border / filled shape
            scores["element"] += self._soft_score(ed, 0.03, 0.55, 1.5)
            # Some internal structure
            if s["child_count"] >= 1:
                scores["element"] += 1.0
            # Fill ratio
            scores["element"] += self._soft_score(fill, 0.2, 1.0, 0.8)
            # Near-square small things strongly = icon/button
            if 0.7 <= a <= 1.5 and area <= 6000:
                scores["element"] += 2.0
            # Penalty for extreme aspect
            if a > 10 or a < 0.15:
                scores["element"] -= 2.0
            # Penalty for near-text dimensions
            if a > 2.0 and h < 50 and s["mean_std"] < 40 and s["child_count"] <= 1:
                scores["element"] -= 1.5
            # Penalty for huge area
            if area > 50000:
                scores["element"] -= 2.0

        # ==============================
        # Tie-breaking & confidence
        # ==============================
        # If other is leading but has a weak score, try to avoid it
        if scores["other"] > 0 and max(scores["text"], scores["image"], scores["element"]) > scores["other"]:
            scores["other"] *= 0.3

        total = sum(scores.values()) + 1e-6
        confidences = {k: round(v / total, 3) for k, v in scores.items()}
        best_cat = max(confidences, key=confidences.get)
        best_conf = confidences[best_cat]

        subtype = None
        if best_cat == "element":
            subtype = self._classify_element_subtype(features)

        return best_cat, subtype, best_conf, confidences

    # ------------------------------------------------------------------
    # Element subtype — hard-gate heuristics
    # ------------------------------------------------------------------
    def _classify_element_subtype(self, s):
        a = s["aspect"]
        area = s["area"]
        h = s["height"]
        ed = s["edge_density"]
        rec = s["is_rectangular"]
        sol = s["solidity"]

        scores = {}

        # BUTTON
        scores["button"] = 0.0
        scores["button"] += self._soft_score(a, 1.0, 8.0, 2.5)
        scores["button"] += self._soft_score(area, 500, 50000, 1.5)
        scores["button"] += self._soft_score(h, 30, 120, 1.0)
        if rec and sol > 0.80:
            scores["button"] += 2.0
        scores["button"] += self._soft_score(ed, 0.03, 0.5, 1.0)
        # Button usually has internal text or icon (child contours)
        if s["child_count"] >= 1:
            scores["button"] += 1.5
        # Large near-rectangular strongly = button
        if a >= 1.5 and area > 2000 and rec:
            scores["button"] += 1.5

        # ICON
        scores["icon"] = 0.0
        scores["icon"] += self._soft_score(a, 0.6, 2.0, 2.5)
        scores["icon"] += self._soft_score(area, 80, 6000, 2.0)
        scores["icon"] += self._soft_score(h, 16, 80, 1.0)
        # Icons are compact, high edge density
        scores["icon"] += self._soft_score(ed, 0.06, 0.7, 1.5)
        # Few colours
        if s["dom_color_count"] <= 3:
            scores["icon"] += 1.0
        else:
            scores["icon"] -= 0.5
        if s["mean_std"] < 55:
            scores["icon"] += 1.0
        # Very near-square small = strong icon signal
        if 0.7 <= a <= 1.4 and area <= 3500:
            scores["icon"] += 2.0
        # Penalise if it looks like a button (too large)
        if area > 8000 and a > 1.5:
            scores["icon"] -= 2.0

        # INPUT
        scores["input"] = 0.0
        scores["input"] += self._soft_score(a, 2.0, 15.0, 2.5)
        scores["input"] += self._soft_score(h, 24, 100, 1.5)
        scores["input"] += self._soft_score(area, 300, 20000, 1.0)
        # Input is a rectangle with border, mostly empty inside
        if rec and sol > 0.82:
            scores["input"] += 2.0
        scores["input"] += self._soft_score(ed, 0.04, 0.4, 1.0)
        # Sparse interior
        if s["child_count"] <= 1:
            scores["input"] += 1.0
        else:
            scores["input"] -= 0.5
        # Penalise if filled too much
        if s["fill_ratio"] > 0.8:
            scores["input"] -= 1.0

        # CHECKBOX
        scores["checkbox"] = 0.0
        scores["checkbox"] += self._soft_score(a, 0.7, 1.4, 2.5)
        scores["checkbox"] += self._soft_score(area, 100, 3000, 2.0)
        scores["checkbox"] += self._soft_score(h, 16, 70, 1.0)
        if rec and sol > 0.80:
            scores["checkbox"] += 2.0
        scores["checkbox"] += self._soft_score(ed, 0.06, 0.6, 1.0)

        # TOGGLE
        scores["toggle"] = 0.0
        scores["toggle"] += self._soft_score(a, 1.4, 3.2, 2.5)
        scores["toggle"] += self._soft_score(h, 18, 70, 1.5)
        scores["toggle"] += self._soft_score(area, 250, 4500, 1.5)
        if rec:
            scores["toggle"] += 1.0
        scores["toggle"] += self._soft_score(ed, 0.06, 0.5, 1.0)

        # DROPDOWN
        scores["dropdown"] = 0.0
        scores["dropdown"] += self._soft_score(a, 1.5, 6.0, 2.0)
        scores["dropdown"] += self._soft_score(area, 400, 18000, 1.5)
        scores["dropdown"] += self._soft_score(h, 30, 100, 1.0)
        if rec:
            scores["dropdown"] += 1.5
        # Usually has a little arrow/chevron child
        if s["child_count"] >= 1:
            scores["dropdown"] += 1.0

        # SLIDER
        scores["slider"] = 0.0
        scores["slider"] += self._soft_score(a, 2.5, 20.0, 3.0)
        scores["slider"] += self._soft_score(h, 8, 50, 1.5)
        scores["slider"] += self._soft_score(area, 150, 8000, 1.0)
        scores["slider"] += self._soft_score(ed, 0.03, 0.4, 1.0)
        # Very thin & wide = slider / progress bar
        if a > 5 and h < 30:
            scores["slider"] += 2.0

        # LINK
        scores["link"] = 0.0
        scores["link"] += self._soft_score(a, 1.5, 10.0, 2.0)
        scores["link"] += self._soft_score(h, 12, 60, 1.5)
        scores["link"] += self._soft_score(area, 200, 12000, 1.0)
        # Text-like edge density
        scores["link"] += self._soft_score(ed, 0.04, 0.4, 1.0)
        if s["mean_std"] < 45:
            scores["link"] += 1.0
        # Penalise if it scores higher as button
        if scores.get("button", 0) > scores["link"] + 1.0:
            scores["link"] -= 2.0

        best = max(scores, key=scores.get)
        return best

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    @staticmethod
    def get_capabilities(subtype):
        mapping = {
            "button": ["clickable", "tappable", "pressable"],
            "icon": ["clickable", "tappable"],
            "input": ["editable", "focusable", "typeable", "clickable"],
            "checkbox": ["toggleable", "clickable", "checkable"],
            "toggle": ["toggleable", "clickable", "switchable"],
            "dropdown": ["expandable", "clickable", "selectable"],
            "slider": ["draggable", "adjustable", "slideable"],
            "link": ["clickable", "tappable"],
        }
        return mapping.get(subtype, [])

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run(self):
        self.load_coordinates()
        image_files = sorted(self.element_dir.glob("el_*.jpg"))
        if not image_files:
            print("No element images found in", self.element_dir)
            return self.results

        for img_path in image_files:
            name = img_path.stem
            features = self.extract_features(img_path)
            if features is None:
                continue

            cat, sub, conf, all_conf = self.classify(features)
            coords = self.coordinates.get(name)

            entry = {
                "coordinates": coords,
                "confidence": conf,
            }
            if cat == "element":
                entry["capabilities"] = self.get_capabilities(sub)
                self.results["element"].setdefault(sub, {})[name] = entry
            else:
                self.results[cat][name] = entry

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved classification to {self.output_path} ({len(image_files)} elements)")
        return self.results


if __name__ == "__main__":
    ec = ElementClassifier()
    ec.run()
