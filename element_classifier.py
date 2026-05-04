import os
import json
import cv2
import numpy as np
from pathlib import Path


class ElementClassifier:
    """
    Model B — Heuristic classifier for UI element crops.
    Reads JPGs from an element folder and produces a structured JSON with
    categories: element (with subtypes), text, image, other.
    """

    TOP_CATEGORIES = ["element", "text", "image", "other"]
    ELEMENT_SUBTYPES = [
        "button", "icon", "input", "checkbox",
        "toggle", "dropdown", "slider", "link"
    ]

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
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
    # Feature extraction (pure OpenCV, zero external deps)
    # ------------------------------------------------------------------
    def extract_features(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        area = h * w
        aspect = w / max(h, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- colour / texture ---
        mean_c = cv2.mean(img)[:3]
        std_c = np.std(img, axis=(0, 1))
        mean_std = float(np.mean(std_c))
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # --- edges ---
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.count_nonzero(edges)
        edge_density = edge_count / area if area else 0.0

        # horizontal / vertical edge ratios
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        h_edges = np.count_nonzero(np.abs(sobelx) > 30)
        v_edges = np.count_nonzero(np.abs(sobely) > 30)
        h_edge_ratio = h_edges / area
        v_edge_ratio = v_edges / area

        # --- contour geometry ---
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

        # --- dominant colours (k-means k=3) ---
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixels, 3, None, criteria, 5,
                                    cv2.KMEANS_PP_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dom_color_count = sum(1 for c in counts if c / len(labels) > 0.10)

        # --- MSER (text-like regions) ---
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        mser_count = sum(
            1 for r in regions
            if 10 < cv2.contourArea(r) < area * 0.6
        )

        # --- Connected components ---
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cc_count = sum(
            1 for i in range(1, num_labels)
            if 5 < stats[i, cv2.CC_STAT_AREA] < area * 0.5
        )

        # --- Hu moments (shape descriptor) ---
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

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
            "hu1": float(hu[0]) if len(hu) > 0 else 0.0,
            "hu2": float(hu[1]) if len(hu) > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    # Soft range scorer
    # ------------------------------------------------------------------
    @staticmethod
    def _score_in_range(val, lo, hi, weight=1.0):
        """
        Returns `weight` when val is inside [lo, hi].
        Linearly decays to 0 when outside the range (decay band = range width on each side).
        """
        centre = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        if half <= 0:
            return weight if abs(val - centre) < 0.001 else 0.0
        dist = abs(val - centre)
        if dist <= half:
            return weight
        # linear fall-off beyond the band, using same width as band
        decay = (dist - half) / half
        return max(0.0, weight * (1.0 - decay))

    # ------------------------------------------------------------------
    # Top-level classification
    # ------------------------------------------------------------------
    def classify(self, features):
        s = features

        # ---- TEXT ----
        text_score = 0.0
        text_score += self._score_in_range(s["aspect"], 1.5, 12.0, 2.0)
        text_score += self._score_in_range(s["height"], 12, 100, 1.5)
        text_score += self._score_in_range(s["h_edge_ratio"], 0.05, 0.6, 2.0)
        text_score += self._score_in_range(s["edge_density"], 0.03, 0.4, 1.0)
        text_score += (1.5 if s["child_count"] <= 3 else 0.2)
        text_score += (1.0 if s["mean_std"] < 50 else 0.2)
        text_score += self._score_in_range(s["mser_count"], 2, 40, 1.5)
        text_score += (0.8 if s["cc_count"] >= 3 else 0.1)
        # penalties
        if s["aspect"] < 1.0:
            text_score -= 3.0
        if s["area"] > 80000:
            text_score -= 2.0

        # ---- IMAGE ----
        image_score = 0.0
        image_score += self._score_in_range(s["area"], 5000, 200000, 1.5)
        image_score += self._score_in_range(s["laplacian_var"], 50, 3000, 1.5)
        image_score += (1.0 if s["dom_color_count"] >= 3 else 0.3)
        image_score += self._score_in_range(s["edge_density"], 0.05, 0.6, 1.0)
        image_score += (1.0 if s["child_count"] > 2 else 0.2)
        image_score += (1.0 if s["mean_std"] > 30 else 0.2)
        if s["aspect"] > 20 or s["aspect"] < 0.05:
            image_score -= 2.0
        if s["edge_density"] < 0.02:
            image_score -= 2.0

        # ---- ELEMENT (interactive) ----
        element_score = 0.0
        element_score += self._score_in_range(s["area"], 80, 50000, 1.2)
        element_score += (1.5 if 0.3 <= s["aspect"] <= 6.0 else 0.2)
        element_score += (1.0 if s["is_rectangular"] > 0.5 else 0.3)
        element_score += self._score_in_range(s["solidity"], 0.75, 1.0, 1.0)
        element_score += self._score_in_range(s["fill_ratio"], 0.3, 1.0, 0.8)
        element_score += self._score_in_range(s["edge_density"], 0.03, 0.5, 1.0)
        element_score += (1.0 if s["child_count"] >= 1 else 0.3)
        # strong border + rectangular = likely UI control
        if s["is_rectangular"] and s["solidity"] > 0.85 and s["edge_density"] > 0.05:
            element_score += 2.0
        if s["area"] > 100000:
            element_score -= 2.0
        if s["aspect"] > 10 or s["aspect"] < 0.1:
            element_score -= 1.5

        # ---- OTHER ----
        other_score = 1.0
        if s["area"] > 50000 and s["edge_density"] < 0.05:
            other_score += 3.0
        if s["aspect"] > 15 or s["aspect"] < 0.05:
            other_score += 2.0
        if s["solidity"] < 0.5:
            other_score += 1.0
        # suppress if it looks like a known category
        if text_score > 4:
            other_score -= 3
        if image_score > 4:
            other_score -= 3
        if element_score > 4:
            other_score -= 3

        scores = {
            "text": max(0.0, text_score),
            "image": max(0.0, image_score),
            "element": max(0.0, element_score),
            "other": max(0.0, other_score),
        }

        total = sum(scores.values()) + 1e-6
        confidences = {k: round(v / total, 3) for k, v in scores.items()}
        best_cat = max(confidences, key=confidences.get)
        best_conf = confidences[best_cat]

        subtype = None
        if best_cat == "element":
            subtype = self._classify_element_subtype(features)

        return best_cat, subtype, best_conf, confidences

    # ------------------------------------------------------------------
    # Element subtype classification
    # ------------------------------------------------------------------
    def _classify_element_subtype(self, s):
        scores = {}

        # BUTTON
        scores["button"] = 0.0
        scores["button"] += self._score_in_range(s["aspect"], 0.8, 5.0, 2.0)
        scores["button"] += self._score_in_range(s["area"], 200, 25000, 1.0)
        scores["button"] += self._score_in_range(s["edge_density"], 0.04, 0.5, 1.0)
        scores["button"] += (1.0 if s["child_count"] >= 1 else 0.2)
        scores["button"] += (
            1.0 if s["is_rectangular"] and s["solidity"] > 0.8 else 0.0
        )

        # ICON
        scores["icon"] = 0.0
        scores["icon"] += self._score_in_range(s["aspect"], 0.5, 2.0, 2.0)
        scores["icon"] += self._score_in_range(s["area"], 80, 6000, 1.5)
        scores["icon"] += self._score_in_range(s["edge_density"], 0.08, 0.7, 1.5)
        scores["icon"] += (1.0 if s["dom_color_count"] <= 3 else 0.3)
        scores["icon"] += (1.0 if s["mean_std"] < 60 else 0.2)

        # INPUT
        scores["input"] = 0.0
        scores["input"] += self._score_in_range(s["aspect"], 2.0, 15.0, 2.0)
        scores["input"] += self._score_in_range(s["height"], 25, 120, 1.0)
        scores["input"] += (
            1.5 if s["is_rectangular"] and s["solidity"] > 0.85 else 0.2
        )
        scores["input"] += self._score_in_range(s["edge_density"], 0.05, 0.4, 1.0)
        scores["input"] += (1.0 if s["child_count"] <= 2 else 0.3)

        # CHECKBOX
        scores["checkbox"] = 0.0
        scores["checkbox"] += self._score_in_range(s["aspect"], 0.7, 1.4, 2.0)
        scores["checkbox"] += self._score_in_range(s["area"], 100, 2500, 1.5)
        scores["checkbox"] += (1.5 if s["is_rectangular"] else 0.2)
        scores["checkbox"] += (1.0 if s["edge_density"] > 0.1 else 0.2)

        # TOGGLE
        scores["toggle"] = 0.0
        scores["toggle"] += self._score_in_range(s["aspect"], 1.5, 3.0, 2.0)
        scores["toggle"] += self._score_in_range(s["height"], 20, 80, 1.5)
        scores["toggle"] += self._score_in_range(s["area"], 300, 4000, 1.0)
        scores["toggle"] += (1.0 if s["edge_density"] > 0.1 else 0.2)

        # DROPDOWN
        scores["dropdown"] = 0.0
        scores["dropdown"] += self._score_in_range(s["aspect"], 1.5, 6.0, 1.5)
        scores["dropdown"] += self._score_in_range(s["area"], 200, 15000, 1.0)
        scores["dropdown"] += (1.0 if s["is_rectangular"] else 0.2)
        scores["dropdown"] += (1.0 if s["child_count"] >= 1 else 0.2)

        # SLIDER
        scores["slider"] = 0.0
        scores["slider"] += self._score_in_range(s["aspect"], 3.0, 20.0, 2.0)
        scores["slider"] += self._score_in_range(s["height"], 10, 60, 1.0)
        scores["slider"] += self._score_in_range(s["area"], 200, 10000, 1.0)
        scores["slider"] += (1.0 if s["edge_density"] > 0.05 else 0.2)

        # LINK
        scores["link"] = 0.0
        scores["link"] += self._score_in_range(s["aspect"], 1.5, 10.0, 2.0)
        scores["link"] += self._score_in_range(s["height"], 12, 60, 1.0)
        scores["link"] += self._score_in_range(s["edge_density"], 0.05, 0.4, 1.0)
        scores["link"] += (1.0 if s["mean_std"] < 50 else 0.2)
        # penalise against button look-alikes
        if scores.get("button", 0) > scores["link"]:
            scores["link"] *= 0.5

        best = max(scores, key=scores.get)
        return best

    # ------------------------------------------------------------------
    # Capabilities lookup
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
