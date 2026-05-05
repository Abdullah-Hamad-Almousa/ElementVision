import json
import cv2
import numpy as np
from pathlib import Path


class TextValidator:
    """
    Text validator using OpenCV DNN EAST text detector.
    Validates elements in the 'text' category only.
    Non-text items are moved to the 'uncertain' category.
    """
    MODEL_PATH = "frozen_east_text_detection.pb"
    INPUT_WIDTH = 320
    INPUT_HEIGHT = 320
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, element_dir="Element", elements_json="elements.json",
                 debug=False):
        self.element_dir = Path(element_dir)
        self.elements_json = Path(elements_json)
        self.debug = debug
        self._net = None

    def _load_net(self):
        if self._net is None:
            model_path = Path(self.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"EAST model not found: {model_path.resolve()}. "
                    "Please download frozen_east_text_detection.pb to the project root."
                )
            self._net = cv2.dnn.readNet(str(model_path))
        return self._net

    def _has_text(self, img_path: Path) -> bool:
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        h, w = img.shape[:2]
        area = h * w

        # ---- Fast pre-reject: color-richness = photo / illustration ----
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))

        if self.debug:
            print(f"  {img_path.name}: unique_colors={unique_colors}")

        if unique_colors > 800:
            if self.debug:
                print(f"    -> REJECTED by color-richness filter (not text)")
            return False

        # ---- EAST text detection ----
        blob = cv2.dnn.blobFromImage(
            img, 1.0,
            (self.INPUT_WIDTH, self.INPUT_HEIGHT),
            (123.68, 116.78, 103.94),
            swapRB=True, crop=False
        )

        net = self._load_net()
        net.setInput(blob)

        try:
            scores, _ = net.forward([
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ])
        except cv2.error as e:
            if self.debug:
                print(f"  [TextValidator] DNN forward failed for {img_path.name}: {e}")
            return False

        max_score = float(np.max(scores))

        if self.debug:
            print(f"    EAST score = {max_score:.3f}")

        return max_score >= self.CONFIDENCE_THRESHOLD

    def run(self):
        if not self.elements_json.exists():
            print(f"[TextValidator] {self.elements_json} not found.")
            return

        with open(self.elements_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_entries = data.get("text", {})
        if not text_entries:
            print("[TextValidator] No 'text' entries to validate.")
            return data

        uncertain = data.setdefault("uncertain", {})
        moved = 0
        kept = 0

        for el_id, info in list(text_entries.items()):
            img_path = self.element_dir / f"{el_id}.jpg"
            if not img_path.exists():
                if self.debug:
                    print(f"  Missing image for {el_id}")
                continue

            if self._has_text(img_path):
                kept += 1
            else:
                uncertain[el_id] = info
                del text_entries[el_id]
                moved += 1

        if moved:
            data["uncertain"] = uncertain
            data["text"] = text_entries

        with open(self.elements_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(
            f"[TextValidator] Kept {kept} as text, moved {moved} to uncertain."
        )
        return data


if __name__ == "__main__":
    validator = TextValidator(debug=True)
    validator.run()
