import os
import shutil
from pathlib import Path
from main import extract_ui_elements
from element_classifier import ElementClassifier
from text_validator import TextValidator


def run_pipeline(image_path="UI-edited.jpg",
                 element_dir="Element",
                 json_path="elements.json",
                 skip_extraction=False):
    element_path = Path(element_dir)

    # --- Step 1: Extract (Model A) ---
    crops_exist = list(element_path.glob("el_*.jpg"))
    if not skip_extraction and not crops_exist:
        print("[Pipeline] Running Model A – extracting UI elements ...")
        count = extract_ui_elements(image_path, output_dir=element_dir)
        print(f"[Pipeline] Model A extracted {count} elements.")
    else:
        if skip_extraction:
            print("[Pipeline] Skipping Model A by request.")
        else:
            print("[Pipeline] Existing crops found – skipping Model A (no overwrite).")

    # --- Step 2: Classify (Model B) ---
    print("[Pipeline] Running Model B – classifying extracted elements ...")
    classifier = ElementClassifier(
        element_dir=element_dir,
        coords_file=str(element_path / "coordinatesElement.json"),
        output_path=json_path,
    )
    result = classifier.run()

    # --- Step 3: Validate text entries ---
    print("[Pipeline] Running TextValidator – checking 'text' entries ...")
    validator = TextValidator(
        element_dir=element_dir,
        elements_json=json_path,
    )
    result = validator.run()

    # --- Step 4: Cleanup after classification ---
    if element_path.exists():
        for child in element_path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(str(child))
        print(f"[Pipeline] Cleaned up everything in '{element_dir}'.")
    else:
        print(f"[Pipeline] '{element_dir}' not found – nothing to clean.")

    return result


if __name__ == "__main__":
    IMAGE = "UI-edited.jpg"
    if not os.path.exists(IMAGE):
        print(f"[Error] Screenshot '{IMAGE}' not found.")
    else:
        run_pipeline(IMAGE)
