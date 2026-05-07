## Scope

Two-stage pipeline:

1. **Model A — Extract UI elements** (classical CV: contour detection + geometry)
2. **Model B — Define extracted elements** (semantic labeling + role + text) — planned, not started

## Current status (done)

- Extracted UI elements via OpenCV contour detection (`cv2.findContours`)
- Returned **center coordinates** for each element in original pixel space
- Saved per-element crops as **JPG** (quality 90)
- Performance improvements applied in `main.py`:
  - Downscale by `scale_factor=2` before processing
  - Grayscale conversion
  - Gaussian blur + Otsu thresholding
  - Morphological closing (`MORPH_CLOSE`)
  - Output crops as JPG for lighter usage

## Model A — Extract all UI elements (Done)

### A1) Input pipeline

- Load screenshot via `cv2.imread`
- Downscale for processing: `small = original // scale_factor` (2x)
- Grayscale conversion (`COLOR_BGR2GRAY`)
- Gaussian blur `(3, 3)`
- Otsu thresholding (`THRESH_BINARY_INV + THRESH_OTSU`)
- No input-side JPG compression (input is whatever screenshot format is provided)

### A2) Detection / segmentation

- **Approach:** Classical contour detection (`cv2.findContours` with `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`)
- Not a learned model — no bounding-box regressor or instance segmentation network
- Output geometry (internal):
  - `bbox` (x, y, w, h) computed on the downscaled image
  - `center` (cx, cy) mapped back to original resolution
- No polygon / mask output currently

### A3) Post-processing

- Area filter: `min_area=50`, `max_area=90%` of downscaled image
- Size filter: width >= 5, height >= 5
- Padding: 5 px on downscaled image (equivalent to 10 px on original)
- Crop clamped to image bounds
- **Not yet implemented:** NMS, merge/split heuristics, nested-element handling, or deduplication

### A4) Output schema (actual)

Files produced per run inside `output_dir` (default `"Element"`):

- `el_NNN.jpg` — cropped element image (JPG quality 90)
- `coordinatesElement.json` — compact JSON: `{"els": {"el_NNN": [cx, cy], ...}}`

Current fields:
- Element key (`el_NNN`)
- `center` `[cx, cy]` in **original pixel coordinates**
- `cropRef` implicit via filename mapping

Planned / missing for v1:
- `id` (sequential index)
- `bbox` (x, y, w, h) exported to JSON
- `confidence`
- `source` (visual / a11y / hybrid)
- `screenshotRef`

### A5) Evaluation

- This is a classical CV pipeline — no trained model, so mAP/IoU and dataset splits do not apply yet
- Current validation is manual inspection of output crops
- Runtime benchmarks: near-instant on desktop (OpenCV C++ backend)
- If we later replace Model A with a learned detector, add:
  - Dataset split (train/val/test)
  - mAP / IoU for boxes
  - Recall-focused thresholding (don’t miss interactive elements)
  - End-to-end latency budget
  - Memory budget (mobile + server)

## Model B — Define extracted elements (Not started)

Goal: take extracted geometry and produce a usable element definition for automation + reasoning.

### B1) Inputs

- Element geometry (bbox/center) — **available**
- Cropped image per element — **available**
- Full screenshot context — **available**
- OCR text — **not available** (optional addition)
- Accessibility / UI tree attributes — **not available**

### B2) Outputs (Element definition v1) — planned

- `type` (button, text, input, icon, image, container, etc.)
- `visualRole` / `interactionRole` (interactive, static, scrollable, etc.)
- `label` (best human label)
- `text` (raw extracted text, if any)
- `state` (enabled/disabled/selected/checked, if detectable)
- `confidenceByField`

### B3) Modeling approach

- Start with **rules + geometric heuristics** (no training data required):
  - Aspect ratio, compactness, area for type guessing
  - Optional OCR (`pytesseract` / `easyocr`) for text fields
- Move to a **lightweight classifier** only if heuristics fail on real data
- Multimodal model is overkill until the baseline is insufficient
- Fallback when OCR/a11y is missing: rely on image-only heuristics or label as `unknown`

### B4) Training & data

- Not applicable for a rules/heuristics baseline
- If upgrading to a learned classifier later:
  - Annotation guidelines for role/type consistency
  - Edge cases: icons without text, composite controls, menus
  - Weak labels from heuristics + human review on high-impact screens

### B5) Integration

- Contract between Model A → Model B: filename (`el_NNN.jpg`) + `coordinatesElement.json`
- Caching:
  - Cache per screenshot hash
  - Cache per element crop hash
- Export format to SDK:
  - Extend `coordinatesElement.json` or produce a new `elements.json`
  - Compatibility with existing `/unitrs/inspect` or elements endpoint

## Next milestones

- [ ] Export `bbox` (x, y, w, h) to `coordinatesElement.json`
- [ ] Add element metadata (area, aspect ratio) to JSON to support heuristics
- [ ] Implement Model B baseline (rules + geometric heuristics for type/role)
- [ ] Add optional OCR for text extraction
- [ ] Add post-processing: NMS / deduplication for overlapping contours
- [ ] Integrate into Unitrs SDK response format

## Open questions (answered)

1. **What is the minimum element set we must detect (interactive only vs all visible UI layers, including containers/background)?**
   > Currently all **external contours** above `min_area` are extracted, so we capture visible foreground elements. Nested elements and holes are ignored because of `RETR_EXTERNAL`. Containers/background will be extracted only if they appear as distinct outer contours. If nested/interactive-only filtering is needed later, switch to `RETR_TREE` and add hierarchy filtering.

2. **What coordinate space should be the contract: pixel coordinates (screenshot) vs logical coordinates (WDA points / Android dp), and how do we map across DPR/scale?**
   > **Pixel coordinates** from the original screenshot. The pipeline maps all coordinates back to the original image resolution before exporting. No logical coordinate mapping or DPR handling is implemented yet; that layer can be added at the SDK/driver level if needed.

3. **What is the target latency + throughput per device (p50/p95), and where will inference run (device vs server)?**
   > Model A runs locally via OpenCV (C++ backend) and completes in milliseconds on desktop. There is no learned inference, so no server/device split is needed at this stage. If Model A is later replaced with a neural detector, latency budgets and runtime location should be revisited.

4. **Do we treat OCR as mandatory for Model B, optional, or replaced by accessibility text when available (and what’s the fallback when both are missing)?**
   > **Optional**. OCR is not used anywhere in the current pipeline. For Model B it can be added as an optional post-processing step per crop. Fallback when both OCR and a11y are missing: rely on image-only heuristics or assign `label: "unknown"`.

5. **What are the canonical labels/roles for Model B (e.g., button/input/text/icon/image/scroll-view), and do we need cross-platform normalization (iOS vs Android)?**
   > Proposed initial heuristic set (not yet finalized):
   > - `text` — wide aspect ratio, moderate height
   > - `button` — compact, near-square or slightly wide
   > - `icon` — small, near-square
   > - `input` — wide with consistent height (hard without context)
   > - `image` — large area, variable aspect
   > - `container` — very large, outer contours
   >
   > Cross-platform normalization is not needed at this stage because we work directly on screenshots.
