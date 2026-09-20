"""Stage 3: a pretrained floor-plan segmentation baseline, in PyTorch on MPS.

Model: Yytsi/floorplan-to-3d-walls on Hugging Face. A U-Net with a
ResNet-34 encoder (segmentation_models_pytorch), four classes
(floor, wall, door, window), MIT-licensed weights, trained on CubiCasa5K,
which is CC BY-NC-SA 4.0. That lineage is disclosed in the README and
is one reason this study is non-commercial.

This stage has not run yet. Four decisions are still open and are marked
OPEN below: the order of the four output channels, how a room is counted,
a page-level confidence, and merging the output into extractions.csv so
that room_count is scored like every other field.

Run:  env/bin/python -m src.segment_floorplan
Reads:  data/processed/extractions.csv (to find pages OCR'd as floor plans)
        data/processed/pages/*.png
Writes: data/processed/segmentation.csv (reference,sheet,field,predicted,confidence)
        data/processed/masks/<page>.png (gitignored; never commit a mask of
        a whole drawing, it reproduces a substantial part of the work)
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data" / "processed" / "pages"
EXTR = ROOT / "data" / "processed" / "extractions.csv"
MASKS = ROOT / "data" / "processed" / "masks"
OUT = ROOT / "data" / "processed" / "segmentation.csv"
WEIGHTS = ROOT / "models" / "floorplan_unet_resnet34.safetensors"
CLASSES = ["floor", "wall", "door", "window"]


def load_model():
    import segmentation_models_pytorch as smp
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import shutil
    WEIGHTS.parent.mkdir(exist_ok=True)
    if not WEIGHTS.exists():
        shutil.copy(hf_hub_download("Yytsi/floorplan-to-3d-walls", "best.safetensors"), WEIGHTS)
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=len(CLASSES))
    state = load_file(WEIGHTS)
    # Verified 5 Sept 2026: the 278 tensors match smp.Unet(resnet34, classes=4)
    # exactly (0 missing, 0 unexpected) and a 512x512 forward pass runs.
    # OPEN 1: the class order (floor, wall, door, window) is taken from the
    # model card and has not yet been confirmed on a known plan; the tensor
    # names do not identify which channel is which. No room count is
    # trusted until it is.
    model.load_state_dict(state, strict=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.eval().to(device), device


def letterbox(img, size=512):
    """Pad to square then resize; returns array in [0,1] and the scale."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    arr = np.asarray(canvas.resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    return arr


def predict(model, device, arr):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()   # (4, H, W)
    return probs


def rooms_from_floor_mask(floor_mask):
    """OPEN 2: how a room is counted. Current rule: connected components of
    the floor class larger than 0.5 percent of the image. Doors join rooms,
    so the wall class matters, and whether "room" is the field an
    underwriter actually needs is part of the question."""
    from scipy import ndimage  # noqa: F401  (scipy comes with scikit-learn)
    labeled, n = ndimage.label(floor_mask)
    sizes = ndimage.sum(floor_mask, labeled, range(1, n + 1))
    return int((sizes > 0.005 * floor_mask.size).sum())


def main():
    MASKS.mkdir(parents=True, exist_ok=True)
    ex = pd.read_csv(EXTR)
    fp_pages = ex[(ex.field == "drawing_type") & (ex.predicted == "floor_plan")][["reference", "sheet"]].drop_duplicates()
    model, device = load_model()
    rows = []
    for _, p in fp_pages.iterrows():
        doc_index, page = str(p.sheet).split("-")          # sheet = document index + page, e.g. 03-1
        pngs = list(PAGES.glob(f"{p.reference}__{doc_index}_*__p{page}.png"))
        if not pngs:
            continue
        img = Image.open(pngs[0]).convert("RGB")
        probs = predict(model, device, letterbox(img))
        pred = probs.argmax(0)
        # OPEN 3: page-level confidence. Current rule: mean softmax
        # probability of the winning class over wall and floor pixels.
        conf = float(probs.max(0)[np.isin(pred, [0, 1])].mean())
        n_rooms = rooms_from_floor_mask(pred == 0)
        wall_frac = float((pred == 1).mean())
        Image.fromarray((pred * 60).astype(np.uint8)).save(MASKS / (pngs[0].stem + "_mask.png"))
        rows += [[p.reference, p.sheet, "room_count", n_rooms, round(conf, 3)],
                 [p.reference, p.sheet, "wall_fraction", round(wall_frac, 3), round(conf, 3)]]
        print(f"{pngs[0].name}: rooms {n_rooms}, wall fraction {wall_frac:.3f}, conf {conf:.3f}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reference", "sheet", "field", "predicted", "confidence"]); w.writerows(rows)
    print(f"wrote {OUT}")
    # OPEN 4: segmentation.csv is merged into extractions.csv before
    # scripts/evaluate_fields.py runs, so room_count is scored against the
    # annotations like every other field.


if __name__ == "__main__":
    main()
