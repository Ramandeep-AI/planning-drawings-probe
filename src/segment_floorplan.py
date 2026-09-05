"""Stage 3: a pretrained floor-plan segmentation baseline, in PyTorch on MPS.

Model: Yytsi/floorplan-to-3d-walls on Hugging Face. A U-Net with a
ResNet-34 encoder (segmentation_models_pytorch), four classes
(floor, wall, door, window), MIT-licensed weights, trained on CubiCasa5K,
which is CC BY-NC-SA 4.0. That lineage is disclosed in the README and
is one reason this study is non-commercial.

THIS IS YOURS TO FINISH. Decisions marked TODO are yours.

Run:  env/bin/python -m src.segment_floorplan
Reads:  data/processed/extractions.csv (to find pages OCR'd as floor plans)
        data/processed/pages/*.png
Writes: data/processed/segmentation.csv (reference,page,field,predicted,confidence)
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
    # TODO 1 - confirm the class ORDER (floor, wall, door, window) against
    # the model card before trusting any room count; the key names do not
    # tell you which channel is which.
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
    """TODO 2 - decide how a room is counted. Placeholder: connected
    components of the floor class larger than 0.5% of the image. Doors
    connect rooms, so the wall class matters. Think about what an
    underwriter means by a room and whether that is even the right field."""
    from scipy import ndimage  # noqa: F401  (scipy comes with scikit-learn)
    labeled, n = ndimage.label(floor_mask)
    sizes = ndimage.sum(floor_mask, labeled, range(1, n + 1))
    return int((sizes > 0.005 * floor_mask.size).sum())


def main():
    MASKS.mkdir(parents=True, exist_ok=True)
    ex = pd.read_csv(EXTR)
    fp_pages = ex[(ex.field == "drawing_type") & (ex.predicted == "floor_plan")][["reference", "page"]].drop_duplicates()
    model, device = load_model()
    rows = []
    for _, p in fp_pages.iterrows():
        pngs = list(PAGES.glob(f"{p.reference}__*__p{p.page}.png"))
        if not pngs:
            continue
        img = Image.open(pngs[0]).convert("RGB")
        probs = predict(model, device, letterbox(img))
        pred = probs.argmax(0)
        # TODO 3 - a confidence for the page-level field. Placeholder: mean
        # softmax probability of the winning class over wall+floor pixels.
        conf = float(probs.max(0)[np.isin(pred, [0, 1])].mean())
        n_rooms = rooms_from_floor_mask(pred == 0)
        wall_frac = float((pred == 1).mean())
        Image.fromarray((pred * 60).astype(np.uint8)).save(MASKS / (pngs[0].stem + "_mask.png"))
        rows += [[p.reference, p.page, "room_count", n_rooms, round(conf, 3)],
                 [p.reference, p.page, "wall_fraction", round(wall_frac, 3), round(conf, 3)]]
        print(f"{pngs[0].name}: rooms {n_rooms}, wall fraction {wall_frac:.3f}, conf {conf:.3f}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reference", "page", "field", "predicted", "confidence"]); w.writerows(rows)
    print(f"wrote {OUT}")
    # TODO 4 - merge segmentation.csv into extractions.csv before running
    # scripts/evaluate_fields.py, so room_count is scored against your
    # annotations like every other field.


if __name__ == "__main__":
    main()
