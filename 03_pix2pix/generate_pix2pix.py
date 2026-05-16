"""
Pix2Pix Synthetic Image Generator (Phase 2)
============================================
Uses the trained Pix2Pix generator to synthesise N full 640×640 defect
images from clean templates, copying the REAL label file from the matched
defective image so every synthetic output has a guaranteed correct tight
bounding box annotation.

layout written to:
    gan_synthetic/deeppcb_pix2pix/
        images/   px2px_000000.png ...
        labels/   px2px_000000.txt ...

Usage:
    python generate_pix2pix.py --n 1000
    python generate_pix2pix.py --n 500 --device 0
"""

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/wv/omaadeaj_nobackup/downloads/GAN/.ultralytics")

import argparse, shutil, random
from pathlib import Path
import numpy as np
import torch
import cv2

GAN_DIR   = Path("/wv/omaadeaj_nobackup/downloads/GAN")
CKPT_DIR  = GAN_DIR / "pix2pix_checkpoints"
SYNTH_DIR = GAN_DIR / "gan_synthetic" / "deeppcb_pix2pix"
IMG_SIZE  = 256   # must match training

# YOLO-format label directories (all splits, searched in order)
YOLO_LABEL_DIRS = [
    GAN_DIR / "DeepPcb dataset" / "deeppcb_yolo" / "train" / "labels",
    GAN_DIR / "DeepPcb dataset" / "deeppcb_yolo" / "valid" / "labels",
    GAN_DIR / "DeepPcb dataset" / "deeppcb_yolo" / "test"  / "labels",
]


def find_yolo_label(stem: str) -> Path:
    """Return the YOLO-format label for a raw DeepPCB image stem.

    Raw stem (e.g. '20085000') maps to YOLO label '20085000_test.txt'.
    Searches across train/valid/test label directories.
    """
    yolo_name = stem + "_test.txt"
    for d in YOLO_LABEL_DIRS:
        p = d / yolo_name
        if p.exists():
            return p
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",       type=int, default=1000, help="Number of images to generate")
    p.add_argument("--device",  default="0")
    p.add_argument("--seed",    type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")

    (SYNTH_DIR / "images").mkdir(parents=True, exist_ok=True)
    (SYNTH_DIR / "labels").mkdir(parents=True, exist_ok=True)

    # Load generator
    import sys; sys.path.insert(0, str(GAN_DIR))
    from train_pix2pix import UNetGenerator
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(CKPT_DIR / "generator_final.pt", map_location=device))
    G.eval()

    # Collect all template/label pairs from trainval split
    raw_pcb = GAN_DIR / "DeepPcb dataset" / "DeepPCB-master" / "PCBData"
    pairs   = []
    skipped = 0
    with open(raw_pcb / "trainval.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_rel  = parts[0]
            stem     = Path(img_rel).stem
            group    = Path(img_rel).parts[0]
            subdir   = Path(img_rel).parts[1]
            temp_img = raw_pcb / group / subdir / f"{stem}_temp.jpg"
            # Use YOLO-format label (normalized coords, class 0-5)
            # NOT the raw DeepPCB label (pixel coords, class 1-6)
            yolo_lbl = find_yolo_label(stem)
            if temp_img.exists() and yolo_lbl is not None:
                pairs.append((str(temp_img), str(yolo_lbl), stem))
            else:
                skipped += 1
    if skipped:
        print(f"  [WARN] {skipped} entries skipped (template or YOLO label not found)")

    if not pairs:
        raise RuntimeError("No pairs found. Check DeepPCB-master path.")

    print(f"\n  Found {len(pairs)} source pairs. Generating {args.n} synthetic images …\n")

    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    generated = 0
    while generated < args.n:
        temp_path, lbl_src, stem = random.choice(pairs)

        # Load and preprocess template
        temp = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        if temp is None:
            continue
        orig_h, orig_w = temp.shape
        temp_256 = cv2.resize(temp, (IMG_SIZE, IMG_SIZE))
        inp = tf(temp_256[:, :, None]).unsqueeze(0).to(device)

        # Generate defective image
        with torch.no_grad():
            out = G(inp)   # 1×1×256×256 in [-1,1]
        out_np = ((out.squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        # Scale back to 640×640
        out_640 = cv2.resize(out_np, (640, 640), interpolation=cv2.INTER_LANCZOS4)

        fname = f"px2px_{generated:07d}"
        cv2.imwrite(str(SYNTH_DIR / "images" / f"{fname}.png"), out_640)

        # Copy the YOLO-format label (normalised coords, class 0-5 → scale-invariant)
        shutil.copy2(lbl_src, SYNTH_DIR / "labels" / f"{fname}.txt")

        generated += 1
        if generated % 100 == 0:
            print(f"  Generated {generated}/{args.n} …")

    print(f"\n  Done. {generated} images saved to {SYNTH_DIR}")
    print(f"\n  Next:")
    print(f"    python merge_and_retrain.py --dataset deeppcb --name pcb_defects_pix2pix\n")
    print(f"  (point merge_and_retrain at gan_synthetic/deeppcb_pix2pix for Phase 2 merge)")


if __name__ == "__main__":
    main()
