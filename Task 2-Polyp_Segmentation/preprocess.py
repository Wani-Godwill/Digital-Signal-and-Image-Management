import os
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Config 
IMG_SIZE      = 256
MASK_THRESH   = 127       # binarise GT masks at this threshold
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15      # remainder
AUG_PER_IMAGE = 3         # extra augmented copies per training image
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Image helpers

def load_image(path: str) -> np.ndarray:
    """Load RGB image, resize to IMG_SIZE×IMG_SIZE, return float32 [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    """
    Load ground-truth mask, resize, binarise to 0/255 uint8.
    Kvasir-SEG masks are RGB PNGs where the polyp region is white.
    We convert to grayscale first to handle any colour encoding.
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, MASK_THRESH, 255, cv2.THRESH_BINARY)
    return mask  # uint8, values 0 or 255


# Augmentation

def augment_pair(img: np.ndarray, mask: np.ndarray) -> list[tuple]:
    """
    Return AUG_PER_IMAGE (image, mask) pairs with consistent transforms.
    Both image and mask get the same spatial transforms; only the image
    gets colour jitter.
    Transforms: h-flip, rotation ±20°, brightness/contrast jitter.
    """
    h, w = img.shape[:2]
    pairs = []

    for _ in range(AUG_PER_IMAGE):
        aug_img  = img.copy()
        aug_mask = mask.copy()

        # Horizontal flip
        if random.random() > 0.5:
            aug_img  = cv2.flip(aug_img, 1)
            aug_mask = cv2.flip(aug_mask, 1)

        # Rotation ±20°
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug_img  = cv2.warpAffine(aug_img,  M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        aug_mask = cv2.warpAffine(aug_mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)
        # Re-binarise mask after rotation (interpolation adds mid-values)
        _, aug_mask = cv2.threshold(aug_mask, MASK_THRESH, 255, cv2.THRESH_BINARY)

        # Brightness jitter (image only, ±20 %)
        factor = random.uniform(0.80, 1.20)
        aug_img = np.clip(aug_img * factor, 0.0, 1.0).astype(np.float32)

        # Contrast jitter (image only)
        contrast = random.uniform(0.85, 1.15)
        aug_img = np.clip((aug_img - 0.5) * contrast + 0.5, 0.0, 1.0).astype(np.float32)

        pairs.append((aug_img, aug_mask))

    return pairs


# YOLO label generation

def mask_to_yolo(mask: np.ndarray, class_id: int = 0) -> list[str]:
    """
    Derive YOLO bounding-box labels from the ground-truth binary mask.
    Returns a list of label strings:  "<class_id> cx cy w h"  (normalised 0–1)

    One polyp → usually one bounding box.
    Multiple disconnected regions → one box per contour.
    Empty mask (no polyp) → empty list (background sample).
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    min_area = 100   # ignore tiny noise blobs

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx = (bx + bw / 2) / w
        cy = (by + bh / 2) / h
        nw = bw / w
        nh = bh / h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


# Save helpers

def save_img(img_float: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (img_float * 255).astype(np.uint8))


def save_mask_file(mask: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mask)


def save_label(lines: list[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# Core processing     

def process_one(stem: str, img: np.ndarray, mask: np.ndarray,
                split: str, out: Path, suffix: str = ""):
    """Write one (image, mask) pair to both U-Net and YOLO output trees."""
    name = f"{stem}{suffix}"

    # U-Net
    save_img(img,  str(out / "unet" / "images" / split / f"{name}.jpg"))
    save_mask_file(mask, str(out / "unet" / "masks"  / split / f"{name}.png"))

    # YOLO
    save_img(img,  str(out / "yolo" / "images" / split / f"{name}.jpg"))
    labels = mask_to_yolo(mask)
    save_label(labels, str(out / "yolo" / "labels" / split / f"{name}.txt"))


# YAML

def write_yaml(out: Path):
    p = out / "yolo" / "kvasir.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"""# Kvasir-SEG — YOLOv8 dataset config
path: {str(out / 'yolo')}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: polyp
""")
    print(f"[YOLO] YAML written → {p}")


# Entry point        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./data")
    parser.add_argument("--output_dir",  default="./processed")
    parser.add_argument("--no_augment",  action="store_true")
    args = parser.parse_args()

    src  = Path(args.dataset_dir)
    out  = Path(args.output_dir)

    img_dir  = src / "images"
    mask_dir = src / "masks"

    stems = sorted([p.stem for p in img_dir.glob("*.jpg")] +
                   [p.stem for p in img_dir.glob("*.png")])
    if not stems:
        print(f"[ERROR] No images found in {img_dir}")
        return

    print(f"[INFO] Found {len(stems)} image–mask pairs")
    print(f"[INFO] Output → {out}")
    print(f"[INFO] Image size: {IMG_SIZE}×{IMG_SIZE}")

    # Shuffle and split
    random.shuffle(stems)
    n     = len(stems)
    n_tr  = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    split_map = {}
    for i, s in enumerate(stems):
        if i < n_tr:
            split_map[s] = "train"
        elif i < n_tr + n_val:
            split_map[s] = "val"
        else:
            split_map[s] = "test"

    # Clean previous run
    if out.exists():
        shutil.rmtree(out)

    # Process
    aug = not args.no_augment
    for stem in tqdm(stems, desc="Processing"):
        # Find image (may be .jpg or .png)
        img_path  = img_dir  / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.jpg"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.png"

        img  = load_image(str(img_path))
        mask = load_mask(str(mask_path))
        sp   = split_map[stem]

        process_one(stem, img, mask, sp, out)

        if aug and sp == "train":
            for i, (aug_img, aug_mask) in enumerate(augment_pair(img, mask)):
                process_one(stem, aug_img, aug_mask, sp, out, suffix=f"_aug{i}")

    write_yaml(out)

    # Stats
    print("\nSplit statistics ─────────────────────────────")
    for model in ["unet", "yolo"]:
        print(f"\n  {model.upper()}")
        for sp in ["train", "val", "test"]:
            d = out / model / "images" / sp
            n_imgs = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"    {sp:5s}: {n_imgs:4d} images")

    print(f"\n[DONE]")
    print(f"  U-Net data : {out}/unet/")
    print(f"  YOLO data  : {out}/yolo/")
    print(f"  YOLO YAML  : {out}/yolo/kvasir.yaml")


if __name__ == "__main__":
    main()