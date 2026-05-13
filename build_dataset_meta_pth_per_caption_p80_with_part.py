import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

VOC116_PART_CLASS = ["aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", 
               "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", 
               "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", 
               "bottle's body", "bottle's cap", 
               "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", 
               "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", 
               "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", 
               "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn", 
               "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", 
               "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", "horse's muzzle", "horse's hoof", 
               "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", 
               "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand",
               "pottedplant's pot", "pottedplant's plant", 
               "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", 
               "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", 
               "train's coach", "tvmonitor's screen"]
PART_NAME_TO_ID = {name: idx for idx, name in enumerate(VOC116_PART_CLASS)}

VOC116_OBJ_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PC59_CLASSES = [
    "aeroplane", "bag", "bed", "bedclothes", "bench",
    "bicycle", "bird", "boat", "book", "bottle",
    "building", "bus", "cabinet", "car", "cat",
    "ceiling", "chair", "cloth", "computer", "cow",
    "cup", "curtain", "dog", "door", "fence",
    "floor", "flower", "food", "grass", "ground",
    "horse", "keyboard", "light", "motorbike", "mountain",
    "mouse", "person", "plate", "platform", "pottedplant",
    "road", "rock", "sheep", "shelves", "sidewalk",
    "sign", "sky", "snow", "sofa", "table",
    "track", "train", "tree", "truck", "tvmonitor",
    "wall", "water", "window", "wood",
]

CLASSES = VOC116_OBJ_CLASSES

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def build_prompts(class_name: str) -> List[str]:
    return [template.format(class_name) for template in IMAGENET_TEMPLATES]


def build_part_lookup(part_classes: List[str]) -> Dict[str, List[str]]:
    part_lookup: Dict[str, List[str]] = {}
    for full_name in part_classes:
        obj_name, _ = full_name.split("'s ", 1)
        part_lookup.setdefault(obj_name, []).append(full_name)
    return part_lookup


OBJ_TO_PART_CLASS_NAMES = build_part_lookup(VOC116_PART_CLASS)


def find_image_for_mask(mask_path: Path, images_dir: Path) -> Optional[Path]:
    stem = mask_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None



def extract_present_class_ids(mask_path: Path, num_classes: int, one_based_masks: bool) -> List[int]:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    uniq = sorted(int(x) for x in np.unique(mask))

    present: List[int] = []
    for uid in uniq:
        if uid == 255:
            continue
        if one_based_masks:
            # background is 0, class ids are 1..num_classes
            if 1 <= uid <= num_classes:
                present.append(uid - 1)
        else:
            # class ids are 0..num_classes-1; values outside that range ignored
            if 0 <= uid < num_classes:
                present.append(uid)
    return present



def build_split(
    split: str,
    ann_dir: Path,
    img_dir: Path,
    with_background: bool,
) -> Dict:
    masks_dir = ann_dir / split
    images_dir = img_dir / split
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No PNG masks found in: {masks_dir}")

    images: List[Dict] = []
    annotations: List[Dict] = []
    skipped_no_img: List[str] = []
    skipped_empty: List[str] = []

    for img_id, mask_path in enumerate(tqdm(mask_files, desc=f"Building {split}")):
        image_path = find_image_for_mask(mask_path, images_dir)
        if image_path is None:
            skipped_no_img.append(mask_path.name)
            continue

        present_ids = extract_present_class_ids(mask_path, len(CLASSES), with_background)
        if not present_ids:
            skipped_empty.append(mask_path.name)
            continue

        images.append(
            {
                "id": len(images),
                "file_name": str(image_path),
                "seg_file_name": str(mask_path),
                "split": split,
            }
        )
        current_image_id = images[-1]["id"]

        for class_idx in present_ids:
            class_name = CLASSES[class_idx]
            part_class_name = OBJ_TO_PART_CLASS_NAMES.get(class_name, [])
            part_category_id = [PART_NAME_TO_ID[name] for name in part_class_name]
            part_caption = [build_prompts(part_name) for part_name in part_class_name]

            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": current_image_id,
                    "category_id": class_idx,
                    "class_name": class_name,
                    "part_category_id": part_category_id,
                    "part_class_name": part_class_name,
                    "caption": build_prompts(class_name),
                    "part_caption": part_caption,
                }
            )

    result = {"images": images, "annotations": annotations}

    print(f"[{split}] images kept: {len(images)}")
    print(f"[{split}] annotations: {len(annotations)}")
    print(f"[{split}] skipped (no matching image): {len(skipped_no_img)}")
    print(f"[{split}] skipped (empty/no valid classes): {len(skipped_empty)}")
    if images:
        print(f"[{split}] example image: {images[0]}")
    if annotations:
        print(f"[{split}] example annotation: {annotations[0]}")
    if skipped_no_img:
        print(f"[{split}] first missing image mask: {skipped_no_img[0]}")
    if skipped_empty:
        print(f"[{split}] first empty mask: {skipped_empty[0]}")

    return result



def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset's meta pth file.")
    parser.add_argument("--ann_dir", type=str, required=True, help="Dir of annotations_detectron2_obj, containing train/ and val/.")
    parser.add_argument("--img_dir", type=str, required=True, help="Dir of images, containing train/ and val/.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for train/val .pth files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process.")
    parser.add_argument("--with_background", action="store_true", help="Set this if mask labels are 1..20 with 0 as background.")
    parser.add_argument(
        "--output_name",
        type=str,
        default="{split}_meta.pth",
        help="Output file name.",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        result = build_split(
            split=split,
            ann_dir=ann_dir,
            img_dir=img_dir,
            with_background=args.with_background,
        )
        out_path = out_dir / args.output_name.format(split=split)
        torch.save(result, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
