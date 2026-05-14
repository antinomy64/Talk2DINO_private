import argparse
import clip
import json
import math
import os
import requests
import webdataset as wds
import tarfile
import timm
import torch
import torchvision.transforms as T
import numpy as np

from io import BytesIO
from src.hooks import get_self_attention, process_self_attention, get_second_last_out, get_vit_out, get_dinov1_patches, feats
from src.webdatasets_util import cc2coco_format, create_webdataset_tar, read_coco_format_wds
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken

import sys
def generate_caption(model, processor, images, prompt="a photography of"):
    image_token = AddedToken("<image>", normalized=False, special=True)
    processor.tokenizer.add_tokens([image_token], special_tokens=True)

    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
    model.config.image_token_index = len(processor.tokenizer) - 1
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(next(model.parameters()).device)
    inputs['pixel_values'] = inputs['pixel_values'].float()

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [x.strip() for x in generated_text]

def resolve_existing_path(path_str, data_dir=None):
    if path_str and os.path.exists(path_str):
        return path_str
    if path_str and data_dir is not None:
        candidate = os.path.join(data_dir, path_str)
        if os.path.exists(candidate):
            return candidate
    return path_str


def load_pil_image_from_img_info(img_info, data_dir=None):
    file_name = img_info.get('file_name', '')
    coco_url = img_info.get('coco_url', '')

    if file_name and os.path.exists(file_name):
        pil_img = Image.open(file_name)
    elif file_name and data_dir is not None and os.path.exists(os.path.join(data_dir, file_name)):
        pil_img = Image.open(os.path.join(data_dir, file_name))
    elif file_name and 'http' in file_name:
        pil_img = Image.open(BytesIO(requests.get(file_name).content))
    elif file_name and 'train' in file_name:
        pil_img = Image.open(os.path.join(data_dir, f"train2014/{file_name}"))
    elif file_name and 'val' in file_name:
        pil_img = Image.open(os.path.join(data_dir, f"val2014/{file_name}"))
    elif file_name and 'test' in file_name:
        pil_img = Image.open(os.path.join(data_dir, f"test2014/{file_name}"))
    elif coco_url and 'train' in coco_url:
        pil_img = Image.open(os.path.join(data_dir, f"train2017/{file_name}"))
    elif coco_url and 'val' in coco_url:
        pil_img = Image.open(os.path.join(data_dir, f"val2017/{file_name}"))
    else:
        raise FileNotFoundError(f"Cannot resolve image path for image id={img_info.get('id', 'unknown')}")

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img


def load_binary_mask(seg_path, category_id, with_background=False, data_dir=None):
    seg_path = resolve_existing_path(seg_path, data_dir)
    mask = np.array(Image.open(seg_path))
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask_value = category_id + 1 if with_background else category_id
    binary = (mask == mask_value).astype(np.uint8)
    return binary


def square_crop_from_binary_mask(pil_img, binary_mask, expand_ratio=1.2):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # fallback to whole image
        w, h = pil_img.size
        return pil_img, (0, 0, w, h)

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max() + 1, ys.max() + 1

    w_box = x2 - x1
    h_box = y2 - y1
    side = max(w_box, h_box) * expand_ratio

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))

    img_w, img_h = pil_img.size
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    crop_img = pil_img.crop((left, top, right, bottom))
    return crop_img, (left, top, right, bottom)

def run_dinov2_extraction(model_name, data_dir, ann_path, batch_size, resize_dim=518, crop_dim=518, out_path=None, 
                          write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                          extract_cls=False, extract_avg_self_attn=False, extract_second_last_out=False,
                          extract_patch_tokens=False, extract_self_attn_maps=False, extract_disentangled_self_attn=False,
                          blip_model_name=None, extract_cropaug_patch_tokens=False, with_background=False, crop_expand_ratio=1.2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
        embed_dim = 1024
    elif 'vitb' in model_name or '_base' in model_name or 'ViT-B' in model_name:
        embed_dim = 768
    elif 'vits' in model_name or 'vit_small' in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")

    scale = 0.125
    batch_size_ = batch_size

    # loading the model
    if 'dinov2' in model_name:
        model_family = 'facebookresearch/dinov2'
        model = torch.hub.load(model_family, model_name)
        image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        num_attn_heads = model.num_heads

    elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=crop_dim
        )
        data_config = timm.data.resolve_model_data_config(model)
        image_transforms = timm.data.create_transform(**data_config, is_training=False)

        if 'dinov3' in model_name:
            data_config['input_size'] = (3, crop_dim, crop_dim)
            image_transforms = timm.data.create_transform(**data_config, is_training=False)
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_global_tokens = 5
            num_tokens = num_global_tokens + num_patch_tokens
        elif 'mae' in model_name or 'dino' in model_name or 'beit' in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = 1 + num_patch_tokens
        elif 'sam' in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = num_patch_tokens
            num_global_tokens = 0
            model.blocks[-1].register_forward_hook(get_vit_out)
        elif 'clip' in model_name:
            crop_dim = resize_dim = 224
            num_patch_tokens = crop_dim // 16 * crop_dim // 16 if 'vit_base' in model_name else crop_dim // 14 * crop_dim // 14
            num_tokens = 1 + num_patch_tokens  
        num_attn_heads = model.blocks[-1].attn.num_heads
    elif 'ViT' in model_name:
        model, image_transforms = clip.load(model_name, device)
        num_attn_heads = model.num_heads
    else:
        raise Exception("Unknown ViT model")

    model.eval()
    model.to(device)

    if blip_model_name is not None:
        blip_processor = Blip2Processor.from_pretrained(blip_model_name)
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip_model_name, torch_dtype=torch.float16,
        ).to(device)
        blip_processor.num_query_tokens = blip_model.config.num_query_tokens

    if os.path.isdir(ann_path):
        data = cc2coco_format(ann_path, n_in_splits, in_batch_offset)
    elif '.tar' in ann_path:
        data = read_coco_format_wds(ann_path)
    else:
        if ann_path.endswith('.json'):
            print("Loading the annotations JSON")
            with open(ann_path, 'r') as f:
                data = json.load(f)
        else:
            print("Loading the annotations PTH")
            data = torch.load(ann_path)

    if extract_second_last_out:
        model.blocks[-2].register_forward_hook(get_second_last_out)
    if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        if 'beit' in model_name:
            model.blocks[-1].attn.qkv_bias_separate = True

    print("Starting the features extraction...")
    n_imgs = len(data['images'])
    n_batch = math.ceil(n_imgs / batch_size)
    n_errors = 0

    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        batch_size_ = end - start
        raw_imgs = []
        failed_ids = []
        for j in range(start, end):
            if 'jpg' in data['images'][j]:
                pil_img = data['images'][j]['jpg']
                del data['images'][j]['jpg']
            else:
                file_name = data['images'][j].get('file_name', '')
                coco_url = data['images'][j].get('coco_url', '')

                if file_name and os.path.exists(file_name):
                    pil_img = Image.open(file_name)
                elif file_name and data_dir is not None and os.path.exists(os.path.join(data_dir, file_name)):
                    pil_img = Image.open(os.path.join(data_dir, file_name))
                elif file_name and 'http' in file_name:
                    try:
                        pil_img = Image.open(BytesIO(requests.get(file_name).content))
                    except Exception:
                        pil_img = Image.new("RGB", (224, 224))
                        failed_ids.append(j)
                        n_errors += 1
                elif file_name and 'train' in file_name:
                    pil_img = Image.open(os.path.join(data_dir, f"train2014/{file_name}"))
                elif file_name and 'val' in file_name:
                    pil_img = Image.open(os.path.join(data_dir, f"val2014/{file_name}"))
                elif file_name and 'test' in file_name:
                    pil_img = Image.open(os.path.join(data_dir, f"test2014/{file_name}"))
                elif coco_url and 'train' in coco_url:
                    pil_img = Image.open(os.path.join(data_dir, f"train2017/{file_name}"))
                elif coco_url and 'val' in coco_url:
                    pil_img = Image.open(os.path.join(data_dir, f"val2017/{file_name}"))
                else:
                    pil_img = Image.new("RGB", (224, 224))
                    failed_ids.append(j)
                    n_errors += 1

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            raw_imgs.append(pil_img)

        batch_imgs = torch.stack([image_transforms(img) for img in raw_imgs]).to(device)

        with torch.no_grad():
            if 'dinov2' in model_name:
                outs = model(batch_imgs, is_training=True)
            elif 'dinov3' in model_name:
                output = model.forward_features(batch_imgs)
                outs = {
                    'x_norm_clstoken': output[:, 0, :],
                    'x_norm_patchtokens': output[:, 5:, :],
                }
            elif 'mae' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
                output = model.forward_features(batch_imgs)
                outs = {
                    'x_norm_clstoken': output[:, 0, :],
                    'x_norm_patchtokens': output[:, 1:, :],
                }
            elif 'sam' in model_name:
                sam_output = model.forward_features(batch_imgs)
                if extract_cls:
                    cls = model.forward_head(sam_output, pre_logits=True)
                else:
                    cls = None
                outs = {
                    'x_norm_clstoken': cls,
                    'x_norm_patchtokens': feats['vit_out'].reshape(batch_size_, num_patch_tokens, embed_dim)
                }
            elif 'ViT' in model_name:
                outs = {
                    'x_norm_clstoken': model.encode_image(batch_imgs)
                }

            cls_token = outs['x_norm_clstoken']
            if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
                self_attn, self_attn_maps = process_self_attention(feats['self_attn'], (end-start), num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=True)
            if extract_avg_self_attn:
                avg_self_attn_token = (self_attn.unsqueeze(-1) * outs['x_norm_patchtokens']).mean(dim=1)
            if extract_disentangled_self_attn:
                self_attn_maps = self_attn_maps.softmax(dim=-1)
                disentangled_self_attn = (outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
            if extract_second_last_out:
                second_last_cls = feats['second_last_out'][:, 0, :]
            if blip_model_name is not None:
                new_capts = generate_caption(blip_model, blip_processor, raw_imgs)

        for j in range(start, end):
            if j in failed_ids:
                continue

            if extract_cls or (not extract_avg_self_attn and not extract_second_last_out):
                data['images'][j]['dino_features'] = cls_token[j - start].to(dtype=torch.float16, device='cpu')
            if extract_avg_self_attn:
                data['images'][j]['avg_self_attn_out'] = avg_self_attn_token[j - start].to(dtype=torch.float16, device='cpu')
            if extract_second_last_out:
                data['images'][j]['second_last_out'] = second_last_cls[j - start].to(dtype=torch.float16, device='cpu')
            if extract_patch_tokens:
                data['images'][j]['patch_tokens'] = outs['x_norm_patchtokens'][j - start].to(dtype=torch.float16, device='cpu')
            if extract_self_attn_maps:
                data['images'][j]['self_attn_maps'] = self_attn_maps[j - start].to(dtype=torch.float16, device='cpu')
            if extract_disentangled_self_attn:
                data['images'][j]['disentangled_self_attn'] = disentangled_self_attn[j - start].to(dtype=torch.float16, device='cpu')
            if blip_model_name is not None and j < len(data.get('annotations', [])):
                data['annotations'][j]['caption'] = new_capts[j - start]

    print("Feature extraction done!")
    print(f"Failed to extract {n_errors} of {len(data['images'])}")

    if extract_cropaug_patch_tokens:
        print("Starting crop-aug patch token extraction...")

        images_by_id = {img["id"]: img for img in data["images"]}
        anns = data["annotations"]
        n_anns = len(anns)
        n_ann_batch = math.ceil(n_anns / batch_size)

        for i in tqdm(range(n_ann_batch), desc="CropAug annotation batches"):
            start = i * batch_size
            end = start + batch_size if i < n_ann_batch - 1 else n_anns

            raw_crop_imgs = []
            valid_ann_indices = []
            crop_boxes = []

            for j in range(start, end):
                ann = anns[j]
                img_info = images_by_id[ann["image_id"]]

                try:
                    pil_img = load_pil_image_from_img_info(img_info, data_dir=data_dir)
                    seg_path = img_info["seg_file_name"]
                    binary_mask = load_binary_mask(
                        seg_path,
                        ann["category_id"],
                        with_background=with_background,
                        data_dir=data_dir,
                    )

                    crop_img, crop_box = square_crop_from_binary_mask(
                        pil_img,
                        binary_mask,
                        expand_ratio=crop_expand_ratio,
                    )

                    raw_crop_imgs.append(crop_img)
                    valid_ann_indices.append(j)
                    crop_boxes.append(crop_box)

                except Exception as e:
                    print(f"[CropAug] skip ann {j}: {e}")

            if len(raw_crop_imgs) == 0:
                continue

            batch_crop_imgs = torch.stack([image_transforms(img) for img in raw_crop_imgs]).to(device)

            with torch.no_grad():
                if 'dinov2' in model_name:
                    crop_outs = model(batch_crop_imgs, is_training=True)
                elif 'dinov3' in model_name:
                    output = model.forward_features(batch_crop_imgs)
                    crop_outs = {
                        'x_norm_clstoken': output[:, 0, :],
                        'x_norm_patchtokens': output[:, 5:, :],
                    }
                elif 'mae' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
                    output = model.forward_features(batch_crop_imgs)
                    crop_outs = {
                        'x_norm_clstoken': output[:, 0, :],
                        'x_norm_patchtokens': output[:, 1:, :],
                    }
                elif 'sam' in model_name:
                    _ = model.forward_features(batch_crop_imgs)
                    crop_outs = {
                        'x_norm_patchtokens': feats['vit_out'].reshape(len(raw_crop_imgs), num_patch_tokens, embed_dim)
                    }
                else:
                    raise Exception("cropaug currently supports ViT-like patch-token models only")

            for local_idx, ann_idx in enumerate(valid_ann_indices):
                anns[ann_idx]['cropaug_patch_tokens'] = crop_outs['x_norm_patchtokens'][local_idx].to(dtype=torch.float16, device='cpu')
                anns[ann_idx]['cropaug_box_xyxy'] = torch.tensor(crop_boxes[local_idx], dtype=torch.long)

    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(data, out_path, num_shards, out_offset)
    else:
        if out_path is None:
            out_path = os.path.splitext(ann_path)[0] + '.pth' 
        torch.save(data, out_path)
        print(f"Features saved at {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default="coco/test1k.json", help="Directory of the annotation file") 
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--data_dir', type=str, default="../coco/", help="Directory of the images") 
    parser.add_argument('--blip_model', type=str, default=None, help="BLIP model to recaption with. If None we will use standard captions. For CC3M we use Salesforce/blip2-opt-6.7b-coco") 
    parser.add_argument('--model', type=str, default="dinov2_vitl14_reg", help="Model configuration to extract features from")
    parser.add_argument('--resize_dim', type=int, default=518, help="Resize dimension")
    parser.add_argument('--crop_dim', type=int, default=518, help="Crop dimension")
    parser.add_argument('--extract_cls', default=False, action="store_true", help="If setted, adds the CLS token to the output")
    parser.add_argument('--extract_avg_self_attn', default=False, action="store_true", help="If setted, adds the token obtained by weighting using the self attention to the output")
    parser.add_argument('--extract_second_last_out', default=False, action="store_true", help="If setted, adds the second last CLS to the output") 
    parser.add_argument('--extract_patch_tokens', default=False, action="store_true", help="If setted, we extract all the patch tokens") 
    parser.add_argument('--extract_self_attn_maps', default=False, action="store_true", help="If setted, we extract all the self-attention maps") 
    parser.add_argument('--extract_disentangled_self_attn', default=False, action="store_true", help="If setted, adds the token obtained by weighting using the self attention to the output, without averaging the attention heads") 
    parser.add_argument('--extract_cropaug_patch_tokens', action="store_true", default=False, help="If set, extract object square-crop patch tokens per annotation")
    parser.add_argument('--with_background', action="store_true", default=False, help="Set this if segmentation masks are 1..C with 0 as background")
    parser.add_argument('--crop_expand_ratio', type=float, default=1.2, help="Expand ratio for square object crop")
    parser.add_argument('--out_path', type=str, default=None, help="Pth of the output file, if setted to None. out_pat = ann_path") 
    parser.add_argument('--write_as_wds', action="store_true", default=False, help="If setted, the output will be written as a webdataset") 
    parser.add_argument('--n_shards', type=int, default=25, help="Number of shards in which the webdataset is splitted. Only relevant if --write_as_wds is setted.")
    parser.add_argument('--n_in_split', type=int, default=1, help="Number of splits in which we want to divide the tar files. For example, with 4 n_split we elaborate 332 // 4 = 83 tar files.")
    parser.add_argument('--in_batch_offset', type=int, default=0, help="Of the n_splits in which we have divided tars, we decide which of them elaborate")
    parser.add_argument('--out_offset', type=int, default=0, help="Index of the first shard to save")
    args = parser.parse_args()

    run_dinov2_extraction(args.model, args.data_dir, args.ann_path, args.batch_size, args.resize_dim, args.crop_dim, args.out_path,
                          args.write_as_wds, args.n_shards, args.n_in_split, args.in_batch_offset, args.out_offset,
                          args.extract_cls, args.extract_avg_self_attn, args.extract_second_last_out, args.extract_patch_tokens, args.extract_self_attn_maps,
                          args.extract_disentangled_self_attn, args.blip_model, args.extract_cropaug_patch_tokens, args.with_background, args.crop_expand_ratio)


if __name__ == '__main__':
    main()
