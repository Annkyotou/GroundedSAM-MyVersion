# Step02_instance_segmentation.py (updated with label projection)

import os
import cv2
import time
import torch
import tifffile
import argparse
import numpy as np
from PIL import Image
from math import pi
from plyfile import PlyElement, PlyData

from segment_anything.segment_anything import sam_model_registry, SamPredictor
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap, clean_state_dict
from GroundingDINO.groundingdino.datasets import transforms as T
import GroundingDINO.groundingdino.util.slconfig as slconfig


def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def load_grounded_sam_models(device, config_file, dino_checkpoint, sam_checkpoint, sam_version="vit_h"):
    args = slconfig.SLConfig.fromfile(config_file)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(dino_checkpoint, map_location=device)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval().to(device)

    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    return model, predictor

def segment_image(image_path, model, predictor, device, text_prompt, box_thresh=0.3, text_thresh=0.25):
    print(f"[{timestamp()}] Segmenting image: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor, _ = transform(image_pil, None)

    with torch.no_grad():
        outputs = model(image_tensor[None].to(device), captions=[text_prompt])
    logits = outputs["pred_logits"].sigmoid()[0].cpu()
    boxes = outputs["pred_boxes"][0].cpu()

    mask = logits.max(dim=1)[0] > box_thresh
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(text_prompt)
    phrases = [get_phrases_from_posmap(logit > text_thresh, tokenized, tokenizer) for logit in logits_filt]

    cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(cv_image)
    H, W = image_pil.size[1], image_pil.size[0]

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, cv_image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    labeled_mask = np.zeros((H, W), dtype=np.uint16)
    instance_to_class = {}

    for idx, mask_tensor in enumerate(masks):
        mask_np = mask_tensor.cpu().numpy()[0] > 0
        if np.sum(mask_np) == 0:
            continue
        labeled_mask[mask_np] = idx + 1
        instance_to_class[idx + 1] = phrases[idx]
        print(f"[INFO] Mask {idx + 1}: \"{phrases[idx]}\" | Area: {np.sum(mask_np)} pixels")

    return labeled_mask, instance_to_class

def spherical_projection(points, distances, height, width):
    x, y, z = points
    theta = np.arctan2(y, x)
    phi = np.arccos(z / distances)
    pixel_x = ((theta + pi) / (2 * pi)) * width
    pixel_y = (phi / pi) * height
    return pixel_x.astype(np.int32), pixel_y.astype(np.int32)

def label_projection(mask, used_points, sensor_pos, height):
    rel = (used_points - sensor_pos)
    distances = np.linalg.norm(rel, axis=1)
    width = height * 2
    px, py = spherical_projection(rel.T, distances, height, width)
    instance_ids = np.zeros((used_points.shape[0],), dtype=np.uint16)

    for i in range(len(px)):
        x, y = px[i], py[i]
        if 0 <= x < width and 0 <= y < height:
            instance_id = mask[y, x]
            instance_ids[i] = instance_id
    return instance_ids

def write_labeled_ply(filename, points, colors, instance_ids, instance_to_class):
    unique_classes = sorted(set(
        list(v)[0].strip().lower().rstrip(".") if isinstance(v, set) else v.strip().lower().rstrip(".")
        for v in instance_to_class.values()
    ))
    semantic_to_idx = {label: i + 1 for i, label in enumerate(unique_classes)}

    semantic_labels = np.array([
        semantic_to_idx.get(
            (list(instance_to_class[i])[0] if isinstance(instance_to_class.get(i), set)
             else instance_to_class.get(i, '')).strip().lower().rstrip("."),
            0
        ) for i in instance_ids
    ], dtype=np.uint8)

    vertex_data = np.array([
        (x, y, z, r, g, b, iid, sid)
        for (x, y, z), (r, g, b), iid, sid in zip(points, colors, instance_ids, semantic_labels)
    ], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('instance_id', 'u2'), ('semantic_class', 'u1')
    ])
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(filename)
    print(f"[{timestamp()}] [✓] Saved labeled PLY: {filename}")

def save_mask_and_metadata(labeled_mask, instance_to_class, output_prefix):
    os.makedirs("./outputs", exist_ok=True)
    mask_path = f"./outputs/{output_prefix}_mask.tif"
    tifffile.imwrite(mask_path, labeled_mask)
    print(f"[{timestamp()}] [✓] Saved mask: {mask_path}")

    label_txt = f"./outputs/{output_prefix}_instances.txt"
    with open(label_txt, "w") as f:
        for k, v in instance_to_class.items():
            f.write(f"{k}: {v}\n")
    print(f"[{timestamp()}] [✓] Saved instance metadata: {label_txt}")

def main():
    parser = argparse.ArgumentParser(description="Step 02: Instance Segmentation and Label Projection")
    parser.add_argument("-image", required=True, help="Path to the input panorama image")
    parser.add_argument("-used_points", required=True, help=".npz file with used_points and sensor_pos")
    parser.add_argument("--text_prompt", default="Ceiling . Wall . Door . Window . Floor", help="Text prompt for segmentation")
    parser.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino_ckpt", default="grounded_sam_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_ckpt", default="grounded_sam_checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, predictor = load_grounded_sam_models(
        device, args.config, args.dino_ckpt, args.sam_ckpt
    )

    labeled_mask, instance_to_class = segment_image(
        args.image, model, predictor, device, args.text_prompt
    )
    save_mask_and_metadata(labeled_mask, instance_to_class, base_name)

    data = np.load(args.used_points)
    used_points = data['used_points']
    sensor_pos = data['sensor_pos']
    colors = data['colors']

    instance_ids = label_projection(labeled_mask, used_points, sensor_pos, height=labeled_mask.shape[0])
    write_labeled_ply(f"./outputs/{base_name}_labeled_combined.ply", used_points, colors, instance_ids, instance_to_class)

if __name__ == "__main__":
    main()
