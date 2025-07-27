# Step01_02_combined_segmentation.py (scene classification + projection + labeled ply)

import os
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from plyfile import PlyElement, PlyData

import open_clip
import pye57

from segment_anything.segment_anything import sam_model_registry, SamPredictor
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap, clean_state_dict
from GroundingDINO.groundingdino.datasets import transforms as T
import GroundingDINO.groundingdino.util.slconfig as slconfig


def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def spherical_projection(points, distances, height, width):
    x, y, z = points
    theta = np.arctan2(y, x)
    phi = np.arccos(z / distances)
    pixel_x = ((theta + np.pi) / (2 * np.pi)) * width
    pixel_y = (phi / np.pi) * height
    return pixel_x.astype(np.int32), pixel_y.astype(np.int32)

def load_e57_and_generate_panorama(e57_path, image_height):
    e57 = pye57.E57(e57_path)
    data = e57.read_scan(0, intensity=True, colors=True, ignore_missing_fields=True)
    sensor_pos = np.array(e57.get_header(0).translation)

    xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
    rel = xyz - sensor_pos[:, None]
    distances = np.linalg.norm(rel, axis=0)
    valid = distances > 0
    rel = rel[:, valid]
    distances = distances[valid]

    colors = np.stack([
        data["colorRed"][valid],
        data["colorGreen"][valid],
        data["colorBlue"][valid]
    ], axis=-1).astype(np.uint8)
    used_points = rel.T + sensor_pos

    height = image_height
    width = height * 2
    px, py = spherical_projection(rel, distances, height, width)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    px_mapping = np.full((height, width), -1, dtype=np.int32)

    for i in range(len(px)):
        x, y = px[i], py[i]
        if 0 <= x < width and 0 <= y < height:
            if distances[i] < depth[y, x]:
                depth[y, x] = distances[i]
                image[y, x] = colors[i]
                px_mapping[y, x] = i

    return image, used_points, colors, px_mapping, sensor_pos

def classify_scene(image, model, preprocess, tokenizer, device):
    image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    text_prompts = [
        "an indoor surface adjacent to another room or partition",
        "an outdoor surface exposed to outdoor weather conditions like rain or sunlight",
        "a structural element part of the building foundation such as a slab or retaining wall",
        "a surface exposed to soil or earth, like basement walls or underground slabs"
    ]
    class_labels = ["Indoor Surface", "Outdoor Surface", "Foundation", "Ground"]
    text_inputs = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax().item()
        label = class_labels[best_idx]
        confidence = similarity[0, best_idx].item()

    print(f"[{timestamp()}] [✓] Scene classified as: {label} ({confidence:.2%} confidence)")
    return label

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

def segment_image(image, model, predictor, device, text_prompt, box_thresh=0.3, text_thresh=0.25):
    image_pil = Image.fromarray(image)
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

    predictor.set_image(image)
    H, W = image.shape[:2]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

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

def project_labels(px_mapping, labeled_mask):
    instance_ids = np.zeros(px_mapping.max() + 1, dtype=np.uint16)
    for y in range(px_mapping.shape[0]):
        for x in range(px_mapping.shape[1]):
            idx = px_mapping[y, x]
            if idx >= 0:
                instance_ids[idx] = labeled_mask[y, x]
    return instance_ids

def write_labeled_ply(filename, points, instance_ids, instance_to_class, keep_unlabeled=True):
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

    color_map = {
        i: (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        for i in semantic_to_idx.values()
    }

    vertex_data = []
    for (x, y, z), iid, sid in zip(points, instance_ids, semantic_labels):
        if not keep_unlabeled and iid == 0:
            continue
        r, g, b = color_map.get(sid, (180, 180, 180))
        vertex_data.append((x, y, z, r, g, b, iid, sid))

    vertex_array = np.array(vertex_data, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('instance_id', 'u2'), ('semantic_class', 'u1')
    ])
    PlyData([PlyElement.describe(vertex_array, 'vertex')], text=False).write(filename)
    print(f"[{timestamp()}] [✓] Saved labeled PLY: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Step 01+02: Scene Classification + Projection")
    parser.add_argument("-e57", required=False, default="./assets/2ndLab-7.e57", help="Path to input .e57 file")
    parser.add_argument("--image_height", type=int, default=2048)
    parser.add_argument("--text_prompt", default="Ceiling . Wall . Door . Window . Floor")
    parser.add_argument("--keep_unlabeled", action="store_true", help="Include unlabeled points in output .ply")
    parser.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino_ckpt", default="grounded_sam_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_ckpt", default="grounded_sam_checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.e57))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image, used_points, _, px_mapping, sensor_pos = load_e57_and_generate_panorama(args.e57, args.image_height)
    cv2.imwrite(f"./outputs/{base_name}_pano.jpg", image)
    print(f"[{timestamp()}] [✓] Saved 2D Panoramic image: ./outputs/{base_name}_pano.jpg")

    # Scene classification
    model_clip, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32', device=device, force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    classify_scene(image, model_clip, preprocess, tokenizer, device)

    # Segmentation
    model_sam, predictor = load_grounded_sam_models(device, args.config, args.dino_ckpt, args.sam_ckpt)
    labeled_mask, instance_to_class = segment_image(image, model_sam, predictor, device, args.text_prompt)
    instance_ids = project_labels(px_mapping, labeled_mask)
        # Save mask image
    mask_path = f"./outputs/{base_name}_mask.tif"
    cv2.imwrite(mask_path, labeled_mask.astype(np.uint16))
    print(f"[{timestamp()}] [✓] Saved 2D mask: {mask_path}")

    # Save labeled .ply
    write_labeled_ply(f"./outputs/{base_name}_labeled_combined.ply", used_points, instance_ids, instance_to_class, keep_unlabeled=args.keep_unlabeled)

if __name__ == "__main__":
    main()
