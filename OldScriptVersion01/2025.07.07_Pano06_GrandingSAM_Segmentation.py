import os
import sys
import time
import argparse
import numpy as np
import cv2
import pye57
import tifffile
import torch
from PIL import Image
from transformers import pipeline
from math import pi
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import same modules as in the demo:
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from GroundingDINO.groundingdino.datasets import transforms as T
from segment_anything.segment_anything import sam_model_registry, SamPredictor

import GroundingDINO.groundingdino.util.slconfig as slconfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.models import build_model

# ========== Utilities ==========
def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def setup_device():
    device = 0 if torch.cuda.is_available() else -1
    print(f"[{timestamp()}] Using {'GPU' if device == 0 else 'CPU'}")
    return device

def load_sam_model(device):
    print(f"[{timestamp()}] Loading original SAM model...")
    sam_checkpoint = "./grounded_sam_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    return sam

# ========== Step 1: Panorama Generation ==========
def load_point_cloud(filename):
    e57 = pye57.E57(filename)
    data = e57.read_scan(0, intensity=True, colors=True, ignore_missing_fields=True)
    sensor_position = np.array(e57.get_header(0).translation)
    return data, sensor_position

def flip_x_axis(pano_image):
    return np.fliplr(pano_image)

def correct_azimuth(pano_image, angle_deg):
    h, w, _ = pano_image.shape
    shift_px = int((angle_deg / 360.0) * w)
    pano_image = np.roll(pano_image, shift_px, axis=1)
    return pano_image


def spherical_projection(points, distances, height, width):
    x, y, z = points
    theta = np.arctan2(y, x)
    phi = np.arccos(z / distances)
    pixel_x = ((theta + pi) / (2 * pi)) * width
    pixel_y = (phi / pi) * height
    return pixel_x.astype(np.int32), pixel_y.astype(np.int32)

def create_panorama_image(data, sensor_pos, height):
    xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
    rel_xyz = xyz - sensor_pos[:, None]
    distances = np.linalg.norm(rel_xyz, axis=0)
    valid = distances > 0
    rel_xyz = rel_xyz[:, valid]
    distances = distances[valid]
    colors = np.stack([
        data["colorRed"][valid],
        data["colorGreen"][valid],
        data["colorBlue"][valid]
    ], axis=-1).astype(np.uint8)
    width = height * 2
    px, py = spherical_projection(rel_xyz, distances, height, width)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    def process_range(start, end):
        for i in range(start, end):
            x, y = px[i], py[i]
            if 0 <= x < width and 0 <= y < height:
                if distances[i] < depth[y, x]:
                    depth[y, x] = distances[i]
                    image[y, x] = colors[i]

    chunk_size = len(px) // os.cpu_count()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_range, i, min(i + chunk_size, len(px))) for i in range(0, len(px), chunk_size)]
        for future in as_completed(futures):
            pass

    return image, width, height

# ========== Step 2: Segmentation + Classification ==========
def segment_and_classify_with_grounded_sam(
    image_path,
    config_file,
    grounded_checkpoint,
    sam_checkpoint,
    text_prompt,
    device="cuda",
    box_threshold=0.3,
    text_threshold=0.25,
    sam_version="vit_h",
    bert_base_uncased_path=None,
    use_sam_hq=False
):
    """
    Fully compatible version matching the official Grounded-SAM demo.

    Args:
        image_path (str): Path to input panorama.
        config_file (str): Path to Grounding DINO config.
        grounded_checkpoint (str): Path to Grounding DINO .pth checkpoint.
        sam_checkpoint (str): Path to SAM .pth checkpoint.
        text_prompt (str): Text prompt, e.g., "Window . Door . Ceiling . Floor . Column"
        device (str): "cuda" or "cpu"
        box_threshold (float): Box confidence threshold.
        text_threshold (float): Text confidence threshold.
        sam_version (str): "vit_h", "vit_b", or "vit_l"
        bert_base_uncased_path (str): Optional BERT path for config.
        use_sam_hq (bool): Use SAM-HQ if desired.

    Returns:
        labeled_mask (np.ndarray): Instance id mask.
        instance_to_class (dict): Mapping id → phrase.
        mask_path (str): Saved .tif path.
    """

    # === 1) Load image ===
    print(f"[INFO] Loading image: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor, _ = transform(image_pil, None)  # (3, H, W)

    # === 2) Load Grounding DINO model ===
    args = slconfig.SLConfig.fromfile(config_file)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path

    model = build_model(args)
    checkpoint = torch.load(grounded_checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval().to(device)

    # === 3) Get boxes & phrases ===
    with torch.no_grad():
        outputs = model(image_tensor[None].to(device), captions=[text_prompt])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(text_prompt)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)

    print(f"[INFO] Detected {len(boxes_filt)} boxes")

    # === 4) Initialize SAM predictor ===
    if use_sam_hq:
        raise NotImplementedError("Add your SAM HQ logic if needed.")
    else:
        sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
        predictor = SamPredictor(sam.to(device))

    cv2_image = cv2.imread(image_path)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    predictor.set_image(cv2_image)

    # === 5) Transform boxes ===
    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, cv2_image.shape[:2]).to(device)

    # === 6) Get masks ===
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    # === 7) Build labeled mask ===
    labeled_mask = np.zeros((H, W), dtype=np.uint16)
    instance_to_class = {}

    for idx, mask_tensor in enumerate(masks):
        mask_np = mask_tensor.cpu().numpy()[0] > 0
        if np.sum(mask_np) == 0:
            continue
        labeled_mask[mask_np] = idx + 1
        instance_to_class[idx + 1] = pred_phrases[idx]
        print(f"[INFO] Mask {idx+1}: \"{pred_phrases[idx]}\" — area: {np.sum(mask_np)}")

    mask_path = image_path.replace(".jpg", "_mask.tif")
    tifffile.imwrite(mask_path, labeled_mask)
    print(f"[✓] Saved mask: {mask_path}")

    return labeled_mask, instance_to_class, mask_path

# ========== Step 3: Label Projection ==========
def spherical_projection_mask(points, colors, sensor, mask, width, height):
    rel = points - sensor
    r = np.linalg.norm(rel, axis=1)
    valid = r > 0
    rel = rel[valid]
    r = r[valid]
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    theta = (np.arctan2(y, x) + pi) / (2 * pi)
    phi = np.arccos(z / r) / pi
    px = np.clip((theta * width).astype(np.int32), 0, width - 1)
    py = np.clip((phi * height).astype(np.int32), 0, height - 1)
    instance_ids = mask[py, px]
    return points[valid], colors[valid], instance_ids

def save_combined_and_separate_ply(points, colors, instance_ids, instance_to_class, semantic_to_idx, output_base):
    semantic_labels = np.array([semantic_to_idx[instance_to_class.get(i, 'Unknown')] if i in instance_to_class else 0 for i in instance_ids], dtype=np.uint8)

    # Combined file
    combined_file = f"{output_base}_labeled_combined.ply"
    write_ply(points, colors, instance_ids, semantic_labels, combined_file)
    print(f"[✓] Saved combined labeled PLY: {combined_file}")

    # # Separate files per semantic class
    # for semantic_class, idx in semantic_to_idx.items():
    #     if idx == 0:
    #         continue  # skip unknown
    #     mask = semantic_labels == idx
    #     if np.sum(mask) == 0:
    #         continue
    #     part_file = f"{output_base}_{semantic_class.replace(' ', '_')}.ply"
    #     write_ply(points[mask], colors[mask], instance_ids[mask], semantic_labels[mask], part_file)
    #     print(f"[✓] Saved: {part_file}")

def save_semantic_panorama(labeled_mask, instance_to_class, semantic_colors, output_base):
    h, w = labeled_mask.shape
    semantic_img = np.zeros((h, w, 3), dtype=np.uint8)

    for instance_id in np.unique(labeled_mask):
        if instance_id == 0:
            continue  # background
        semantic_class = instance_to_class.get(instance_id, 'Unknown')
        color = semantic_colors.get(semantic_class, (0, 0, 0))
        semantic_img[labeled_mask == instance_id] = color

    out_path = f"{output_base}_semantic_pano.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(semantic_img, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved semantic panorama: {out_path}")

def save_per_class_panorama(labeled_mask, instance_to_class, canonical_classes, output_base):
    h, w = labeled_mask.shape

    for semantic_class in canonical_classes:
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        for instance_id, sem_class in instance_to_class.items():
            cleaned = sem_class.strip().lower().rstrip(".")
            if cleaned == semantic_class:
                binary_mask[labeled_mask == instance_id] = 255

        out_path = f"{output_base}_{semantic_class.replace(' ', '_')}_pano.jpg"
        cv2.imwrite(out_path, binary_mask)
        print(f"[✓] Saved class mask: {out_path}")



def write_ply(points, colors, instance_ids, semantic_labels, path):
    with open(path, "w") as f:
        f.write(f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property ushort instance_id
property uchar semantic_class
end_header
""")
        data = np.column_stack((points, colors, instance_ids, semantic_labels))
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d %d %d")

# ========== Main ==========
def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="E57 Pipeline: pano -> segment -> classify -> label point cloud")
    parser.add_argument("-input_e57", help="Input .e57 file", default="./assets/2ndLab-1.e57")
    parser.add_argument("--image_height", type=int, default=2048, help="Panorama image height")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.input_e57))[0]
    pano_path = f"{base_name}_pano.jpg"

    print(f"[{timestamp()}] Step 1: Generating panorama")
    data, sensor_pos = load_point_cloud(args.input_e57)
    pano_img, width, height = create_panorama_image(data, sensor_pos, args.image_height)
    cv2.imwrite(pano_path, pano_img)
    print(f"[✓] Saved panorama: {pano_path}")

    # # 1. Load both images
    # ref_pano = pano_img
    # captured_pano = cv2.imread(f"{base_name}_Capture.jpg")
    # captured_pano = cv2.cvtColor(captured_pano, cv2.COLOR_BGR2RGB)
    # # 2. Flip captured pano X-axis
    # captured_pano = flip_x_axis(captured_pano)
    # # 3. Resize captured pano to match reference dimensions
    # captured_pano = cv2.resize(captured_pano, (ref_pano.shape[1], ref_pano.shape[0]))
    # # 4. Compute azimuth offset
    # azimuth_offset = estimate_azimuth_offset(ref_pano, captured_pano)
    # # 5. Apply azimuth correction
    # captured_pano_aligned = correct_azimuth(captured_pano, azimuth_offset)
    # # 6. Save or use it
    # cv2.imwrite(f"{base_name}_Capture_aligned.jpg", cv2.cvtColor(captured_pano_aligned, cv2.COLOR_RGB2BGR))
    # print(f"[✓] Saved panorama: {base_name}_Capture_aligned.jpg")
    # pano_path = f"{base_name}_Capture_aligned.jpg"


    print(f"[{timestamp()}] Step 2: Segmenting + Classifying image")

    # Load models once
    labeled_mask, instance_to_class, mask_path = segment_and_classify_with_grounded_sam(
        pano_path,
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounded_checkpoint="grounded_sam_checkpoints/groundingdino_swint_ogc.pth",
        sam_checkpoint="grounded_sam_checkpoints/sam_vit_h_4b8939.pth",
        text_prompt="Window . Door . Ceiling . Floor . Column",
        device="cuda",
        box_threshold=0.3,
        text_threshold=0.25
    )
    text_prompt="Window . Door . Ceiling . Floor . Column"
    print(f"[✓] Segmentation saved: {mask_path}")

    print(f"[{timestamp()}] Step 3: Projecting labels to 3D")
    points = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], axis=-1)
    colors = np.stack([data["colorRed"], data["colorGreen"], data["colorBlue"]], axis=-1)
    valid = ~(np.isnan(points).any(axis=1))
    points = points[valid]
    colors = colors[valid]
    instance_ids = spherical_projection_mask(points, colors, sensor_pos, labeled_mask, width, height)[2]

    # Map semantic class to index for consistency
    canonical_classes = ["window", "door", "ceiling", "floor", "column"]
    semantic_to_idx = {label: i+1 for i, label in enumerate(canonical_classes)}
    semantic_to_idx['unknown'] = 0


    save_combined_and_separate_ply(points, colors, instance_ids, instance_to_class, semantic_to_idx, base_name)

    semantic_colors = {
    # "Building": (255, 0, 0),      # Red
    # "Road": (0, 255, 0),          # Green
    # "Vegetation": (0, 128, 0),    # Dark green
    # "Vehicle": (0, 0, 255),       # Blue
    # "Light pole": (255, 255, 0),  # Yellow
    # "Pedestrian": (255, 0, 255),  # Magenta
    "window": (0, 185, 197),
    "door": (36, 157, 230),
    "ceiling": (164, 199, 61),
    "floor": (247, 219, 42),
    "column": (117, 204, 96),
    "unknown": (125, 125, 125)
    }

    save_semantic_panorama(labeled_mask, instance_to_class, semantic_colors, base_name)
    save_per_class_panorama(labeled_mask, instance_to_class, canonical_classes, base_name)

    end_time = time.time()
    print(f"[{timestamp()}] Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
