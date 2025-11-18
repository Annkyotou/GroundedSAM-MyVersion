import os
import cv2
import time
import torch
import numpy as np
import argparse
from PIL import Image
from math import pi
from plyfile import PlyElement, PlyData
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import csv

import pye57
import tifffile
import open_clip

from segment_anything.segment_anything import sam_model_registry, SamPredictor
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap, clean_state_dict
from GroundingDINO.groundingdino.datasets import transforms as T
from GroundingDINO.groundingdino.util.slconfig import SLConfig

def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def spherical_projection(points, distances, height, width):
    x, y, z = points
    theta = np.arctan2(y, x)
    phi = np.arccos(z / distances)
    pixel_x = ((theta + pi) / (2 * pi)) * width
    pixel_y = (phi / pi) * height
    return pixel_x.astype(np.int32), pixel_y.astype(np.int32)

def load_point_cloud(filename):
    e57 = pye57.E57(filename)
    data = e57.read_scan(0, intensity=True, colors=True, ignore_missing_fields=True)
    sensor_position = np.array(e57.get_header(0).translation)
    return data, sensor_position

def create_panorama_image(data, sensor_pos, height, base_name):
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

    used_points = rel_xyz.T + sensor_pos
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

    # Save depth map (replace inf with 0)
    # depth[np.isinf(depth)] = 0.0
    # tifffile.imwrite(f"./outputs/{base_name}_pano_depth.tif", depth.astype(np.float32))
    # print(f"[INFO] Saved depth map to: ./outputs/{base_name}_pano_depth.tif")

    # Save visual preview with colorbar
    import matplotlib.pyplot as plt
    vis_path = f"./outputs/{base_name}_pano_depth_vis.png"
    plt.figure(figsize=(12, 6))
    im = plt.imshow(depth, cmap='jet')
    plt.colorbar(im, label='Distance (m)')
    plt.title("Distance Map")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved depth preview to: {vis_path}")

    return image, used_points, colors, depth

# === Two-pass instance segmentation ===
def segment_image_two_pass(image_path, model, predictor, device, text_prompt_main, text_prompt_fallback, base_name):
    # First pass
    labeled_mask, instance_to_class, instance_areas, instance_confidences = segment_image(
        image_path, model, predictor, device, text_prompt_main, base_name + "_pass1"
    )
    H, W = labeled_mask.shape

    # Identify unclassified area
    unclassified_mask = (labeled_mask == 0).astype(np.uint8) * 255
    if np.count_nonzero(unclassified_mask) < 100:
        print("[INFO] No significant unclassified region for second pass.")
        return labeled_mask, instance_to_class, instance_areas, instance_confidences

    # Save intermediate unclassified mask
    unclassified_path = f"./outputs/{base_name}_unclassified.jpg"
    base_image = cv2.imread(image_path)
    masked_image = cv2.bitwise_and(base_image, base_image, mask=unclassified_mask)
    cv2.imwrite(unclassified_path, masked_image)
    print(f"[INFO] Saved image for second pass: {unclassified_path}")

    # Merge masks
    offset = labeled_mask.max()
    merged_mask = labeled_mask.copy()

    # # Second pass with new prompt
    # second_mask, second_classes, second_areas, second_confidences = segment_image(
    #     unclassified_path, model, predictor, device, text_prompt_fallback, base_name + "_pass2"
    # )

    # for i in range(H):
    #     for j in range(W):
    #         sid = second_mask[i, j]
    #         if sid > 0:
    #             new_id = sid + offset
    #             merged_mask[i, j] = new_id
    #             instance_to_class[new_id] = second_classes[sid]
    #             instance_areas[new_id] = second_areas[sid]
    #             instance_confidences[new_id] = second_confidences[sid]

    # print(f"[INFO] Completed second pass with prompt: '{text_prompt_fallback}'")

    # Save final merged mask
    final_mask_path = os.path.join("./outputs", f"{base_name}_pano_mask.tif")
    tifffile.imwrite(final_mask_path, merged_mask.astype(np.uint16))
    print(f"[INFO] Saved merged mask to: {final_mask_path}")

    return merged_mask, instance_to_class, instance_areas, instance_confidences


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

    print(f"[✓] Scene classified as: {label} ({confidence:.2%} confidence)")
    return label, confidence

def load_grounded_sam_models(device, config_file, dino_checkpoint, sam_checkpoint, sam_version="vit_h"):
    args = SLConfig.fromfile(config_file)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(dino_checkpoint, map_location=device)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval().to(device)

    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return model, predictor

def segment_image(image_path, model, predictor, device, text_prompt, base_name):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor, _ = transform(image_pil, None)
    text_prompt = text_prompt.lower()
    text_prompt = text_prompt.strip()
    if not text_prompt.endswith("."):
        text_prompt += "."

    print(f"[DEBUG] Final processed text_prompt: \"{text_prompt}\"")
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor[None].to(device), captions=[text_prompt])
    logits = outputs["pred_logits"].sigmoid()[0].cpu()
    boxes = outputs["pred_boxes"][0].cpu()

    mask = logits.max(dim=1)[0] > 0.30
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(text_prompt)
    phrases = [get_phrases_from_posmap(logit > 0.25, tokenized, tokenlizer) for logit in logits_filt]

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
        boxes=transformed_boxes.to(device),
        multimask_output=False
    )

    labeled_mask = np.zeros((H, W), dtype=np.uint16)
    instance_to_class = {}
    instance_areas = {}
    instance_confidences = {}

    scores = logits.max(dim=1)[0][mask]
    for idx, mask_tensor in enumerate(masks):
        mask_np = mask_tensor.cpu().numpy()[0] > 0
        if np.sum(mask_np) == 0:
            continue
        area = int(np.sum(mask_np))
        instance_id = idx + 1
        labeled_mask[mask_np] = instance_id
        instance_to_class[instance_id] = phrases[idx]
        instance_areas[instance_id] = area
        instance_confidences[instance_id] = float(scores[idx])  # confidence in [0, 1]
        print(f"[{timestamp()}] [INFO] Mask {instance_id}: \"{phrases[idx]}\" | Area: {area} pixels | Confidence: {scores[idx]:.2f}")

    # tifffile.imwrite(f"./outputs/{base_name}_pano_mask.tif", labeled_mask)

    # Save visualization
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    save_mask_visualization(
        os.path.join("./outputs", f"{base_name}_pano_mask.jpg"),
        masks,
        boxes_filt,
        phrases,
        image_rgb,
        confidences=instance_confidences
    )
    print(f"[{timestamp()}] [✓] Saved masked image to: ./outputs/{base_name}_pano_mask.jpg")

    return labeled_mask, instance_to_class, instance_areas, instance_confidences

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

def write_partial_metadata(output_path, scene_type, confidence, instance_to_class, instance_confidences, instance_areas):
    header = ["Scene Type", "Confidence", "Instance ID", "Label", "Instance Confidence", "Area"]
    rows = []
    for iid, label in instance_to_class.items():
        label_text = list(label)[0] if isinstance(label, set) else label
        rows.append([scene_type, confidence, iid, label_text, instance_confidences.get(iid, 0.0), instance_areas.get(iid, 0)])
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sorted(rows, key=lambda x: x[2]))
    print(f"[{timestamp()}] [✓] Created metadata file: {output_path}")

def interactive_click_prompt(image_path, display_size=(1280, 720)):
    image = cv2.imread(image_path)
    original_shape = image.shape[:2]  # (H, W)
    scale_x = display_size[0] / original_shape[1]
    scale_y = display_size[1] / original_shape[0]
    scale = min(scale_x, scale_y)

    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    clicks = []

    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(((x / scale, y / scale), 1))  # Rescale back to original image
            cv2.circle(resized, (x, y), 5, (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicks.append(((x / scale, y / scale), 0))
            cv2.circle(resized, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click Prompt", resized)

    print("[INFO] Left click = foreground, Right click = background, ESC = finish")
    cv2.imshow("Click Prompt", resized)
    cv2.setMouseCallback("Click Prompt", click_callback)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    coords, lbls = zip(*clicks) if clicks else ([], [])
    return list(coords), list(lbls)

def segment_image_manual(image_path, predictor, device, base_name):
    coords, lbls = interactive_click_prompt(image_path)
    if not coords:
        print("[WARN] No points clicked. Skipping segmentation.")
        return np.zeros(cv2.imread(image_path).shape[:2], dtype=np.uint8), {}, {}

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)  # ✅ ADD THIS LINE

    point_coords = torch.tensor([coords], dtype=torch.float).to(device)
    point_labels = torch.tensor([lbls], dtype=torch.int).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    mask_np = masks[0, 0].cpu().numpy() > 0
    labeled_mask = np.zeros(mask_np.shape, dtype=np.uint16)
    labeled_mask[mask_np] = 1
    tifffile.imwrite(f"./outputs/{base_name}_pano_mask.tif", labeled_mask)

    return labeled_mask, {1: 'manual'}, {1: int(np.sum(mask_np))}


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # RGBA
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label, fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

def save_mask_visualization(output_path, mask_list, boxes, labels, image_rgb, confidences=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb)

    for idx, mask in enumerate(mask_list):
        show_mask(mask.cpu().numpy(), ax, random_color=True)
        label_text = labels[idx]
        instance_id = idx + 1
        conf_text = f"{instance_id} ({round(100.0, 2)}%)"  # ← update with real confidence
        if isinstance(label_text, set):
            label_text = list(label_text)[0]
        full_label = f"{label_text}\nID:{instance_id} Conf:{confidences.get(instance_id, 0.0):.2f}"
        show_box(boxes[idx].numpy(), ax, full_label)

    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Step01+02: Scene Classification + Instance Segmentation")
    parser.add_argument("-input_e57", required=False, default="./assets/JPWanger01/JPWangerData01_Scan05.e57", help="Path to input .e57 file")
    parser.add_argument("--image_height", type=int, default=2048, help="Height of panorama image")
    parser.add_argument("--text_prompt", default="Ground . Vegetation . Vehicle . Building . Light Pole . Road .")
    parser.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino_ckpt", default="grounded_sam_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_ckpt", default="grounded_sam_checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.input_e57))[0]
    os.makedirs("./outputs", exist_ok=True)
    pano_img_path = f"./outputs/{base_name}_pano.jpg"
    ply_output_path = f"./outputs/{base_name}_pano_labeled_combined.ply"
    metadata_path = f"./outputs/{base_name}_pano_final_metadata.csv"

    print(f"[{timestamp()}] Loading .e57 file and generating panorama...")
    data, sensor_pos = load_point_cloud(args.input_e57)
    pano_image, used_points, used_colors, _ = create_panorama_image(data, sensor_pos, args.image_height, base_name)
    cv2.imwrite(pano_img_path, pano_image)
    print(f"[{timestamp()}] [✓] Saved panorama to: {pano_img_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, preprocess_clip, tokenizer_clip = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32', device=device, force_quick_gelu=True)
    tokenizer_clip = open_clip.get_tokenizer('ViT-B-32')

    print(f"[{timestamp()}] Classifying scene...")
    scene_label, scene_conf = classify_scene(pano_image, model_clip, preprocess_clip, tokenizer_clip, device)

    print(f"[{timestamp()}] Performing two-pass instance segmentation...")
    model_gs, predictor = load_grounded_sam_models(device, args.config, args.dino_ckpt, args.sam_ckpt)
    mask, instance_to_class, instance_areas, instance_confidences = segment_image_two_pass(
        pano_img_path,
        model_gs, predictor, device,
        args.text_prompt,
        "wall .",
        base_name
    )

    instance_ids = label_projection(mask, used_points, sensor_pos, args.image_height)
    write_labeled_ply(ply_output_path, used_points, used_colors, instance_ids, instance_to_class)

    write_partial_metadata(metadata_path, scene_label, round(scene_conf, 4), instance_to_class, instance_confidences, instance_areas)

    print(f"[{timestamp()}] Step 01+02 complete.")
    print(f"Scene Type: {scene_label} ({scene_conf:.2%})")
    print(f"Saved outputs: {pano_img_path}, {ply_output_path}, mask.tif, metadata.csv")


if __name__ == "__main__":
    main()
