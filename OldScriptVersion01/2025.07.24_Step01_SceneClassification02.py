# Step01_scene_classification.py (minimal output version)

import os
import torch
import open_clip
import numpy as np
import cv2
import pye57
from math import pi
from PIL import Image
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def timestamp():
    import time
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

    return image

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
    return label

def main():
    parser = argparse.ArgumentParser(description="Step 01: Scene Classification from .e57")
    parser.add_argument("-input_e57", required=True, help="Path to input .e57 file")
    parser.add_argument("--image_height", type=int, default=2048, help="Height of panorama image")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.input_e57))[0]
    pano_img_path = f"./outputs/{base_name}_pano.jpg"

    print(f"[{timestamp()}] Loading .e57 file and generating panorama...")
    data, sensor_pos = load_point_cloud(args.input_e57)
    pano_image = create_panorama_image(data, sensor_pos, args.image_height)
    cv2.imwrite(pano_img_path, pano_image)
    print(f"[✓] Saved panorama to: {pano_img_path}")

    print(f"[{timestamp()}] Classifying panorama scene...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32', device=device, force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    scene_label = classify_scene(pano_image, model, preprocess, tokenizer, device)

    # Instead of saving to txt, just print and return label
    print(f"[✓] Final scene type: {scene_label}")

if __name__ == "__main__":
    main()
