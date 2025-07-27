import os
import cv2
import csv
import torch
import open_clip
import tifffile
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
from plyfile import PlyData
from collections import defaultdict

def timestamp():
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')

MATERIAL_CLASSES = [
    "Aluminum", "Brick", "Carpet", "Cement board", "Concrete", "Concrete block",
    "Felt", "GFRC (Glass fiber reinforced concrete)", "Glass", "Gypsum board", "Steel",
    "Stone panel", "Stucco", "Synthetic rubber", "Terrazzo", "Vinyl composition tile (VCT)", "Wood"
]
MATERIAL_PROMPTS = [f"a surface made of {label}" for label in MATERIAL_CLASSES]

def extract_patch(image_path, mask_path, instance_id):
    image = cv2.imread(image_path)
    mask = tifffile.imread(mask_path)
    region = (mask == instance_id).astype(np.uint8)
    if np.sum(region) < 100:
        return None
    x, y, w, h = cv2.boundingRect(region)
    crop = image[y:y+h, x:x+w]
    return crop

def classify_material(image_crop, model, preprocess, tokenizer, device):
    image_input = preprocess(Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    text_inputs = tokenizer(MATERIAL_PROMPTS).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax().item()
        return MATERIAL_CLASSES[best_idx], similarity[0, best_idx].item()

def load_labeled_ply(ply_path):
    plydata = PlyData.read(ply_path)
    data = plydata['vertex'].data
    points = np.stack([data['x'], data['y'], data['z']], axis=-1)
    colors = np.stack([data['red'], data['green'], data['blue']], axis=-1) / 255.0
    semantics = np.array(data['semantic_class']).astype(np.uint8)
    instances = np.array(data['instance_id']).astype(np.uint16)
    return points, colors, semantics, instances

def group_points_by_instance(points, semantics, instances):
    grouped = defaultdict(list)
    for i, iid in enumerate(instances):
        if iid == 0:
            continue
        grouped[iid].append(i)
    return grouped

def estimate_oriented_bbox(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    plane_pcd = pcd.select_by_index(inliers)
    obb = plane_pcd.get_oriented_bounding_box()
    corners = np.asarray(obb.get_box_points())
    center = obb.center
    up = obb.R[:, 1]
    right = obb.R[:, 0]
    projections = [(corner - center) for corner in corners]
    projections_2d = [(np.dot(p, right), np.dot(p, up)) for p in projections]
    min_x = min(p[0] for p in projections_2d)
    max_x = max(p[0] for p in projections_2d)
    min_y = min(p[1] for p in projections_2d)
    max_y = max(p[1] for p in projections_2d)
    corner_3d = [
        center + min_x * right + min_y * up,
        center + max_x * right + min_y * up,
        center + max_x * right + max_y * up,
        center + min_x * right + max_y * up,
    ]
    return np.array(corner_3d), obb

def main():
    parser = argparse.ArgumentParser(description="Step03+04: Material Classification and Geometry Extraction")
    parser.add_argument("-image", required=False, default="./outputs/2ndLab-7_pano.jpg", help="Panorama image path")
    parser.add_argument("-mask", required=False, default="./outputs/2ndLab-7_pano_mask.tif", help="Instance mask path (.tif)")
    parser.add_argument("-ply", required=False, default="./outputs/2ndLab-7_pano_labeled_combined.ply", help="Labeled point cloud (.ply)")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    metadata_path = f"./outputs/{base_name}_final_metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"[{timestamp()}] [ERROR] Metadata file not found: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32', device=device, force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    points, colors, semantics, instances = load_labeled_ply(args.ply)
    grouped = group_points_by_instance(points, semantics, instances)

    results = []
    updated_rows = []
    for i, row in df.iterrows():
        iid = int(row['Instance ID'])
        crop = extract_patch(args.image, args.mask, iid)
        if crop is None:
            print(f"[{timestamp()}] [WARN] Skipping instance {iid} due to small region.")
            continue

        material, confidence = classify_material(crop, model, preprocess, tokenizer, device)
        pts = points[grouped[iid]]
        try:
            corners, _ = estimate_oriented_bbox(pts)
            flat_corners = corners.flatten().tolist()
        except Exception as e:
            print(f"[{timestamp()}] [ERROR] Instance {iid} geometry: {e}")
            flat_corners = [None] * 12
        updated_rows.append([
            row['Scene Type'], iid, row['Label'], material, confidence
        ] + flat_corners)

    corner_labels = [f"P{i}_{axis}" for i in range(1, 5) for axis in ['x', 'y', 'z']]
    header = ["Scene Type", "Instance ID", "Label", "Material", "Confidence"] + corner_labels

    with open(metadata_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_rows)
    print(f"[{timestamp()}] [✓] Final metadata updated: {metadata_path}")

if __name__ == "__main__":
    main()
