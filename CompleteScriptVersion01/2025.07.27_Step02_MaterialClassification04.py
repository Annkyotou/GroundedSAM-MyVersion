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

    plane_normal = np.array(plane_model[:3])
    centroid = np.mean(np.asarray(plane_pcd.points), axis=0)

    up = np.array([0, 0, 1]) if abs(plane_normal[2]) > 0.5 else np.array([0, 1, 0])
    x_axis = np.cross(up, plane_normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(plane_normal, x_axis)

    points_np = np.asarray(plane_pcd.points)
    coords_2d = np.stack([
        [np.dot(p - centroid, x_axis), np.dot(p - centroid, y_axis)]
        for p in points_np
    ], axis=0).astype(np.float32)

    rect = cv2.minAreaRect(coords_2d)
    box = cv2.boxPoints(rect)
    corner_3d = [centroid + pt[0] * x_axis + pt[1] * y_axis for pt in box]

    return np.array(corner_3d), [[0, 1], [1, 2], [2, 3], [3, 0]]

def main():
    parser = argparse.ArgumentParser(description="Step03+04: Material Classification and Geometry Extraction")
    parser.add_argument("-image", default="./outputs/2ndLab-7_pano.jpg")
    parser.add_argument("-mask", default="./outputs/2ndLab-7_pano_mask.tif")
    parser.add_argument("-ply", default="./outputs/2ndLab-7_pano_labeled_combined.ply")
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

    MERGE_GROUPS = {
        101: [3, 4],
        102: [5, 6]
    }
    REMOVE_INSTANCES = {2}

    for new_id, group in MERGE_GROUPS.items():
        merged = []
        for iid in group:
            if iid in grouped:
                merged.extend(grouped[iid])
        grouped[new_id] = merged
        for old in group:
            grouped.pop(old, None)

    used_iids = [iid for iid in grouped.keys() if iid not in REMOVE_INSTANCES]
    updated_rows = []
    for iid in used_iids:
        matching_rows = df[df['Instance ID'].isin(MERGE_GROUPS.get(iid, [iid])) if iid >= 100 else df['Instance ID'] == iid]
        raw_label = matching_rows['Label'].values[0] if not matching_rows.empty else "merged"
        label = f"{raw_label}_merged" if iid >= 100 else raw_label

        crop = extract_patch(args.image, args.mask, iid) if iid < 100 else None
        if crop is None and iid < 100:
            print(f"[{timestamp()}] [WARN] Skipping instance {iid} due to small region.")
            continue

        if crop is None:
            print(f"[{timestamp()}] [WARN] Unknown merged instance {iid}.")
            material, confidence = "Unknown", 0.0
        else:
            material, confidence = classify_material(crop, model, preprocess, tokenizer, device)

        pts = points[grouped[iid]]
        clr = colors[grouped[iid]]
        try:
            corners, edges = estimate_oriented_bbox(pts)
            print(f"[{timestamp()}] [✓] Instance {iid}: {material} ({confidence:.2%})")
            flat_corners = corners.flatten().tolist()

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(clr)

            corner_pcd = o3d.geometry.PointCloud()
            corner_pcd.points = o3d.utility.Vector3dVector(corners)
            corner_pcd.paint_uniform_color([1, 0, 0])

            # Increase corner size by duplicating points
            corner_spheres = []
            for point in corners:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.07)
                sphere.translate(point)
                sphere.paint_uniform_color([1, 0, 0])
                corner_spheres.append(sphere)

            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(corners)
            lines.lines = o3d.utility.Vector2iVector(edges)
            lines.paint_uniform_color([0.5, 0, 0])

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=cloud.get_center())
            o3d.visualization.draw_geometries(
                [cloud, *corner_spheres, lines, axis],
                window_name=f"Instance {iid}: {label}",
                zoom=0.7,
                front=[0.0, 0.0, -1.0],
                lookat=cloud.get_center(),
                up=[0.0, -1.0, 0.0]
            )
        except Exception as e:
            print(f"[{timestamp()}] [ERROR] Instance {iid} geometry: {e}")
            flat_corners = [None] * 12

        row_scene_type = matching_rows['Scene Type'].values[0] if not matching_rows.empty else "Unknown"
        row_confidence = matching_rows['Confidence'].values[0] if not matching_rows.empty else 0.0
        row_area = matching_rows['Area'].sum() if iid >= 100 else matching_rows['Area'].values[0] if not matching_rows.empty else 0

        updated_rows.append([
            row_scene_type,
            row_confidence,
            iid,
            label,
            row_area,
            material,
            confidence
        ] + flat_corners)

    corner_labels = [f"P{i}_{axis}" for i in range(1, 5) for axis in ['x', 'y', 'z']]
    header = ["Scene Type", "Confidence", "Instance ID", "Label", "Area", "Material", "Confidence"] + corner_labels

    with open(metadata_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_rows)
    print(f"[{timestamp()}] [✓] Final metadata updated: {metadata_path}")

if __name__ == "__main__":
    main()
