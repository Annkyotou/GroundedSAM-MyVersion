# Step04_geometry_extraction.py (enhanced with binary bounding box visualization)

import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from collections import defaultdict
import pandas as pd
import csv


def timestamp():
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')

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
    if corners.shape[0] != 8:
        raise ValueError("Bounding box must have 8 corners")

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
    parser = argparse.ArgumentParser(description="Step 04: Geometry Extraction and 3D Bounding Corners")
    parser.add_argument("-ply", required=True, help="Path to labeled combined binary .ply file")
    parser.add_argument("-material_csv", required=True, help="Path to material CSV file")
    parser.add_argument("-scene_type_txt", required=True, help="Scene type file")
    parser.add_argument("--save_vis", action="store_true", help="Save bounding box visualization to .ply")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.ply))[0]
    output_csv = f"./outputs/{base_name}_geometry.csv"

    points, colors, semantics, instances = load_labeled_ply(args.ply)
    grouped = group_points_by_instance(points, semantics, instances)

    mat_df = pd.read_csv(args.material_csv)
    mat_lookup = dict(zip(mat_df['Instance ID'], zip(mat_df['Label'], mat_df['Material'], mat_df['Confidence'])))

    with open(args.scene_type_txt, 'r') as f:
        scene_type = f.read().strip().split(',')[0].strip()

    results = []
    os.makedirs("./outputs", exist_ok=True)

    for iid in sorted(grouped.keys()):
        try:
            indices = grouped[iid]
            pts = points[indices]
            clr = colors[indices]
            if iid not in mat_lookup:
                print(f"[{timestamp()}] [WARN] No material found for instance {iid}, skipping")
                continue
            label, material, confidence = mat_lookup[iid]
            corners, obb = estimate_oriented_bbox(pts)
            results.append([scene_type, iid, label, material, confidence] + corners.flatten().tolist())
            print(f"[{timestamp()}] [✓] Instance {iid}: {label} => 4 corners extracted")

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(clr)
            bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            bbox_lines.paint_uniform_color([0.5, 0.0, 0.0])
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=cloud.get_center())
            o3d.visualization.draw_geometries(
                    [cloud, bbox_lines, axis],
                    window_name=f"Instance {iid}: {label}",
                    zoom=0.7,
                    front=[0.0, 0.0, -1.0],
                    lookat=cloud.get_center(),
                    up=[0.0, -1.0, 0.0]
            )
            
            if args.save_vis:
                o3d.io.write_point_cloud(f"./outputs/{base_name}_instance_{iid}_points.ply", cloud, write_ascii=False)

        except Exception as e:
            print(f"[{timestamp()}] [ERROR] Instance {iid}: {e}")

    corner_labels = [f"P{i}_{axis}" for i in range(1, 5) for axis in ['x', 'y', 'z']]
    header = ["Scene Type", "Instance ID", "Label", "Material", "Confidence"] + corner_labels

    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"[{timestamp()}] [✓] Geometry extraction saved to {output_csv}")

if __name__ == "__main__":
    main()
