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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==========================================================
# Utilities & data
# ==========================================================

def timestamp():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")

MATERIAL_CLASSES = [
    "Aluminum", "Brick", "Carpet", "Cement board", "Concrete",
    "Concrete block", "Felt", "GFRC", "Glass", "Gypsum board",
    "Steel", "Stone panel", "Stucco", "Synthetic rubber",
    "Terrazzo", "Vinyl composition tile (VCT)", "Wood"
]

LABEL_ALIASES = {
    "wall": {"wall", "walls"},
    "floor": {"floor", "floors"},
    "ceiling": {"ceiling", "ceilings"}
}


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

@dataclass
class Plane:
    n: np.ndarray  # unit normal (3,)
    d: float       # plane offset, s.t. n·x + d = 0

@dataclass
class OrientedBox:
    corners3d: np.ndarray       # (4,3) CCW with deterministic start
    edges: List[Tuple[int,int]] # [(0,1),(1,2),(2,3),(3,0)]
    plane: Plane                # fitted plane
    x_axis: np.ndarray          # in-plane unit axis
    y_axis: np.ndarray          # in-plane unit axis
    centroid: np.ndarray        # (3,)

@dataclass
class RoomFit:
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    floor_z: float
    ceil_z: float
    x_left: float
    x_right: float
    y_back: float
    y_front: float
    planes: Dict[str, Plane]             # keys: 'floor','ceiling','east','west','north','south'
    floor_corners: np.ndarray            # (4,3) P1..P4 CCW
    ceiling_corners: np.ndarray          # (4,3) P1..P4 CCW
    wall_order_ccw_from_east: List[str]  # ['east','north','west','south']

# ==========================================================
# CLIP material classifier
# ==========================================================

def load_clip_model(device="cpu"):
    print(f"[{timestamp()}] Loading CLIP zero-shot classifier…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    return model, preprocess, tokenizer


def classify_material_with_clip(image_bgr: np.ndarray,
                                mask: Optional[np.ndarray],
                                model=None, preprocess=None, tokenizer=None,
                                device="cpu") -> Tuple[str, float]:
    """Crop instance from pano via mask, run zero-shot CLIP with prompts
    like "a surface made of {material}" and return (label, confidence).
    If mask is None, we classify the whole image.
    """
    if model is None:
        return "Unknown", 0.0

    # Crop tight bbox from mask if present
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        crop = image_bgr[y0:y1, x0:x1]
    else:
        crop = image_bgr

    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_img).unsqueeze(0).to(device)

    # Build text prompts
    texts = [f"a surface made of {m}" for m in MATERIAL_CLASSES]
    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

        image_feats = model.encode_image(image_input)
        image_feats /= image_feats.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_feats @ text_feats.T).softmax(dim=-1)
        best = logits.argmax(dim=-1).item()
        return MATERIAL_CLASSES[best], float(logits[0, best].item())

# ==========================================================
# Geometry helpers
# ==========================================================

def canonicalize_quad_order(corners2d: np.ndarray) -> np.ndarray:
    if corners2d.shape != (4,2):
        raise ValueError("corners2d must be (4,2)")
    c = corners2d.mean(axis=0)
    ang = np.arctan2(corners2d[:,1] - c[1], corners2d[:,0] - c[0])
    ccw_idx = np.argsort(ang)
    ccw_pts = corners2d[ccw_idx]
    start = np.lexsort((ccw_pts[:,0], ccw_pts[:,1]))[0]  # lower-left
    order = np.roll(ccw_idx, -start)
    return order.astype(int)


def estimate_oriented_bbox(points: np.ndarray,
                           ransac_distance: float = 0.02,
                           ransac_n: int = 3,
                           ransac_iters: int = 1000) -> OrientedBox:
    if points.shape[0] < 50:
        raise ValueError("Not enough points for plane/OBB fit")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_distance,
                                ransac_n=ransac_n,
                                num_iterations=ransac_iters)
    a, b, c, d = plane_model
    n = _normalize(np.array([a,b,c], dtype=float))

    # PCA on plane-projected points
    pts = points - points.mean(axis=0)
    pts_proj = pts - (pts @ n)[:,None] * n[None,:]
    U, S, Vt = np.linalg.svd(pts_proj, full_matrices=False)
    x_axis = _normalize(Vt[0])
    y_axis = _normalize(np.cross(n, x_axis))

    centroid = (points.mean(axis=0) - (points.mean(axis=0) @ n) * n)

    coords2d = np.column_stack(((points - centroid) @ x_axis,
                                (points - centroid) @ y_axis)).astype(np.float32)
    rect = cv2.minAreaRect(coords2d)
    box = cv2.boxPoints(rect).astype(np.float32)  # CW, start not fixed
    order = canonicalize_quad_order(box)
    box = box[order]

    corners3d = np.stack([centroid + p[0]*x_axis + p[1]*y_axis for p in box], axis=0)
    plane = Plane(n=n, d=-float(n @ corners3d.mean(axis=0)))

    return OrientedBox(corners3d=corners3d,
                       edges=[(0,1),(1,2),(2,3),(3,0)],
                       plane=plane,
                       x_axis=x_axis,
                       y_axis=y_axis,
                       centroid=centroid)


def fit_room_box(planes: Dict[str, Dict]) -> RoomFit:
    floor = planes.get('floor', None)
    ceil  = planes.get('ceiling', None)
    walls = planes.get('walls', [])
    if floor is None or ceil is None or len(walls) < 1:
        raise ValueError("Need floor, ceiling, and at least 1 wall")

    n_f = _normalize(floor['plane'].n)
    n_c = _normalize(ceil['plane'].n)
    if np.dot(n_f, n_c) > 0:
        n_c = -n_c
    z = _normalize(n_f)

    # Collect horizontal wall normals & all structural points for extents
    horiz_normals = []
    wall_infos = []
    all_pts = []
    for w in walls:
        nw = _normalize(w['plane'].n)
        nh = _normalize(nw - np.dot(nw, z) * z)
        if np.linalg.norm(nh) >= 1e-6:
            horiz_normals.append(nh)
            wall_infos.append({'orig': w, 'nh': nh})
        if w.get('points') is not None:
            all_pts.append(w['points'])
    if floor.get('points') is not None:
        all_pts.append(floor['points'])
    if ceil.get('points') is not None:
        all_pts.append(ceil['points'])
    all_pts = np.concatenate(all_pts, axis=0) if len(all_pts) else None

    # If fewer than 2 distinct horizontal normals, derive u from covariance of wall points
    if len(horiz_normals) >= 2:
        H = np.stack(horiz_normals, axis=0)
        C = H.T @ H
        eigvals, eigvecs = np.linalg.eigh(C)
        u = _normalize(eigvecs[:, np.argmax(eigvals)])
        u = _normalize(u - np.dot(u, z) * z)
    else:
        # Fallback: use PCA on horizontal coordinates of all wall points
        if all_pts is None:
            raise ValueError("Insufficient data to establish room axes")
        Pw = all_pts - (all_pts @ z)[:,None]*z[None,:]
        _, _, Vt = np.linalg.svd(Pw, full_matrices=False)
        u = _normalize(Vt[0])
        u = _normalize(u - np.dot(u, z) * z)
    v = _normalize(np.cross(z, u))

    def _med_proj(pts: np.ndarray, axis: np.ndarray) -> float:
        return float(np.median(pts @ axis))

    x_left_vals, x_right_vals = [], []
    y_back_vals, y_front_vals = [], []

    for w in wall_infos:
        nh = w['nh']
        pts = w['orig']['points']
        du = abs(np.dot(nh, u))
        dv = abs(np.dot(nh, v))
        if du >= dv:
            sign = np.sign(np.dot(nh, u))
            med = _med_proj(pts, u) if pts is not None else 0.0
            (x_right_vals if sign > 0 else x_left_vals).append(med)
        else:
            sign = np.sign(np.dot(nh, v))
            med = _med_proj(pts, v) if pts is not None else 0.0
            (y_front_vals if sign > 0 else y_back_vals).append(med)

    # Robust extents from all structural points (fallback for missing families)
    if all_pts is None:
        raise ValueError("No structural points available for extents")
    proj_u = all_pts @ u
    proj_v = all_pts @ v
    x_min, x_max = np.percentile(proj_u, [2.5, 97.5])
    y_min, y_max = np.percentile(proj_v, [2.5, 97.5])

    x_left  = float(np.median(x_left_vals))  if len(x_left_vals)  else float(x_min)
    x_right = float(np.median(x_right_vals)) if len(x_right_vals) else float(x_max)
    y_back  = float(np.median(y_back_vals))  if len(y_back_vals)  else float(y_min)
    y_front = float(np.median(y_front_vals)) if len(y_front_vals) else float(y_max)

    # Heights from floor/ceiling points
    z_floor = float(np.median(floor['points'] @ z)) if floor.get('points') is not None else -float(floor['plane'].d)
    z_ceil  = float(np.median(ceil['points'] @ z))  if ceil.get('points')  is not None else  float(ceil['plane'].d)

    def W(x,y,zs):
        return x*u + y*v + zs*z

    floor_corners = np.stack([
        W(x_left,  y_back,  z_floor),
        W(x_right, y_back,  z_floor),
        W(x_right, y_front, z_floor),
        W(x_left,  y_front, z_floor),
    ], axis=0)

    ceiling_corners = np.stack([
        W(x_left,  y_back,  z_ceil),
        W(x_right, y_back,  z_ceil),
        W(x_right, y_front, z_ceil),
        W(x_left,  y_front, z_ceil),
    ], axis=0)

    planes_out = {
        'floor':   Plane(n=z,    d=-z_floor),
        'ceiling': Plane(n=-z,   d= z_ceil),
        'east':    Plane(n=+u,   d=-x_right),
        'west':    Plane(n=-u,   d= x_left),
        'north':   Plane(n=+v,   d=-y_front),
        'south':   Plane(n=-v,   d= y_back),
    }

    wall_order_ccw_from_east = ['east','north','west','south']

    return RoomFit(
        z=z, u=u, v=v,
        floor_z=z_floor, ceil_z=z_ceil,
        x_left=x_left, x_right=x_right,
        y_back=y_back, y_front=y_front,
        planes=planes_out,
        floor_corners=floor_corners,
        ceiling_corners=ceiling_corners,
        wall_order_ccw_from_east=wall_order_ccw_from_east
    )


def snapped_wall_corners(name: str, rf: RoomFit) -> np.ndarray:
    xL, xR, yB, yF, z0, z1 = rf.x_left, rf.x_right, rf.y_back, rf.y_front, rf.floor_z, rf.ceil_z

    if name == 'west':
        quads = [xL,yB,z0, xL,yF,z0, xL,yF,z1, xL,yB,z1]
    elif name == 'east':
        quads = [xR,yB,z0, xR,yF,z0, xR,yF,z1, xR,yB,z1]
    elif name == 'south':
        quads = [xL,yB,z0, xR,yB,z0, xR,yB,z1, xL,yB,z1]
    elif name == 'north':
        quads = [xL,yF,z0, xR,yF,z0, xR,yF,z1, xL,yF,z1]
    else:
        raise ValueError(f"Unknown wall name: {name}")

    q = np.array(quads, dtype=float).reshape(4,3)
    return q[:,0:1]*rf.u + q[:,1:2]*rf.v + q[:,2:3]*rf.z


def classify_wall_name(n: np.ndarray, rf: RoomFit) -> str:
    nh = _normalize(n - np.dot(n, rf.z)*rf.z)
    du = np.dot(nh, rf.u)
    dv = np.dot(nh, rf.v)
    if abs(du) >= abs(dv):
        return 'east' if du > 0 else 'west'
    return 'north' if dv > 0 else 'south'


def wall_index_from_name(name: str, rf: RoomFit) -> int:
    return {'east':1,'north':2,'west':3,'south':4}[name]

# ==========================================================
# I/O helpers
# ==========================================================

def load_labeled_ply(ply_path: str):
    ply = PlyData.read(ply_path)
    data = ply['vertex'].data

    def get_field(name, default_val=None):
        try:
            return np.asarray(data[name])
        except ValueError:
            if default_val is None:
                raise
            return np.full(len(data), default_val)

    points = np.stack([get_field('x'), get_field('y'), get_field('z')], axis=-1).astype(np.float64)
    colors = np.stack([get_field('red'), get_field('green'), get_field('blue')], axis=-1).astype(np.float32) / 255.0

    # Try various common field names
    for inst_key in ['instance_id', 'instances', 'label_instance', 'iid']:
        try:
            instances = np.asarray(data[inst_key]).astype(np.int32)
            break
        except ValueError:
            instances = None
    if instances is None:
        raise RuntimeError("Instance field not found in PLY (expected 'instance' or similar)")

    semantics = None
    for sem_key in ['semantic', 'semantics', 'label_semantic', 'sid']:
        try:
            semantics = np.asarray(data[sem_key]).astype(np.int32)
            break
        except ValueError:
            semantics = None

    return points, colors, semantics, instances


def load_masks(mask_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(mask_path):
        return None
    # Expect a single-channel mask where pixel value = instance id
    arr = tifffile.imread(mask_path)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr

def apply_merges_and_removals(by_iid: Dict[int, List[int]], df: pd.DataFrame,
                              merge_groups: Dict[int, List[int]],
                              remove_ids: set):
    """
    - Remove any instance IDs in remove_ids from by_iid and df.
    - Merge source -> dest according to merge_groups (dest: [src...]).
    - Collapse duplicates in df by (Instance ID, Label) summing Area.
    Returns: (by_iid2, df2)
    """
    # Remove
    for rid in list(remove_ids):
        by_iid.pop(int(rid), None)
    if len(df) and 'Instance ID' in df.columns:
        df = df[~df['Instance ID'].isin(list(remove_ids))].copy()

    # Remap for merges
    remap = {int(src): int(dest) for dest, srcs in merge_groups.items() for src in srcs}

    # Apply remap to by_iid
    for src, dest in list(remap.items()):
        if src in by_iid:
            by_iid.setdefault(dest, []).extend(by_iid[src])
            by_iid.pop(src, None)

    # Apply remap to df and collapse rows
    if len(df) and 'Instance ID' in df.columns:
        if remap:
            df = df.copy()
            df['Instance ID'] = df['Instance ID'].apply(lambda x: remap.get(int(x), int(x)))

        agg = {
            'Scene Type': 'first',
            'Confidence': 'first',
            'Label': 'first',
            'Instance Confidence': 'first',
            'Area': 'sum',
            'Material': 'first',
            'Material Confidence': 'first',
        }
        # keep snapped corners / wall info if present
        for c in [f'P{i}_{ax}' for i in range(1,5) for ax in ('x','y','z')] + ['Wall Index','Wall Name']:
            if c in df.columns:
                agg[c] = 'first'

        df = df.groupby(['Instance ID','Label'], as_index=False).agg(agg)

    return by_iid, df

# ==========================================================
# Main pipeline
# ==========================================================

def _flatten_corners(c: np.ndarray) -> List[float]:
    return [float(x) for p in c for x in p.tolist()]


def main():
    import matplotlib.pyplot as plt

    def save_topdown_debug(out_path_base: str, room: RoomFit, planes_input: Dict[str, Dict]):
        try:
            # Project a small sample of wall points for context
            samples_u, samples_v = [], []
            for w in planes_input.get('walls', []):
                pts = w.get('points')
                if pts is None or len(pts) == 0:
                    continue
                sel = pts[np.random.choice(len(pts), size=min(5000, len(pts)), replace=False)]
                samples_u.append(sel @ room.u)
                samples_v.append(sel @ room.v)
            if samples_u:
                samples_u = np.concatenate(samples_u); samples_v = np.concatenate(samples_v)
            # Room rectangle
            rect_u = [room.x_left, room.x_right, room.x_right, room.x_left, room.x_left]
            rect_v = [room.y_back, room.y_back, room.y_front, room.y_front, room.y_back]
            plt.figure(figsize=(6,6))
            if samples_u:
                plt.scatter(samples_u, samples_v, s=1, alpha=0.3)
            plt.plot(rect_u, rect_v, linewidth=2)
            for name, (xu, yv) in {
                'East': (room.x_right, 0.5*(room.y_back+room.y_front)),
                'West': (room.x_left,  0.5*(room.y_back+room.y_front)),
                'North':(0.5*(room.x_left+room.x_right), room.y_front),
                'South':(0.5*(room.x_left+room.x_right), room.y_back)
            }.items():
                plt.text(xu, yv, name)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('u (m)'); plt.ylabel('v (m)'); plt.title('Top-down QC (room frame)')
            png = f"{out_path_base}_topdown.png"
            plt.savefig(png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[DEBUG] Saved top-down QC: {png}")
        except Exception as e:
            print(f"[WARN] Top-down QC failed: {e}")
    parser = argparse.ArgumentParser("Step02 with Room Regularization")
    parser.add_argument("--pano", required=False, default="./outputs/2ndLab-7_pano.jpg", help="Path to pano image (RGB)")
    parser.add_argument("--mask", required=False, default="./outputs/2ndLab-7_pano_mask.tif", help="Path to mask TIFF where values are instance IDs")
    parser.add_argument("--input_metadata", required=False, default="./outputs/2ndLab-7_pano_final_metadata.csv", help="CSV from Step01 with columns incl. Scene Type, Confidence, Instance ID, Label, Area")
    parser.add_argument("--ply", required=False, default="./outputs/2ndLab-7_pano_labeled_combined.ply", help="Labeled point cloud with instance & semantic fields")
    parser.add_argument("--out_csv", required=False, default="./outputs/2ndLab-7_pano_final_metadata.csv", help="Output CSV (final geometry/material)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[{timestamp()}] Loading inputs…")
    df = pd.read_csv(args.input_metadata)
    # Apply manual merges/removals early so metadata and geometry align
    by_iid: Dict[int, List[int]] = defaultdict(list)
    # We'll fill by_iid after loading PLY, so temporarily store df; merges use df columns

    pano_bgr = cv2.imread(args.pano, cv2.IMREAD_COLOR)
    if pano_bgr is None:
        raise FileNotFoundError(f"Cannot read pano: {args.pano}")
    mask_img = load_masks(args.mask) if args.mask else None

    points, colors, semantics, instances = load_labeled_ply(args.ply)
    print(f"[{timestamp()}] PLY points: {points.shape[0]}")

    # Group indices by instance id
    by_iid: Dict[int, List[int]] = defaultdict(list)
    for i, iid in enumerate(instances):
        by_iid[int(iid)].append(i)

    MERGE_GROUPS = {
        101: [3, 4],
        102: [5, 6],
        103: [9, 10],
    }
    REMOVE_INSTANCES = {2}

    # Apply manual merges/removals now that we have by_iid and df
    if MERGE_GROUPS or REMOVE_INSTANCES:
        by_iid, df = apply_merges_and_removals(by_iid, df, MERGE_GROUPS, REMOVE_INSTANCES)

    # Optional CLIP
    model = preprocess = tokenizer = None
    try:
        model, preprocess, tokenizer = load_clip_model(device=args.device)
        model = model.to(args.device).eval()
    except Exception as e:
        print(f"[WARN] CLIP not loaded ({e}); material classification will be 'Unknown'.")

    # Build OBBs & planes, and per-instance materials
    obb_by_iid: Dict[int, OrientedBox] = {}
    planes_input = {'floor': None, 'ceiling': None, 'walls': []}
    per_instance_material: Dict[int, Tuple[str, float]] = {}

    # Helper to get binary mask for an instance id
    def mask_for_iid(iid: int) -> Optional[np.ndarray]:
        if mask_img is None:
            return None
        return (mask_img == iid).astype(np.uint8)

    for iid, idxs in by_iid.items():
        # skip background or invalid id
        if int(iid) == 0:
            continue
        pts_i = points[idxs]
        try:
            obb = estimate_oriented_bbox(pts_i)
            obb_by_iid[iid] = obb
        except Exception as e:
            print(f"[WARN] OBB failed for iid={iid}: {e}")

        # Material via CLIP
        inst_mask = mask_for_iid(iid)
        mat_label, mat_conf = classify_material_with_clip(pano_bgr, inst_mask, model, preprocess, tokenizer, args.device)
        per_instance_material[iid] = (mat_label, mat_conf)

    # Collect structural elements for room fit
    def _is_label(row_label: str, name: str) -> bool:
        s = str(row_label).strip().lower()
        return s in LABEL_ALIASES[name]

    for _, row in df.iterrows():
        iid = int(row.get("Instance ID", 0))
        label = str(row.get("Label", "")).strip().lower()
        if iid not in obb_by_iid:
            continue
        obb = obb_by_iid[iid]
        idxs = by_iid.get(iid, [])
        pts_i = points[idxs] if len(idxs) else None
        entry = {'plane': obb.plane, 'points': pts_i}
        if _is_label(label, 'floor'):
            planes_input['floor'] = entry
        elif _is_label(label, 'ceiling'):
            planes_input['ceiling'] = entry
        elif _is_label(label, 'wall'):
            planes_input['walls'].append(entry)

    room = None
    try:
        if planes_input['floor'] and planes_input['ceiling'] and len(planes_input['walls']) >= 1:
            room = fit_room_box(planes_input)
            print(f"[{timestamp()}] Room box fit OK. H={room.ceil_z - room.floor_z:.3f}")
    except Exception as e:
        print(f"[WARN] Room fit skipped: {e}")

    # Save a quick top-down QC image next to out_csv
    out_base, _ = os.path.splitext(args.out_csv)
    if room is not None:
        save_topdown_debug(out_base, room, planes_input)

    # Build final rows
    updated_rows = []
    corner_cols = [f"P{i}_{axis}" for i in range(1,5) for axis in ('x','y','z')]
    header = [
        "Scene Type", "Confidence", "Instance ID", "Label",
        "Instance Confidence", "Area", "Material", "Material Confidence",
        *corner_cols, "Wall Index", "Wall Name"
    ]

    for _, row in df.iterrows():
        iid = int(row.get("Instance ID", 0))
        label = str(row.get("Label", ""))
        scene_type = row.get("Scene Type", "Unknown")
        scene_conf = row.get("Confidence", 0.0)
        inst_conf = row.get("Instance Confidence", 0.0)
        area = row.get("Area", 0.0)

        mat_label, mat_conf = per_instance_material.get(iid, ("Unknown", 0.0))

        corners = None
        wall_index = ""
        wall_name = ""

        obb = obb_by_iid.get(iid)
        if obb is not None:
            if room is not None and str(label).strip().lower() in (LABEL_ALIASES['wall'] | LABEL_ALIASES['floor'] | LABEL_ALIASES['ceiling']):
                s = str(label).strip().lower()
                if s in LABEL_ALIASES['floor']:
                    corners = room.floor_corners
                elif s in LABEL_ALIASES['ceiling']:
                    corners = room.ceiling_corners
                else:  # wall
                    name = classify_wall_name(obb.plane.n, room)
                    wall_name = name.capitalize()
                    wall_index = wall_index_from_name(name, room)
                    corners = snapped_wall_corners(name, room)
            else:
                corners = obb.corners3d

        flat = _flatten_corners(corners) if corners is not None else [np.nan]*12

        updated_rows.append([
            scene_type,
            scene_conf,
            iid,
            label,
            inst_conf,
            area,
            mat_label,
            mat_conf,
            *flat,
            wall_index,
            wall_name
        ])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_rows)

    print(f"[{timestamp()}] [\u2713] Final metadata written: {args.out_csv}")


if __name__ == "__main__":
    main()
