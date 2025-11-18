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
import matplotlib.pyplot as plt

# ==========================================================
# Utilities & constants
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


def safe_str(x, default="Unknown"):
    if x is None:
        return default
    try:
        s = str(x)
    except Exception:
        return default
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return default
    return s


def safe_float(x, default=0.0):
    try:
        f = float(x)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, (np.floating, float)) and np.isnan(x):
            return default
        return int(x)
    except Exception:
        try:
            f = float(x)
            if np.isnan(f):
                return default
            return int(f)
        except Exception:
            return default


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
    planes: Dict[str, Plane]             # 'floor','ceiling','east','west','north','south'
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
    if model is None:
        return "Unknown", 0.0
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        crop = image_bgr[y0:y1, x0:x1]
    else:
        crop = image_bgr
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_img).unsqueeze(0).to(device)
    texts = [f"a surface made of {m}" for m in MATERIAL_CLASSES]
    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_feats = model.encode_text(text_tokens); text_feats /= text_feats.norm(dim=-1, keepdim=True)
        image_feats = model.encode_image(image_input); image_feats /= image_feats.norm(dim=-1, keepdim=True)
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
    pts = points - points.mean(axis=0)
    pts_proj = pts - (pts @ n)[:,None] * n[None,:]
    U, S, Vt = np.linalg.svd(pts_proj, full_matrices=False)
    x_axis = _normalize(Vt[0]); y_axis = _normalize(np.cross(n, x_axis))
    centroid = (points.mean(axis=0) - (points.mean(axis=0) @ n) * n)
    coords2d = np.column_stack(((points - centroid) @ x_axis, (points - centroid) @ y_axis)).astype(np.float32)
    rect = cv2.minAreaRect(coords2d)
    box = cv2.boxPoints(rect).astype(np.float32)
    order = canonicalize_quad_order(box); box = box[order]
    corners3d = np.stack([centroid + p[0]*x_axis + p[1]*y_axis for p in box], axis=0)
    plane = Plane(n=n, d=-float(n @ corners3d.mean(axis=0)))
    return OrientedBox(corners3d=corners3d, edges=[(0,1),(1,2),(2,3),(3,0)], plane=plane, x_axis=x_axis, y_axis=y_axis, centroid=centroid)


def robust_axes_from_wall_normals(walls: List[Dict], z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = []
    for w in walls:
        n = _normalize(w["plane"].n)
        nh = n - np.dot(n, z) * z
        nhn = np.linalg.norm(nh)
        if nhn < 1e-6:
            continue
        H.append(nh / nhn)
    if len(H) == 0:
        raise ValueError("No valid horizontal wall normals")
    H = np.stack(H, axis=0)
    if len(H) == 1:
        u = _normalize(H[0]); v = _normalize(np.cross(z, u)); return u, v
    M = np.abs(H @ H.T)
    np.fill_diagonal(M, 1.0)
    i, j = np.unravel_index(np.argmin(M), M.shape)
    u = _normalize(H[i])
    v0 = _normalize(H[j] - np.dot(H[j], u) * u)
    if np.linalg.norm(v0) < 1e-6:
        v0 = _normalize(np.cross(z, u))
    v = v0
    U_like, V_like = [], []
    for h in H:
        if abs(h @ u) >= abs(h @ v): U_like.append(np.sign(h @ u) * h)
        else: V_like.append(np.sign(h @ v) * h)
    if len(U_like): u = _normalize(np.mean(U_like, axis=0))
    if len(V_like): v = _normalize(np.mean(V_like, axis=0))
    u = _normalize(u - np.dot(u, z) * z)
    v = _normalize(v - np.dot(v, z) * z)
    v = _normalize(v - np.dot(v, u) * u)
    if np.linalg.norm(v) < 1e-6: v = _normalize(np.cross(z, u))
    return u, v


def fit_room_box(planes: Dict[str, Dict]) -> RoomFit:
    floor = planes.get('floor'); ceil  = planes.get('ceiling'); walls = planes.get('walls', [])
    if floor is None or ceil is None or len(walls) < 1:
        raise ValueError("Need floor, ceiling, and at least 1 wall")
    n_f = _normalize(floor['plane'].n); n_c = _normalize(ceil['plane'].n)
    if np.dot(n_f, n_c) > 0: n_c = -n_c
    z = _normalize(n_f)
    u, v = robust_axes_from_wall_normals(walls, z)
    def _proj(pts: np.ndarray, axis: np.ndarray) -> np.ndarray: return pts @ axis
    x_left_vals, x_right_vals, y_back_vals, y_front_vals = [], [], [], []
    all_pts = []
    for w in walls:
        n = _normalize(w['plane'].n); nh = _normalize(n - np.dot(n, z) * z)
        pts = w.get('points')
        if pts is not None and len(pts): all_pts.append(pts)
        du = abs(np.dot(nh, u)); dv = abs(np.dot(nh, v))
        if pts is None or len(pts) == 0: continue
        if du >= dv:
            sign = np.sign(np.dot(nh, u)); projs = _proj(pts, u)
            if sign > 0: x_right_vals.append(np.max(projs))
            else:        x_left_vals.append(np.min(projs))
        else:
            sign = np.sign(np.dot(nh, v)); projs = _proj(pts, v)
            if sign > 0: y_front_vals.append(np.max(projs))
            else:        y_back_vals.append(np.min(projs))
    if floor.get('points') is not None: all_pts.append(floor['points'])
    if ceil.get('points')  is not None: all_pts.append(ceil['points'])
    if len(all_pts) == 0: raise ValueError("No structural points available for extents")
    all_pts = np.concatenate(all_pts, axis=0)
    proj_u = _proj(all_pts, u); proj_v = _proj(all_pts, v)
    x_min, x_max = np.percentile(proj_u, [1.0, 99.0]); y_min, y_max = np.percentile(proj_v, [1.0, 99.0])
    x_left  = float(min(x_left_vals))  if len(x_left_vals)  else float(x_min)
    x_right = float(max(x_right_vals)) if len(x_right_vals) else float(x_max)
    y_back  = float(min(y_back_vals))  if len(y_back_vals)  else float(y_min)
    y_front = float(max(y_front_vals)) if len(y_front_vals) else float(y_max)
    if x_left >= x_right: x_left, x_right = float(x_min), float(x_max)
    if y_back >= y_front: y_back, y_front = float(y_min), float(y_max)
    z_floor = float(np.median(floor['points'] @ z)) if floor.get('points') is not None else -float(floor['plane'].d)
    z_ceil  = float(np.median(ceil['points']  @ z)) if ceil.get('points')  is not None else  float(ceil['plane'].d)
    def W(x,y,zs): return x*u + y*v + zs*z
    floor_corners = np.stack([ W(x_left,y_back,z_floor), W(x_right,y_back,z_floor), W(x_right,y_front,z_floor), W(x_left,y_front,z_floor) ], axis=0)
    ceiling_corners = np.stack([ W(x_left,y_back,z_ceil), W(x_right,y_back,z_ceil), W(x_right,y_front,z_ceil), W(x_left,y_front,z_ceil) ], axis=0)
    planes_out = {
        'floor':   Plane(n=z,    d=-z_floor),
        'ceiling': Plane(n=-z,   d= z_ceil),
        'east':    Plane(n=+u,   d=-x_right),
        'west':    Plane(n=-u,   d= x_left),
        'north':   Plane(n=+v,   d=-y_front),
        'south':   Plane(n=-v,   d= y_back),
    }
    wall_order_ccw_from_east = ['east','north','west','south']
    return RoomFit(z=z, u=u, v=v, floor_z=z_floor, ceil_z=z_ceil, x_left=x_left, x_right=x_right, y_back=y_back, y_front=y_front, planes=planes_out, floor_corners=floor_corners, ceiling_corners=ceiling_corners, wall_order_ccw_from_east=wall_order_ccw_from_east)


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
    du = np.dot(nh, rf.u); dv = np.dot(nh, rf.v)
    if abs(du) >= abs(dv): return 'east' if du > 0 else 'west'
    return 'north' if dv > 0 else 'south'


def wall_index_from_name(name: str, rf: RoomFit) -> int:
    return {'east':1,'north':2,'west':3,'south':4}[name]


# ==========================================================
# I/O helpers
# ==========================================================

def load_labeled_ply(ply_path: str):
    ply = PlyData.read(ply_path); data = ply['vertex'].data
    def get_field(name, default_val=None):
        try: return np.asarray(data[name])
        except ValueError:
            if default_val is None: raise
            return np.full(len(data), default_val)
    points = np.stack([get_field('x'), get_field('y'), get_field('z')], axis=-1).astype(np.float64)
    colors = np.stack([get_field('red'), get_field('green'), get_field('blue')], axis=-1).astype(np.float32) / 255.0
    instances = None
    for k in ['instance', 'instance_id', 'instances', 'label_instance', 'iid', 'instanceId']:
        try: instances = np.asarray(data[k]).astype(np.int32); break
        except ValueError: instances = None
    if instances is None: raise RuntimeError("Instance field not found in PLY (expected 'instance' or similar)")
    semantics = None
    for k in ['semantic', 'semantics', 'label_semantic', 'sid']:
        try: semantics = np.asarray(data[k]).astype(np.int32); break
        except ValueError: semantics = None
    return points, colors, semantics, instances


def load_masks(mask_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(mask_path): return None
    arr = tifffile.imread(mask_path)
    if arr.ndim == 3: arr = arr.squeeze()
    return arr


# ==========================================================
# Manual merge/remove controls
# ==========================================================
# Edit these as needed
MERGE_GROUPS = {
        101: [3, 4],
        102: [5, 6],
        103: [9, 10],
    # Example: 101: [3, 4], 102: [5, 6]
}
REMOVE_INSTANCES = set({2})  # Example: {0, 2}

def apply_merges_and_removals(by_iid: Dict[int, List[int]], df: pd.DataFrame, merge_groups: Dict[int, List[int]], remove_ids: set):
    for rid in list(remove_ids): by_iid.pop(int(rid), None)
    if len(df) and 'Instance ID' in df.columns:
        df = df[~df['Instance ID'].isin(list(remove_ids))].copy()
    remap = {int(src): int(dest) for dest, srcs in merge_groups.items() for src in srcs}
    for src, dest in list(remap.items()):
        if src in by_iid:
            by_iid.setdefault(dest, []).extend(by_iid[src]); by_iid.pop(src, None)
    if len(df) and 'Instance ID' in df.columns and remap:
        df = df.copy(); df['Instance ID'] = df['Instance ID'].apply(lambda x: remap.get(int(x), int(x)))
        df = df.groupby(['Instance ID','Label'], as_index=False).agg({'Scene Type':'first','Confidence':'first','Instance Confidence':'first','Area':'sum'})
    return by_iid, df


# ==========================================================
# QC / visualization helpers (with robust instance anchor)
# ==========================================================

def save_topdown_debug(out_path_base: str, room: RoomFit, planes_input: Dict[str, Dict]):
    try:
        samples_u, samples_v = [], []
        for w in planes_input.get('walls', []):
            pts = w.get('points')
            if pts is None or len(pts) == 0: continue
            take = min(5000, len(pts)); sel = pts[np.random.choice(len(pts), size=take, replace=False)]
            samples_u.append(sel @ room.u); samples_v.append(sel @ room.v)
        has_samples = len(samples_u) > 0
        if has_samples:
            samples_u = np.concatenate(samples_u); samples_v = np.concatenate(samples_v)
        rect_u = [room.x_left, room.x_right, room.x_right, room.x_left, room.x_left]
        rect_v = [room.y_back, room.y_back, room.y_front, room.y_front, room.y_back]
        plt.figure(figsize=(6,6))
        if has_samples: plt.scatter(samples_u, samples_v, s=1, alpha=0.3)
        plt.plot(rect_u, rect_v, linewidth=2)
        for name, (xu, yv) in {'East': (room.x_right, 0.5*(room.y_back+room.y_front)), 'West': (room.x_left,  0.5*(room.y_back+room.y_front)), 'North':(0.5*(room.x_left+room.x_right), room.y_front), 'South':(0.5*(room.x_left+room.x_right), room.y_back)}.items():
            plt.text(xu, yv, name)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('u (m)'); plt.ylabel('v (m)'); plt.title('Top-down QC (room frame)')
        png = f"{out_path_base}_topdown.png"; plt.savefig(png, dpi=150, bbox_inches='tight'); plt.close()
        print(f"[DEBUG] Saved top-down QC: {png}")
    except Exception as e: print(f"[WARN] Top-down QC failed: {e}")


def robust_instance_anchor(points: np.ndarray, room: RoomFit) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return a robust 3D anchor for a door/window instance and the nearest wall name.
    Steps: percentile-trim in (u,v,z), take medians, snap to nearest wall plane,
    and clamp inside the room footprint.
    """
    if points is None or len(points) == 0:
        return None, None
    u,v,z = room.u, room.v, room.z
    U, V, Z = points @ u, points @ v, points @ z
    mask = np.ones(len(points), dtype=bool)
    for arr, lo, hi in [(U,2,98), (V,2,98), (Z,2,98)]:
        a,b = np.percentile(arr, [lo,hi])
        mask &= (arr >= a) & (arr <= b)
    if mask.sum() < max(10, int(0.05*len(points))):
        mask = np.ones(len(points), dtype=bool)  # fallback
    u_med, v_med, z_med = np.median(U[mask]), np.median(V[mask]), np.median(Z[mask])
    # build 3D from medians
    p = u_med*u + v_med*v + z_med*z
    # choose nearest wall
    best, best_name = 1e9, None
    for name in ['east','west','north','south']:
        pl = room.planes[name]
        d = p @ pl.n + pl.d
        if abs(d) < best:
            best, best_name = abs(d), name
    if best_name is None:
        return p, None
    # snap to wall plane
    pl = room.planes[best_name]
    d = p @ pl.n + pl.d
    p = p - d*pl.n
    # clamp into footprint with small margin
    u_val, v_val = p @ u, p @ v
    u_val = np.clip(u_val, room.x_left+1e-2, room.x_right-1e-2)
    v_val = np.clip(v_val, room.y_back+1e-2, room.y_front-1e-2)
    p = u_val*u + v_val*v + z_med*z
    return p, best_name


def save_room_edges_ply(path, room: RoomFit):
    try:
        ls_points, ls_lines, base_idx = [], [], 0
        for i in range(4):
            a = room.floor_corners[i]; b = room.floor_corners[(i+1)%4]
            ls_points.append(a); ls_points.append(b); ls_lines.append([base_idx, base_idx+1]); base_idx += 2
        for i in range(4):
            a = room.ceiling_corners[i]; b = room.ceiling_corners[(i+1)%4]
            ls_points.append(a); ls_points.append(b); ls_lines.append([base_idx, base_idx+1]); base_idx += 2
        for i in range(4):
            a = room.floor_corners[i]; b = room.ceiling_corners[i]
            ls_points.append(a); ls_points.append(b); ls_lines.append([base_idx, base_idx+1]); base_idx += 2
        lineset = o3d.geometry.LineSet(); lineset.points = o3d.utility.Vector3dVector(np.array(ls_points)); lineset.lines  = o3d.utility.Vector2iVector(np.array(ls_lines, dtype=np.int32))
        o3d.io.write_line_set(path, lineset); print(f"[DEBUG] Saved room edges LineSet: {path}")
    except Exception as e: print(f"[WARN] save_room_edges_ply failed: {e}")


def save_plotly_html(out_path_base: str, room: RoomFit, all_points: np.ndarray, annotations: List[Tuple[np.ndarray,str]] = None, max_points: int = 200_000):
    try:
        import plotly.graph_objects as go
        N = len(all_points); take = min(max_points, N); sel = all_points[np.random.choice(N, size=take, replace=False)]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=sel[:,0], y=sel[:,1], z=sel[:,2], mode="markers", marker=dict(size=1), name="points"))
        fc, cc = room.floor_corners, room.ceiling_corners
        edges = [(fc[0], fc[1]), (fc[1], fc[2]), (fc[2], fc[3]), (fc[3], fc[0]), (cc[0], cc[1]), (cc[1], cc[2]), (cc[2], cc[3]), (cc[3], cc[0]), (fc[0], cc[0]), (fc[1], cc[1]), (fc[2], cc[2]), (fc[3], cc[3])]
        for a,b in edges:
            fig.add_trace(go.Scatter3d(x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]], mode="lines", line=dict(width=4), showlegend=False))
        center = 0.5*(fc.mean(axis=0) + cc.mean(axis=0))
        for vec, nm in [(room.u, "u"), (room.v, "v"), (room.z, "z")]:
            a = center; b = center + 1.0*vec
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines", line=dict(width=6), name=nm))
        # Text labels
        if annotations:
            for pos, txt in annotations:
                fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="text", text=[txt], textposition="middle center", name=txt, showlegend=False))
        fig.update_layout(scene=dict(aspectmode="data"), title="Room fit (points + snapped box)")
        html_path = f"{out_path_base}_room3d.html"; fig.write_html(html_path, include_plotlyjs=True, full_html=True)
        print(f"[DEBUG] Saved Plotly 3D HTML: {html_path}")
    except ImportError as e: print(f"[WARN] Plotly not installed: {e}. Try: pip install plotly")
    except Exception as e: print(f"[WARN] save_plotly_html failed: {e}")


# ==========================================================
# Main pipeline
# ==========================================================

def _flatten_corners(c: np.ndarray) -> List[float]:
    return [float(x) for p in c for x in p.tolist()]


def sanitize_rows(rows: List[List]):
    out = []
    for r in rows:
        r = list(r)
        # First 8 metadata columns: fill with safe types
        r[0] = safe_str(r[0])               # Scene Type
        r[1] = safe_float(r[1])             # Confidence
        r[2] = safe_int(r[2], -1)           # Instance ID
        r[3] = safe_str(r[3])               # Label
        r[4] = safe_float(r[4])             # Instance Confidence
        r[5] = safe_float(r[5])             # Area
        r[6] = safe_str(r[6])               # Material
        r[7] = safe_float(r[7])             # Material Confidence
        out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser("Step02 with Room Regularization (fixed5)")
    parser.add_argument("--pano", required=False, default="./outputs/2ndLab-7_pano.jpg")
    parser.add_argument("--mask", required=False, default="./outputs/2ndLab-7_pano_mask.tif")
    parser.add_argument("--input_metadata", required=False, default="./outputs/2ndLab-7_pano_final_metadata.csv")
    parser.add_argument("--ply", required=False, default="./outputs/2ndLab-7_pano_labeled_combined.ply")
    parser.add_argument("--out_csv", required=False, default="./outputs/2ndLab-7_pano_final_metadata.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[{timestamp()}] Loading inputs…")
    df = pd.read_csv(args.input_metadata)
    pano_bgr = cv2.imread(args.pano, cv2.IMREAD_COLOR)
    if pano_bgr is None: raise FileNotFoundError(f"Cannot read pano: {args.pano}")
    mask_img = load_masks(args.mask) if args.mask else None

    points, colors, semantics, instances = load_labeled_ply(args.ply)
    print(f"[{timestamp()}] PLY points: {points.shape[0]}")

    by_iid: Dict[int, List[int]] = defaultdict(list)
    for i, iid in enumerate(instances):
        if int(iid) == 0: continue
        by_iid[int(iid)].append(i)

    if MERGE_GROUPS or REMOVE_INSTANCES:
        by_iid, df = apply_merges_and_removals(by_iid, df, MERGE_GROUPS, REMOVE_INSTANCES)

    model = preprocess = tokenizer = None
    try:
        model, preprocess, tokenizer = load_clip_model(device=args.device); model = model.to(args.device).eval()
    except Exception as e:
        print(f"[WARN] CLIP not loaded ({e}); material classification will be 'Unknown'.")

    obb_by_iid: Dict[int, OrientedBox] = {}
    planes_input = {'floor': None, 'ceiling': None, 'walls': []}
    per_instance_material: Dict[int, Tuple[str, float]] = {}

    def mask_for_iid(iid: int) -> Optional[np.ndarray]:
        if mask_img is None: return None
        return (mask_img == iid).astype(np.uint8)

    for iid, idxs in by_iid.items():
        pts_i = points[idxs]
        try:
            obb = estimate_oriented_bbox(pts_i); obb_by_iid[iid] = obb
        except Exception as e:
            print(f"[WARN] OBB failed for iid={iid}: {e}")
        inst_mask = mask_for_iid(iid)
        mat_label, mat_conf = classify_material_with_clip(pano_bgr, inst_mask, model, preprocess, tokenizer, args.device)
        per_instance_material[iid] = (mat_label, mat_conf)

    def _is_label(s: str, name: str) -> bool:
        s = safe_str(s, "").lower(); return s in LABEL_ALIASES[name]

    for _, row in df.iterrows():
        iid = safe_int(row.get("Instance ID", 0))
        label = safe_str(row.get("Label", ""), "").lower()
        if iid not in obb_by_iid: continue
        obb = obb_by_iid[iid]
        idxs = by_iid.get(iid, [])
        pts_i = points[idxs] if len(idxs) else None
        entry = {'plane': obb.plane, 'points': pts_i}
        if _is_label(label, 'floor'): planes_input['floor'] = entry
        elif _is_label(label, 'ceiling'): planes_input['ceiling'] = entry
        elif _is_label(label, 'wall'): planes_input['walls'].append(entry)

    room = None
    try:
        if planes_input['floor'] and planes_input['ceiling'] and len(planes_input['walls']) >= 1:
            room = fit_room_box(planes_input)
            print(f"[{timestamp()}] Room box fit OK. H={room.ceil_z - room.floor_z:.3f}")
    except Exception as e: print(f"[WARN] Room fit skipped: {e}")

    out_base, _ = os.path.splitext(args.out_csv)
    annotations = []
    if room is not None:
        save_topdown_debug(out_base, room, planes_input)
        save_room_edges_ply(f"{out_base}_room_edges.ply", room)
        # Add labels for faces
        for name in ['east','north','west','south']:
            c = snapped_wall_corners(name, room); center = c.mean(axis=0)
            annotations.append((center, f"{name.title()} Wall"))
        annotations.append((room.floor_corners.mean(axis=0), "Floor"))
        annotations.append((room.ceiling_corners.mean(axis=0), "Ceiling"))

    # === Build final rows with WALL CONSOLIDATION ===
    final_rows = []
    corner_cols = [f"P{i}_{axis}" for i in range(1,5) for axis in ('x','y','z')]
    header = ["Scene Type","Confidence","Instance ID","Label","Instance Confidence","Area","Material","Material Confidence",*corner_cols,"Wall Index","Wall Name"]

    wall_acc = {}  # name -> dict(area_sum, best_iid, best_mat, best_conf)
    non_struct_rows = []

    for _, row in df.iterrows():
        iid = safe_int(row.get("Instance ID", 0))
        label = safe_str(row.get("Label", ""), "unknown")
        scene_type = safe_str(row.get("Scene Type", "Unknown"), "Unknown")
        scene_conf = safe_float(row.get("Confidence", 0.0), 0.0)
        inst_conf = safe_float(row.get("Instance Confidence", 0.0), 0.0)
        area = safe_float(row.get("Area", 0.0), 0.0)
        mat_label, mat_conf = per_instance_material.get(iid, ("Unknown", 0.0))
        obb = obb_by_iid.get(iid)
        idxs = by_iid.get(iid, [])
        pts_i = points[idxs] if len(idxs) else None

        if room is not None and label.lower() in (LABEL_ALIASES['wall'] | LABEL_ALIASES['floor'] | LABEL_ALIASES['ceiling']):
            if label.lower() in LABEL_ALIASES['floor'] or label.lower() in LABEL_ALIASES['ceiling']:
                continue
            if obb is None: continue
            name = classify_wall_name(obb.plane.n, room)
            acc = wall_acc.setdefault(name, {"scene_type":scene_type, "scene_conf":scene_conf, "area":0.0, "best_iid":iid, "best_mat":mat_label, "best_conf":mat_conf, "inst_conf":inst_conf})
            acc["area"] += area
            if mat_conf > acc["best_conf"]:
                acc.update({"best_conf":mat_conf, "best_mat":mat_label, "best_iid":iid, "inst_conf":inst_conf})
        else:
            corners = obb.corners3d if obb is not None else None
            flat = _flatten_corners(corners) if corners is not None else [np.nan]*12
            non_struct_rows.append([scene_type, scene_conf, iid, label, inst_conf, area, mat_label, mat_conf, *flat, "", ""])
            # robust door/window annotations
            if room is not None and pts_i is not None and len(pts_i):
                lbl_low = label.lower()
                if "door" in lbl_low or "window" in lbl_low:
                    p, on_wall = robust_instance_anchor(pts_i, room)
                    if p is not None:
                        txt = label.title() if on_wall is None else f"{label.title()} ({on_wall.title()})"
                        annotations.append((p, txt))

    if room is not None:
        for name in ['east','north','west','south']:
            corners = snapped_wall_corners(name, room); flat = _flatten_corners(corners)
            acc = wall_acc.get(name, None)
            if acc is None:
                final_rows.append(["Unknown", 0.0, -1, "wall", 0.0, 0.0, "Unknown", 0.0, *flat, wall_index_from_name(name, room), name.capitalize()])
            else:
                final_rows.append([ safe_str(acc["scene_type"]), safe_float(acc["scene_conf"]), safe_int(acc["best_iid"], -1), "wall", safe_float(acc["inst_conf"]), safe_float(acc["area"]), safe_str(acc["best_mat"]), safe_float(acc["best_conf"]), *flat, wall_index_from_name(name, room), name.capitalize() ])
        flat_floor = _flatten_corners(room.floor_corners); flat_ceil  = _flatten_corners(room.ceiling_corners)
        final_rows.append(["Unknown", 0.0, -2, "floor", 0.0, 0.0, "Unknown", 0.0, *flat_floor, "", ""]) 
        final_rows.append(["Unknown", 0.0, -3, "ceiling", 0.0, 0.0, "Unknown", 0.0, *flat_ceil,  "", ""]) 

    final_rows.extend(non_struct_rows)

    # sanitize first 8 columns to avoid blanks in Excel
    final_rows = sanitize_rows(final_rows)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline='') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(final_rows)
    print(f"[{timestamp()}] [\u2713] Final metadata written (consolidated walls): {args.out_csv}")

    if room is not None:
        save_plotly_html(out_base, room, points, annotations)


if __name__ == "__main__":
    main()
