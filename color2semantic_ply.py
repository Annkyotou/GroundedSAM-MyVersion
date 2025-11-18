#!/usr/bin/env python3
"""
color2semantic_ply.py

Convert a color-only PLY (XYZRGB) into a labeled PLY by inferring `semantic_class`
from the predefined room-color palette used in your pipeline.
Optionally remap multiple wall colors to one "Wall" class.

Default class IDs (can be overridden):
  1: Ceiling
  2: Wall
  3: Door
  4: Window
  5: Floor
  0: Other/Unknown (ignored if you want)

Usage:
  python color2semantic_ply.py \
    --in ./outputs/2ndLab-7_pano_final_metadata_colored.ply \
    --out ./outputs/2ndLab-7_pano_final_metadata_labeled_from_color.ply

Options:
  --tolerance 8        # max per-channel absolute diff to accept a color match (default 8)
  --names Ceiling,Wall,Door,Window,Floor,Other
  --ids   1,2,3,4,5,0
"""
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

# Final palette used in your Step02 (hex) converted to uint8 RGB.
# We group all wall colors into one semantic class "Wall".
PALETTE = {
    "Wall East":   (0x87, 0xA1, 0x3B),
    "Wall North":  (0x3B, 0xA1, 0x78),
    "Wall West":   (0x3B, 0x5F, 0xA1),
    "Wall South":  (0x3B, 0x40, 0xA1),
    "Floor":       (0xE3, 0xA1, 0x94),
    "Ceiling":     (0xE3, 0x77, 0xC2),
    "Door":        (0xFF, 0x7F, 0x0E),
    "Window":      (0x17, 0xBE, 0xCF),
    "Other":       (0x2D, 0x9A, 0xE8),
}

# Semantic mapping (label -> semantic_id)
DEFAULT_IDS = {
    "Ceiling": 1,
    "Wall":    2,
    "Door":    3,
    "Window":  4,
    "Floor":   5,
    "Other":   0,
}

# Which palette entries count as "Wall"
WALL_KEYS = {"Wall East", "Wall North", "Wall West", "Wall South"}

def parse_id_map(ids_str, names_str):
    # ids_str & names_str are comma separated lists with same length
    names = [s.strip() for s in names_str.split(",")]
    ids = [int(x.strip()) for x in ids_str.split(",")]
    if len(names) != len(ids):
        raise ValueError("--names and --ids lengths must match")
    return {n:i for n,i in zip(names, ids)}

def load_xyzrgb(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex'].data
    x = np.asarray(v['x'], dtype=np.float64)
    y = np.asarray(v['y'], dtype=np.float64)
    z = np.asarray(v['z'], dtype=np.float64)
    r = np.asarray(v['red'], dtype=np.uint8)
    g = np.asarray(v['green'], dtype=np.uint8)
    b = np.asarray(v['blue'], dtype=np.uint8)
    return np.stack([x,y,z], axis=-1), np.stack([r,g,b], axis=-1), ply

def write_xyzrgb_semantic(out_path, points, colors, semantics):
    n = points.shape[0]
    assert colors.shape[0] == n and semantics.shape[0] == n
    verts = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                               ('semantic_class', 'i4')])
    verts['x'] = points[:,0].astype(np.float32)
    verts['y'] = points[:,1].astype(np.float32)
    verts['z'] = points[:,2].astype(np.float32)
    verts['red']   = colors[:,0]
    verts['green'] = colors[:,1]
    verts['blue']  = colors[:,2]
    verts['semantic_class'] = semantics.astype(np.int32)
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(out_path)

def infer_semantic_from_color(rgb, tolerance, id_map):
    # Try exact/tolerant matching against palette entries.
    # Returns semantic ID
    r,g,b = rgb
    # First check walls
    for k in WALL_KEYS:
        pr,pg,pb = PALETTE[k]
        if abs(int(r)-pr) <= tolerance and abs(int(g)-pg) <= tolerance and abs(int(b)-pb) <= tolerance:
            return id_map.get("Wall", DEFAULT_IDS["Wall"])
    # Others
    for label in ["Ceiling","Floor","Door","Window","Other"]:
        pr,pg,pb = PALETTE[label] if label in PALETTE else (0,0,0)
        if abs(int(r)-pr) <= tolerance and abs(int(g)-pg) <= tolerance and abs(int(b)-pb) <= tolerance:
            return id_map.get(label, DEFAULT_IDS[label])
    # If no match, fall back to "Other"
    return id_map.get("Other", DEFAULT_IDS["Other"])

def main():
    ap = argparse.ArgumentParser(description="Infer semantic_class from XYZRGB PLY using room palette.")
    ap.add_argument("--in", dest="inp", required=False, default="./outputs/2ndLab-7_pano_final_metadata_colored.ply", help="Input color-only PLY")
    ap.add_argument("--out", required=False, default="./outputs/2ndLab-7_pano_final_metadata_labeled_from_color.ply", help="Output PLY with semantic_class")
    ap.add_argument("--tolerance", type=int, default=8, help="Per-channel tolerance for color matching (default 8)")
    ap.add_argument("--names", default="Ceiling,Wall,Door,Window,Floor,Other",
                    help="Comma-separated class names for IDs (order must match --ids)")
    ap.add_argument("--ids",   default="1,2,3,4,5,0",
                    help="Comma-separated class IDs for names (order must match --names)")
    args = ap.parse_args()

    id_map = parse_id_map(args.ids, args.names)
    pts, cols, _ = load_xyzrgb(args.inp)
    sem = np.empty(cols.shape[0], dtype=np.int32)
    for i in range(cols.shape[0]):
        sem[i] = infer_semantic_from_color(cols[i], args.tolerance, id_map)
    write_xyzrgb_semantic(args.out, pts, cols, sem)
    print(f"[OK] Wrote labeled PLY with semantic_class: {args.out}")

if __name__ == "__main__":
    main()
