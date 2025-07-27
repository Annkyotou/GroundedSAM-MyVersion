# Step03_material_classification.py

import os
import cv2
import torch
import open_clip
import argparse
import numpy as np
from PIL import Image
import tifffile


def timestamp():
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')

# List of materials in the material bank
MATERIAL_CLASSES = [
    "Aluminum", "Brick", "Carpet", "Cement board", "Concrete", "Concrete block",
    "Felt", "GFRC (Glass fiber reinforced concrete)", "Glass", "Gypsum board", "Steel",
    "Stone panel", "Stucco", "Synthetic rubber", "Terrazzo", "Vinyl composition tile (VCT)", "Wood"
]

# Use natural descriptive phrases for better CLIP response
MATERIAL_PROMPTS = [f"a surface made of {label}" for label in MATERIAL_CLASSES]


def extract_patch(image_path, mask_path, instance_id):
    image = cv2.imread(image_path)
    mask = tifffile.imread(mask_path)

    region = (mask == instance_id).astype(np.uint8)
    if np.sum(region) < 100:
        print(f"[WARN] Too small region for instance {instance_id}, skipping...")
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


def main():
    parser = argparse.ArgumentParser(description="Step 03: Material Classification using CLIP")
    parser.add_argument("-image", required=True, help="Path to the input panorama image")
    parser.add_argument("-mask", required=True, help="Path to the labeled mask image (.tif)")
    parser.add_argument("-instances_txt", required=True, help="Path to instance-to-class text file")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_csv = f"./outputs/{base_name}_materials.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    with open(args.instances_txt, "r") as f:
        instance_map = {int(line.split(":")[0]): line.split(":")[1].strip() for line in f if ":" in line}

    results = []

    for instance_id, label in instance_map.items():
        crop = extract_patch(args.image, args.mask, instance_id)
        if crop is None:
            continue
        material, confidence = classify_material(crop, model, preprocess, tokenizer, device)
        print(f"[{timestamp()}] [✓] Instance {instance_id}: {label} => {material} ({confidence:.2%})")
        results.append((instance_id, label, material, confidence))

    # Save to CSV
    import csv
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Instance ID", "Label", "Material", "Confidence"])
        writer.writerows(results)
    print(f"[{timestamp()}] [✓] Material classification saved to {output_csv}")

if __name__ == "__main__":
    main()
