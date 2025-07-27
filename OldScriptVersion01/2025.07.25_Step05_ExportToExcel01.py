# Step05_export_to_excel.py

import argparse
import pandas as pd

def timestamp():
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')

def merge_outputs(scene_file, material_csv, geometry_csv, output_excel):
    with open(scene_file, 'r') as f:
        scene_type_line = f.read().strip()
        scene_type = scene_type_line.split(',')[0].strip()

    # Load material and geometry CSVs
    mat_df = pd.read_csv(material_csv)
    geo_df = pd.read_csv(geometry_csv)

    # Ensure column alignment
    mat_df.rename(columns={"Instance ID": "Instance ID", "Label": "Label", "Material": "Material"}, inplace=True)

    # Join on label (and optionally instance ID if available in geometry)
    merged = pd.merge(geo_df, mat_df, on=["Label"], how="left")

    # Add scene type to all rows
    if "Scene Type" in merged.columns:
        merged.drop(columns=["Scene Type"], inplace=True)
    merged.insert(0, "Scene Type", scene_type)


    merged.to_excel(output_excel, index=False)
    print(f"[{timestamp()}] [✓] Exported structured results to: {output_excel}")

def main():
    parser = argparse.ArgumentParser(description="Step 05: Export structured results to Excel")
    parser.add_argument("-scene_file", required=True, help="Path to scene type text file")
    parser.add_argument("-material_csv", required=True, help="Material classification CSV")
    parser.add_argument("-geometry_csv", required=True, help="Geometry bounding box CSV")
    parser.add_argument("-output", default="final_output.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    merge_outputs(args.scene_file, args.material_csv, args.geometry_csv, args.output)

if __name__ == "__main__":
    main()
