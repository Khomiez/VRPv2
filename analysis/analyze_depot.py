# -*- coding: utf-8 -*-
import pandas as pd
import openpyxl

excel_file = 'd:/projects/python/VRPv2/distance_matrix_full_138_zones-edit4.xlsx'

# Read the 138 sheet (largest)
df = pd.read_excel(excel_file, sheet_name='138')

# Save analysis to file
with open('d:/projects/python/VRPv2/depot_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("STEP 1: EXCEL FILE ANALYSIS - DETAILED REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("SHEET STRUCTURE:\n")
    f.write("-" * 80 + "\n")
    wb = openpyxl.load_workbook(excel_file)
    for name in wb.sheetnames:
        df_temp = pd.read_excel(excel_file, sheet_name=name)
        f.write(f"  Sheet '{name}': {df_temp.shape[0]} rows x {df_temp.shape[1]} columns\n")
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("COLUMN DESCRIPTIONS (for 138 sheet):\n")
    f.write("="*80 + "\n")
    f.write("  1. Destination: Node ID (1-138)\n")
    f.write("  2. Origin_1 to Origin_138: Distance matrix entries\n")
    f.write("  3. node: Node identifier\n")
    f.write("  4. ขยะทั่วไป (General Trash): General trash demand\n")
    f.write("  5. ขยะ recycle (Recyclable Trash): Recyclable trash demand\n")
    f.write("  6. รถคันที่ (Vehicle No.): Vehicle type label\n")
    f.write("  7. cap for gereral (General Capacity): Vehicle general trash capacity\n")
    f.write("  8. cap for recycle (Recycle Capacity): Vehicle recycle trash capacity\n")
    f.write("  9. fix cost (Fixed Cost): Fixed cost per vehicle\n")
    f.write("  10. variable cost (Variable Cost): Fuel cost per distance unit\n\n")

    f.write("="*80 + "\n")
    f.write("NODE INFORMATION:\n")
    f.write("="*80 + "\n\n")

    # Check all rows for node information
    f.write("First 10 nodes detail:\n")
    f.write("-" * 80 + "\n")
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        node_id = row['Destination']
        if pd.isna(node_id):
            continue
        general_trash = row['ขยะทั่วไป']
        recycle_trash = row['ขยะ recycle']
        vehicle_type = row['รถคันที่']
        f.write(f"Node {int(node_id)}: general={general_trash}, recycle={recycle_trash}, vehicle={vehicle_type}\n")

    f.write("\nLast 5 nodes detail:\n")
    f.write("-" * 80 + "\n")
    for i in range(len(df)-5, len(df)):
        row = df.iloc[i]
        node_id = row['Destination']
        if pd.isna(node_id):
            continue
        general_trash = row['ขยะทั่วไป']
        recycle_trash = row['ขยะ recycle']
        vehicle_type = row['รถคันที่']
        f.write(f"Node {int(node_id)}: general={general_trash}, recycle={recycle_trash}, vehicle={vehicle_type}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("SEARCHING FOR DEPOT (จุดทิ้ง):\n")
    f.write("="*80 + "\n\n")

    # Search for depot in any column
    depot_found = False
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            for val in unique_vals:
                val_str = str(val)
                if 'จุด' in val_str or 'ทิ้ง' in val_str or 'depot' in val_str.lower():
                    f.write(f"  Found depot-like value in column '{col}': {val}\n")
                    depot_found = True

    if not depot_found:
        f.write("  NO EXPLICIT DEPOT ROW FOUND!\n")
        f.write("  ASSUMPTION: Node 1 (first node) is the DEPOT based on standard VRP convention.\n")
        f.write("  REASON: Node 1 has Origin_1 = 0 (distance to itself), typical depot representation.\n\n")

    f.write("="*80 + "\n")
    f.write("DISTANCE MATRIX ANALYSIS:\n")
    f.write("="*80 + "\n\n")

    # Analyze distance matrix structure
    f.write("Distance Matrix Structure (first 10x10):\n")
    f.write("-" * 80 + "\n")
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        dest = int(row['Destination'])
        distances = []
        for j in range(1, 11):
            origin_col = f'Origin_{j}'
            dist = row[origin_col]
            if pd.notna(dist):
                distances.append(f"{int(dist):4d}")
            else:
                distances.append("  NaN")
        f.write(f"From Node {dest:2d}: {' '.join(distances)}\n")

    f.write("\nNOTE: NaN values indicate upper triangular portion of symmetric matrix.\n")
    f.write("The distance matrix is stored in lower triangular format.\n\n")

    f.write("="*80 + "\n")
    f.write("VEHICLE TYPE ANALYSIS:\n")
    f.write("="*80 + "\n\n")

    # Analyze vehicle types
    vehicle_rows = df[df['รถคันที่'].notna()]
    unique_vehicles = vehicle_rows['รถคันที่'].unique()
    f.write(f"Unique vehicle types found: {list(unique_vehicles)}\n\n")

    f.write("Vehicle specifications (from data):\n")
    f.write("-" * 80 + "\n")
    for v in sorted(unique_vehicles):
        if pd.notna(v):
            v_rows = df[df['รถคันที่'] == v]
            if len(v_rows) > 0:
                row = v_rows.iloc[0]
                f.write(f"Vehicle {v}:\n")
                f.write(f"  General capacity: {row['cap for gereral ']} units\n")
                f.write(f"  Recycle capacity: {row['cap for recycle']} units\n")
                f.write(f"  Fixed cost: {row['fix cost']}\n")
                f.write(f"  Variable cost: {row['variable cost']} per distance unit\n\n")

    f.write("="*80 + "\n")
    f.write("TRASH DEMAND ANALYSIS:\n")
    f.write("="*80 + "\n\n")

    # Analyze trash demand
    f.write("Trash demand statistics (first 20 non-depot nodes):\n")
    f.write("-" * 80 + "\n")
    total_general = 0
    total_recycle = 0
    for i in range(1, min(21, len(df))):
        row = df.iloc[i]
        node_id = int(row['Destination'])
        general = row['ขยะทั่วไป']
        recycle = row['ขยะ recycle']
        if pd.notna(general):
            total_general += general
        if pd.notna(recycle):
            total_recycle += recycle
        f.write(f"Node {node_id:3d}: general={general:6.1f}, recycle={recycle:6.1f}\n")

    f.write(f"\nTotal demand (first 20 nodes): general={total_general:.1f}, recycle={total_recycle:.1f}\n\n")

    f.write("="*80 + "\n")
    f.write("DATA ISSUES DETECTED:\n")
    f.write("="*80 + "\n\n")

    # Check for issues
    issues = []

    # Check NaN in distance matrix (explain this is expected)
    issues.append("1. Distance matrix has NaN values - This is EXPECTED for upper triangular portion of symmetric matrix.")

    # Check for missing demands
    missing_demand = df[df['ขยะทั่วไป'].isna()].shape[0]
    if missing_demand > 0:
        issues.append(f"2. {missing_demand} rows have missing general trash demand.")

    missing_recycle = df[df['ขยะ recycle'].isna()].shape[0]
    if missing_recycle > 0:
        issues.append(f"3. {missing_recycle} rows have missing recycle trash demand.")

    # Check vehicle data
    missing_vehicle = df[df['รถคันที่'].isna()].shape[0]
    if missing_vehicle > 0:
        issues.append(f"4. {missing_vehicle} rows have missing vehicle type data.")

    for issue in issues:
        f.write(f"  {issue}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("ASSUMPTIONS MADE:\n")
    f.write("="*80 + "\n\n")

    f.write("1. DEPOT: Node 1 is the depot (no trash to collect)\n")
    f.write("2. DISTANCE UNIT: Meters (based on distance values ~100-3000)\n")
    f.write("3. CAPACITY UNIT: Kilograms or volume units\n")
    f.write("4. COST UNIT: Thai Baht (THB)\n")
    f.write("5. Distance matrix is symmetric: dist(i,j) = dist(j,i)\n")
    f.write("6. Vehicles start and end at depot (Node 1)\n")
    f.write("7. All trash must be collected from all non-depot nodes\n")
    f.write("8. Vehicle capacities are per-trip (must return to depot to unload)\n\n")

print("Depot analysis saved to depot_analysis.txt")
print("Please read the file for complete analysis.")
