# -*- coding: utf-8 -*-
"""
STEP 2 & 3: Solver Input Schema Design & Generation

SCHEMA DESIGN:
==============
We use JSON format because:
1. Human-readable and easy to validate
2. Native Python support (no external parsers needed)
3. Supports nested data structures
4. Easy to modify and debug

SCHEMA STRUCTURE:
{
    "depot": {
        "id": int,           # Depot node ID
        "name": str          # Depot name
    },
    "nodes": [
        {
            "id": int,           # Node ID (1-indexed)
            "name": str,         # Node name/description
            "general_demand": float,   # General trash demand
            "recycle_demand": float    # Recyclable trash demand
        }
    ],
    "vehicles": [
        {
            "type": str,         # Vehicle type identifier
            "general_capacity": float,   # General trash capacity
            "recycle_capacity": float,   # Recyclable trash capacity
            "fixed_cost": float,         # Fixed cost per vehicle
            "fuel_cost_per_distance": float  # Variable cost per distance unit
        }
    ],
    "distance_matrix": [[float]]  # Full NxN symmetric distance matrix
}
"""

import pandas as pd
import json
import numpy as np

def clean_and_generate_input(excel_file, sheet_name, output_file):
    """
    Clean Excel data and generate solver-ready JSON input
    """
    print(f"Processing sheet: {sheet_name}")

    # Read the Excel sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Find depot node (marked as "จุดทิ้ง")
    depot_id = None
    depot_name = "จุดทิ้ง (Depot)"

    for idx, row in df.iterrows():
        general = str(row.get('ขยะทั่วไป', '')).strip()
        if general == 'จุดทิ้ง':
            depot_id = int(row['Destination'])
            break

    if depot_id is None:
        # Fallback: assume last node is depot
        depot_id = int(df.iloc[-1]['Destination'])

    print(f"Found depot: Node {depot_id}")

    # Extract distance matrix
    num_nodes = int(df['Destination'].max())
    distance_matrix = np.zeros((num_nodes, num_nodes))

    # Build full symmetric distance matrix from lower triangular format
    for idx, row in df.iterrows():
        if pd.isna(row['Destination']):
            continue

        i = int(row['Destination']) - 1  # Convert to 0-indexed

        for j in range(1, num_nodes + 1):
            origin_col = f'Origin_{j}'
            if origin_col in df.columns:
                dist = row[origin_col]
                if pd.notna(dist):
                    j_idx = j - 1  # Convert to 0-indexed
                    distance_matrix[i][j_idx] = float(dist)
                    distance_matrix[j_idx][i] = float(dist)  # Symmetric

    # Verify diagonal is zero
    for i in range(num_nodes):
        distance_matrix[i][i] = 0.0

    # Extract nodes (excluding depot from service nodes)
    nodes = []
    total_general = 0
    total_recycle = 0

    for idx, row in df.iterrows():
        if pd.isna(row['Destination']):
            continue

        node_id = int(row['Destination'])

        # Get demands (handle "จุดทิ้ง" as 0)
        general_str = str(row.get('ขยะทั่วไป', 0)).strip()
        recycle_str = str(row.get('ขยะ recycle', 0)).strip()

        try:
            general = float(general_str) if general_str not in ['จุดทิ้ง', 'nan', ''] else 0.0
        except:
            general = 0.0

        try:
            recycle = float(recycle_str) if recycle_str not in ['จุดทิ้ง', 'nan', ''] else 0.0
        except:
            recycle = 0.0

        node_name = f"Node {node_id}"
        if node_id == depot_id:
            node_name = depot_name
            general = 0.0
            recycle = 0.0

        nodes.append({
            "id": node_id,
            "name": node_name,
            "general_demand": general,
            "recycle_demand": recycle
        })

        if node_id != depot_id:
            total_general += general
            total_recycle += recycle

    # Extract vehicle types
    vehicle_types = {}
    for idx, row in df.iterrows():
        v_type = str(row.get('รถคันที่', '')).strip()
        if pd.notna(v_type) and v_type and v_type != 'nan':
            if v_type not in vehicle_types:
                vehicle_types[v_type] = {
                    "type": v_type,
                    "general_capacity": float(row.get('cap for gereral ', 2000)),
                    "recycle_capacity": float(row.get('cap for recycle', 200)),
                    "fixed_cost": float(row.get('fix cost', 2400)),
                    "fuel_cost_per_distance": float(row.get('variable cost', 8))
                }

    vehicles = list(vehicle_types.values())

    # Build solver input
    solver_input = {
        "depot": {
            "id": depot_id,
            "name": depot_name
        },
        "num_nodes": num_nodes,
        "nodes": nodes,
        "vehicles": vehicles,
        "distance_matrix": distance_matrix.tolist(),
        "summary": {
            "total_general_demand": total_general,
            "total_recycle_demand": total_recycle,
            "num_vehicles": len(vehicles)
        }
    }

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(solver_input, f, indent=2, ensure_ascii=False)

    # Write summary to file to avoid encoding issues
    summary_file = output_file.replace('.json', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("INPUT GENERATION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"  Total nodes: {num_nodes}\n")
        f.write(f"  Depot node: {depot_id} ({depot_name})\n")
        f.write(f"  Service nodes: {num_nodes - 1}\n")
        f.write(f"  Vehicle types: {len(vehicles)}\n")
        f.write(f"  Total general demand: {total_general:.1f}\n")
        f.write(f"  Total recycle demand: {total_recycle:.1f}\n")
        f.write(f"\nVehicle specifications:\n")
        for v in vehicles:
            f.write(f"  Type {v['type']}:\n")
            f.write(f"    - General capacity: {v['general_capacity']}\n")
            f.write(f"    - Recycle capacity: {v['recycle_capacity']}\n")
            f.write(f"    - Fixed cost: {v['fixed_cost']}\n")
            f.write(f"    - Variable cost: {v['fuel_cost_per_distance']}/unit\n")
        f.write(f"\nOutput saved to: {output_file}\n")
        f.write("="*80 + "\n\n")

    print(f"\n{'='*80}")
    print(f"INPUT GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Total nodes: {num_nodes}")
    print(f"  Depot node: {depot_id}")
    print(f"  Service nodes: {num_nodes - 1}")
    print(f"  Vehicle types: {len(vehicles)}")
    print(f"  Total general demand: {total_general:.1f}")
    print(f"  Total recycle demand: {total_recycle:.1f}")
    print(f"\nVehicle specifications:")
    for v in vehicles:
        print(f"  Type {v['type']}:")
        print(f"    - General capacity: {v['general_capacity']}")
        print(f"    - Recycle capacity: {v['recycle_capacity']}")
        print(f"    - Fixed cost: {v['fixed_cost']}")
        print(f"    - Variable cost: {v['fuel_cost_per_distance']}/unit")
    print(f"\nOutput saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

    return solver_input

if __name__ == "__main__":
    excel_file = 'd:/projects/python/VRPv2/distance_matrix_full_138_zones-edit4.xlsx'

    # Use the 138 sheet (full dataset)
    output_file = 'd:/projects/python/VRPv2/vrp_input.json'

    solver_input = clean_and_generate_input(excel_file, '138', output_file)

    print("Solver input file generated successfully!")
