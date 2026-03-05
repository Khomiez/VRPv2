#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert VRP JSON input to AMPL format for NEOS Server

Usage:
    python convert_to_ampl.py --size 20
    python convert_to_ampl.py --input inputs/20/vrp_input_20.json
"""

import json
import sys
from pathlib import Path

def convert_to_ampl(json_file: Path, output_dir: Path = None):
    """Convert JSON VRP input to AMPL .mod and .dat files"""

    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_nodes = data['num_nodes']
    depot_id = data['depot']['id']

    # Find checkpoint (node with zero demand that's not depot)
    checkpoint_id = None
    for node in data['nodes']:
        if node['id'] != depot_id and node['general_demand'] == 0 and node['recycle_demand'] == 0:
            checkpoint_id = node['id']
            break

    if checkpoint_id is None:
        # Use first non-depot node as checkpoint
        for node in data['nodes']:
            if node['id'] != depot_id:
                checkpoint_id = node['id']
                break

    # Set output directory
    if output_dir is None:
        output_dir = json_file.parent.parent / 'neos_files'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate .mod file
    mod_file = output_dir / f'vrp_{num_nodes}.mod'
    with open(mod_file, 'w', encoding='utf-8') as f:
        f.write(f'''# VRP Model for NEOS Server - {num_nodes} Nodes
# แบบจำลองปัญหา VRP สำหรับ NEOS Server

# ชุดข้อมูล (Sets)
set NODES;
set VEHICLES;

# พารามิเตอร์ (Parameters)
param depot_node in NODES;
param checkpoint_node in NODES;

# เมทริกซ์ระยะทาง (เมตร)
param distance {{i in NODES, j in NODES}} >= 0;

# ความต้องการขยะ (กก.)
param general_demand {{i in NODES}} >= 0;
param recycle_demand {{i in NODES}} >= 0;

# ข้อมูลรถ
param general_capacity {{k in VEHICLES}} >= 0;
param recycle_capacity {{k in VEHICLES}} >= 0;
param fixed_cost {{k in VEHICLES}} >= 0;
param fuel_cost_per_km {{k in VEHICLES}} >= 0;

param num_vehicles integer >= 1;

# ตัวแปรตัดสินใจ (Decision Variables)
var x {{i in NODES, j in NODES, k in VEHICLES}}, binary;
var vehicle_used {{k in VEHICLES}}, binary;

# MTZ subtour elimination variables (per-vehicle, all non-depot nodes including checkpoint)
var u {{i in NODES, k in VEHICLES: i != depot_node}} >= 0, <= card(NODES);

# ฟังก์ชันเป้าหมาย (Objective): ลดต้นทุนรวม
minimize Total_Cost:
  sum{{k in VEHICLES}} fixed_cost[k] * vehicle_used[k] +
  sum{{i in NODES, j in NODES, k in VEHICLES}}
    (fuel_cost_per_km[k] * distance[i,j] / 1000.0) * x[i,j,k];

# เงื่อนไขข้อจำกัด (Constraints)

# 1. แต่ละจุดเยี่ยมชมเพียงครั้งเดียว (ยกเว้น depot และ checkpoint)
subject to Visit_Once {{j in NODES: j != depot_node and j != checkpoint_node}}:
  sum{{i in NODES, k in VEHICLES}} x[i,j,k] = 1;

# 2. การไหลของจราจรที่ depot
subject to Flow_Depot_Out:
  sum{{j in NODES, k in VEHICLES}} x[depot_node,j,k] = sum{{k in VEHICLES}} vehicle_used[k];

subject to Flow_Depot_In:
  sum{{i in NODES, k in VEHICLES}} x[i,depot_node,k] = sum{{k in VEHICLES}} vehicle_used[k];

# 3. การไหลของจราจรที่จุดอื่นๆ (PER VEHICLE - ป้องกัน vehicle swapping)
subject to Flow_Conservation {{k in VEHICLES, i in NODES: i != depot_node and i != checkpoint_node}}:
  sum{{j in NODES}} x[i,j,k] = sum{{j in NODES}} x[j,i,k];

# 4. การไหลของจราจรที่ checkpoint (PER VEHICLE - ป้องกัน vehicle swapping)
subject to Flow_Checkpoint {{k in VEHICLES}}:
  sum{{i in NODES}} x[i,checkpoint_node,k] = sum{{j in NODES}} x[checkpoint_node,j,k];

# 5. ข้อจำกัดความจุขยะทั่วไป
subject to General_Capacity {{k in VEHICLES}}:
  sum{{i in NODES, j in NODES}} general_demand[j] * x[i,j,k]
  <= general_capacity[k] * vehicle_used[k];

# 6. ข้อจำกัดความจุขยะรีไซเคิล
subject to Recycle_Capacity {{k in VEHICLES}}:
  sum{{i in NODES, j in NODES}} recycle_demand[j] * x[i,j,k]
  <= recycle_capacity[k] * vehicle_used[k];

# 7. จำกัดจำนวนรถที่ใช้
subject to Max_Vehicles:
  sum{{k in VEHICLES}} vehicle_used[k] <= num_vehicles;

# 8. เชื่อมตัวแปร vehicle_used กับ x
subject to Vehicle_Used_Link {{k in VEHICLES, j in NODES: j != depot_node}}:
  x[depot_node,j,k] <= vehicle_used[k];

# 9. ป้องกันการเดินทางภายในจุดเดียวกัน
subject to No_Self_Loop {{i in NODES, k in VEHICLES}}:
  x[i,i,k] = 0;

# 10. Per-vehicle depot departure/return balance
subject to Depot_Balance {{k in VEHICLES}}:
  sum{{j in NODES}} x[depot_node,j,k] = sum{{i in NODES}} x[i,depot_node,k];

# 11. MTZ subtour elimination constraints (per vehicle, includes checkpoint)
# Covers ALL non-depot nodes including checkpoint so subtours via checkpoint are blocked.
# Per-vehicle u[i,k] avoids false infeasibility when multiple vehicles visit checkpoint.
subject to MTZ_Subtour {{i in NODES, j in NODES, k in VEHICLES:
    i != depot_node and j != depot_node and i != j}}:
  u[i,k] - u[j,k] + card(NODES) * x[i,j,k] <= card(NODES) - 1;

# 12. Every active vehicle MUST visit the checkpoint exactly once
subject to Mandatory_Checkpoint_Visit {{k in VEHICLES}}:
  sum{{i in NODES}} x[i,checkpoint_node,k] = vehicle_used[k];

# 13. Checkpoint MUST be the last stop before returning to depot
subject to Checkpoint_Is_Last {{k in VEHICLES}}:
  x[checkpoint_node,depot_node,k] = vehicle_used[k];
''')

    # Generate .dat file
    dat_file = output_dir / f'vrp_{num_nodes}.dat'

    # Build nodes list
    nodes_list = ' '.join(str(i) for i in range(1, num_nodes + 1))

    # Build vehicles list
    vehicles_list = ' '.join(v['type'] for v in data['vehicles'])

    # Build distance matrix in AMPL format
    dist_matrix_lines = []
    dist_matrix = data['distance_matrix']

    # Header with column indices (10 per row for readability)
    header = 'param distance :\n    '
    for i in range(0, num_nodes, 10):
        row_cols = [str(j) for j in range(i+1, min(i+11, num_nodes+1))]
        header += ' '.join([f'{j:>4}' for j in row_cols]) + '\n    '
    header += ':=\n'

    dist_matrix_lines.append(header)

    # Distance matrix rows
    for i in range(num_nodes):
        line = f'{i+1:>3}  '
        for j in range(num_nodes):
            line += f'{int(dist_matrix[i][j]):>5} '
            if (j + 1) % 10 == 0 and j < num_nodes - 1:
                line += '\n     '
        line += '\n'
        dist_matrix_lines.append(line)

    dist_matrix_lines.append(';\n')

    # Build demand parameters
    general_demand_lines = ['param general_demand :=\n']
    for node in data['nodes']:
        general_demand_lines.append(f'  {node["id"]:>3} {int(node["general_demand"]):>5}\n')
    general_demand_lines.append(';\n\n')

    recycle_demand_lines = ['param recycle_demand :=\n']
    for node in data['nodes']:
        recycle_demand_lines.append(f'  {node["id"]:>3} {int(node["recycle_demand"]):>5}\n')
    recycle_demand_lines.append(';\n\n')

    # Build vehicle parameters
    gen_cap_lines = ['param general_capacity :=\n']
    rec_cap_lines = ['param recycle_capacity :=\n']
    fixed_cost_lines = ['param fixed_cost :=\n']
    fuel_cost_lines = ['param fuel_cost_per_km :=\n']

    for v in data['vehicles']:
        gen_cap_lines.append(f'  {v["type"]} {int(v["general_capacity"])}\n')
        rec_cap_lines.append(f'  {v["type"]} {int(v["recycle_capacity"])}\n')
        fixed_cost_lines.append(f'  {v["type"]} {int(v["fixed_cost"])}\n')
        fuel_cost_lines.append(f'  {v["type"]} {int(v["fuel_cost_per_distance"])}\n')

    gen_cap_lines.append(';\n\n')
    rec_cap_lines.append(';\n\n')
    fixed_cost_lines.append(';\n\n')
    fuel_cost_lines.append(';\n\n')

    # Calculate minimum vehicles needed
    total_general = sum(n['general_demand'] for n in data['nodes'])
    total_recycle = sum(n['recycle_demand'] for n in data['nodes'])
    min_gen = (total_general // data['vehicles'][0]['general_capacity']) + 1
    min_rec = (total_recycle // data['vehicles'][0]['recycle_capacity']) + 1
    num_vehicles = max(min_gen, min_rec, 3)  # At least 3

    with open(dat_file, 'w', encoding='utf-8') as f:
        f.write(f'''# VRP Data - {num_nodes} Nodes
# ข้อมูลสำหรับปัญหา VRP {num_nodes} จุด

# Sets
set NODES := {nodes_list};
set VEHICLES := {vehicles_list};

# Parameters
param depot_node := {depot_id};
param checkpoint_node := {checkpoint_id};

# Distance Matrix (เมตร)
{''.join(dist_matrix_lines)}
# General Waste Demand (กก.)
{''.join(general_demand_lines)}
# Recyclable Waste Demand (กก.)
{''.join(recycle_demand_lines)}
# Vehicle Capacity - General Waste (กก.)
{''.join(gen_cap_lines)}
# Vehicle Capacity - Recyclable Waste (กก.)
{''.join(rec_cap_lines)}
# Fixed Cost per Vehicle (บาท/คัน)
{''.join(fixed_cost_lines)}
# Fuel Cost per km (บาท/กม.)
{''.join(fuel_cost_lines)}
# Maximum number of vehicles to use
param num_vehicles := {num_vehicles};

# End of data file
''')

    print(f'✅ สร้างไฟล์ AMPL เรียบร้อยแล้ว:')
    print(f'   - {mod_file}')
    print(f'   - {dat_file}')
    print(f'\nข้อมูล:')
    print(f'   - จำนวนจุด: {num_nodes}')
    print(f'   - Depot: Node {depot_id}')
    print(f'   - Checkpoint: Node {checkpoint_id}')
    print(f'   - จำนวนรถสูงสุด: {num_vehicles}')
    print(f'\nวิธีใช้:')
    print(f'   1. ไปที่ https://neos-server.org/neos/solvers/')
    print(f'   2. เลือก Category: MILP, Solver: CBC, Input: AMPL')
    print(f'   3. อัปโหลดทั้ง 2 ไฟล์ {mod_file.name} และ {dat_file.name}')
    print(f'   4. กรอกอีเมลและกด Submit')

    return mod_file, dat_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='แปลง JSON เป็น AMPL สำหรับ NEOS Server')
    parser.add_argument('--size', type=str, help='ขนาดปัญหา (20, 30, 50, 80, 138)')
    parser.add_argument('--input', type=str, help='พาธไฟล์ JSON ข้อมูลนำเข้า')
    parser.add_argument('--output', type=str, help='โฟลเดอร์ผลลัพธ์')

    args = parser.parse_args()

    # Determine input file
    if args.input:
        json_file = Path(args.input)
    elif args.size:
        base_dir = Path(__file__).parent.parent
        json_file = base_dir / 'inputs' / args.size / f'vrp_input_{args.size}.json'
    else:
        print('❌ ต้องระบุ --size หรือ --input')
        parser.print_help()
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output) if args.output else None

    # Convert
    if not json_file.exists():
        print(f'❌ ไม่พบไฟล์: {json_file}')
        sys.exit(1)

    convert_to_ampl(json_file, output_dir)
