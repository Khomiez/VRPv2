#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
เปรียบเทียบผลลัพธ์จาก OR-Tools กับผลลัพธ์จากแหล่งอื่น
"""

import json
from pathlib import Path

def load_or_tools_result(size: int) -> dict:
    """โหลดผลลัพธ์จาก OR-Tools"""
    result_file = Path(f"D:/projects/python/VRPv2/results_v2/vrp_solution_v2_{size}.json")
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def print_comparison(or_tools_result: dict, external_result: dict = None):
    """แสดงผลการเปรียบเทียบ"""

    print("=" * 80)
    print("เปรียบเทียบผลลัพธ์ VRP")
    print("=" * 80)

    print(f"\nผลลัพธ์จาก OR-Tools:")
    print(f"  สถานะ:           {or_tools_result['status']}")
    print(f"  ต้นทุนรวม:       {or_tools_result['total_cost']:,.2f} บาท")
    print(f"  รถที่ใช้:         {or_tools_result['num_vehicles_used']} คัน")
    print(f"  ระยะทางรวม:     {or_tools_result['total_distance_km']:,.3f} กม.")
    print(f"  ต้นทุนคงที่:     {or_tools_result['total_fixed_cost']:,.2f} บาท")
    print(f"  ต้นทุนน้ำมัน:    {or_tools_result['total_fuel_cost']:,.2f} บาท")

    if or_tools_result['validation']['all_nodes_visited']:
        print(f"  เยี่ยมครบ:        ✓ ใช่")
    else:
        print(f"  เยี่ยมครบ:        ✗ ไม่ (ขาด {len(or_tools_result['validation']['errors'])} จุด)")

    print(f"\nเส้นทาง:")
    for route in or_tools_result['routes']:
        route_str = " → ".join(map(str, route['route']))
        print(f"  รถ {route['vehicle_id']} (Type {route['vehicle_type']}):")
        print(f"    เส้นทาง: {route_str}")
        print(f"    ระยะทาง: {route['distance_km']:.3f} กม.")
        print(f"    ต้นทุน: {route['total_cost']:.2f} บาท")

    if external_result:
        print(f"\nผลลัพธ์จาก Solver อื่น:")
        print(f"  ต้นทุนรวม:       {external_result.get('total_cost', 0):,.2f} บาท")
        print(f"  รถที่ใช้:         {external_result.get('num_vehicles', 0)} คัน")
        print(f"  ระยะทางรวม:     {external_result.get('total_distance', 0):,.3f} กม.")

        # คำนวณความแตกต่าง
        cost_diff = abs(or_tools_result['total_cost'] - external_result.get('total_cost', or_tools_result['total_cost']))
        cost_pct = (cost_diff / or_tools_result['total_cost']) * 100

        print(f"\nความแตกต่าง:")
        print(f"  ต้นทุน:          {cost_diff:,.2f} บาท ({cost_pct:.2f}%)")

        if cost_pct < 1:
            print(f"  สรุป:            ✓ แทบเทียบเท่ากัน (ดีมาก!)")
        elif cost_pct < 3:
            print(f"  สรุป:            ✓ ใกล้เคียงกันมาก (ดีมาก)")
        elif cost_pct < 5:
            print(f"  สรุป:            ✓ ต่างกันเล็กน้อย (ยอมรับได้)")
        else:
            print(f"  สรุป:            ⚠ ต่างกันค่อนข้างมาก (ควรตรวจสอบ)")

    print("\n" + "=" * 80)

def main():
    """ฟังก์ชันหลัก"""
    import sys

    # ขนาดปัญหาเริ่มต้น
    size = 20

    if len(sys.argv) > 1:
        size = int(sys.argv[1])

    print(f"\nกำลังโหลดผลลัพธ์สำหรับ {size} nodes...\n")

    # โหลดผลลัพธ์จาก OR-Tools
    or_tools_result = load_or_tools_result(size)

    if or_tools_result is None:
        print(f"❌ ไม่พบไฟล์ผลลัพธ์สำหรับ {size} nodes")
        print(f"   รัน VRP solver ก่อน: python solvers/vrp_solver_v2.py")
        return

    # แสดงผลลัพธ์ OR-Tools
    print_comparison(or_tools_result)

    # ถ้ามีผลลัพธ์จาก NEOS/solver อื่น
    print(f"\nถ้าได้ผลลัพธ์จาก NEOS หรือ solver อื่น:")
    print(f"ใส่ค่าต่อไปนี้เพื่อเปรียบเทียบ:")
    print(f"  total_cost = ต้นทุนรวม (บาท)")
    print(f"  num_vehicles = จำนวนรถ")
    print(f"  total_distance = ระยะทางรวม (กม.)")

if __name__ == '__main__':
    main()
