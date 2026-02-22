# คู่มือการใช้งานระบบวางแผนเส้นทางการเก็บขยะ (VRP Solver v2)

## สารบัญ
1. [ภาพรวมของระบบ](#ภาพรวมของระบบ)
2. [โครงสร้างไฟล์สำคัญ](#โครงสร้างไฟล์สำคัญ)
3. [ข้อมูลเข้าและข้อมูลออก](#ข้อมูลเข้าและข้อมูลออก)
4. [ส่วนประกอบสำคัญของโค้ด](#ส่วนประกอบสำคัญของโค้ด)
5. [อัลกอริทึมที่ใช้](#อัลกอริทึมที่ใช้)
6. [ข้อจำกัดและเงื่อนไขพิเศษ](#ข้อจำกัดและเงื่อนไขพิเศษ)
7. [การใช้งานระบบ](#การใช้งานระบบ)
8. [ตัวอย่างผลลัพธ์](#ตัวอย่างผลลัพธ์)

---

## ภาพรวมของระบบ

ระบบนี้เป็นเครื่องมือสำหรับแก้ปัญหา **Vehicle Routing Problem (VRP)** 

### จุดเด่นของระบบ
- ✅ รองรับการวางแผนเส้นทางพร้อมกันหลายขนาดพื้นที่ (20, 30, 50, 80, 138 จุดเก็บ)
- ✅ แยกประเภทขยะได้ 2 ชนิด: ขยะทั่วไป และ ขยะรีไซเคิล
- ✅ บังคับให้แต่ละเส้นทางต้องผ่าน "จุดทิ้งขยะ" ก่อนกลับ
- ✅ เลือกประเภทรถที่เหมาะสมที่สุดตามต้นทุน
- ✅ สร้างกราฟและแผนภูมิวิเคราะห์ผลลัพธ์ได้

---

## โครงสร้างไฟล์สำคัญ

```
VRPv2/
├── solvers/
│   └── vrp_solver_v2.py              # โปรแกรมหลักสำหรับแก้ปัญหา VRP
├── visualizations/
│   └── generate_visualizations.py    # สร้างกราฟและแผนภูมิผลลัพธ์
├── analysis/
│   ├── generate_solver_input.py      # แปลงข้อมูล Excel เป็น JSON
│   ├── analyze_excel.py              # วิเคราะห์ไฟล์ Excel
│   └── analyze_depot.py              # วิเคราะห์ตำแหน่ง Depot
├── create_template.py                # สร้างเทมเพลต Excel ว่าง
├── convert_template_to_json.py       # แปลงเทมเพลตเป็น JSON
├── data/                             # เก็บไฟล์ Excel ข้อมูลนำเข้า
└── outputs/                          # เก็บผลลัพธ์
```

### ไฟล์ที่ต้องใช้งานจริง

| ไฟล์ | วัตถุประสงค์ | ความสำคัญ |
|------|-------------|-----------|
| `solvers/vrp_solver_v2.py` | โปรแกรมหลักแก้ปัญหา VRP | ⭐⭐⭐⭐⭐ |
| `visualizations/generate_visualizations.py` | สร้างกราฟผลลัพธ์ | ⭐⭐⭐⭐ |
| `create_template.py` | สร้างเทมเพลต Excel ใหม่ | ⭐⭐⭐ |
| `convert_template_to_json.py` | แปลงข้อมูล | ⭐⭐⭐ |

---

## ข้อมูลเข้าและข้อมูลออก

### ข้อมูลนำเข้า (Input)

#### 1. รูปแบบไฟล์ Excel

ไฟล์ Excel ต้องมีโครงสร้างดังนี้:

**Sheet ชื่อ "Distance_Matrix"**:
```
| Destination | Origin_1 | Origin_2 | ... | Origin_138 |
|-------------|----------|----------|-----|------------|
| 1           | 0        | 1500     | ... | 5200       |
| 2           | 1500     | 0        | ... | 4800       |
| ...         | ...      | ...      | ... | ...        |
```

**คอลัมน์สำคัญ**:
- `Destination`: หมายเลขปลายทาง (1-138)
- `Origin_1` ถึง `Origin_138`: ระยะห่างจากต้นทางถึงปลายทาง (เป็นเมตร)
- `ขยะทั่วไป`: ปริมาณขยะทั่วไปต่อจุด (กิโลกรัม)
- `ขยะ recycle`: ปริมาณขยะรีไซเคิลต่อจุด (กิโลกรัม)
- `รถคันที่`: รหัสประเภทรถ (เช่น V1, V2, V3)
- `cap for general/gereral`: ความจุรถสำหรับขยะทั่วไป
- `cap for recycle`: ความจุรถสำหรับขยะรีไซเคิล
- `fix cost`: ต้นทุนคงที่ต่อคัน (บาท)
- `variable cost`: ต้นทุนน้ำมันต่อกิโลเมตร (บาท/กม.)

#### 2. ข้อกำหนดของข้อมูล
- ระยะห่างต้องเป็น **เมตร** (meters)
- ต้นทุนน้ำมันเป็น **บาทต่อกิโลเมตร**
- Node 1 = จุดรวมรถ (Depot) เสมอ
- Node ที่มีชื่อ "จุดทิ้ง" = จุดทิ้งขยะ

### ข้อมูลส่งออก (Output)

#### 1. ไฟล์ JSON Solution
แต่ละไฟล์มีโครงสร้างดังนี้:

```json
{
  "sheet_name": "20_nodes",
  "status": "OPTIMAL",
  "total_distance": 45.2,
  "total_cost": 1250.50,
  "routes": [
    {
      "vehicle_id": 1,
      "vehicle_type": "V1",
      "nodes": [1, 5, 8, 12, 2, 1],
      "distance_km": 15.3,
      "general_load": 850.5,
      "recycle_load": 120.0,
      "general_utilization": 85.05,
      "recycle_utilization": 24.0,
      "fixed_cost": 500.0,
      "fuel_cost": 183.6,
      "total_cost": 683.6
    }
  ],
  "summary": {
    "num_vehicles": 3,
    "total_cost": 1250.50,
    "total_distance": 45.2
  }
}
```

#### 2. กราฟและแผนภูมิ
- แผนภูมิวงกลม: สัดส่วนต้นทุนคงที่ vs ต้นทุนน้ำมัน
- กราฟแท่ง: ต้นทุนแยกต่อคัน
- กราฟเส้นทาง: แสดงลำดับการเดินทาง
- กราฟเปรียบเทียบ: เปรียบเทียบผลลัพธ์หลายขนาดพื้นที่

---

## ส่วนประกอบสำคัญของโค้ด

### 1. Class VRPSolverV2 (`solvers/vrp_solver_v2.py`)

เป็นคลาสหลักที่รวมฟังก์ชันการทำงานทั้งหมด

```python
class VRPSolverV2:
    def __init__(self, input_file: str, output_dir: str):
        """
        กำหนดค่าเริ่มต้น
        - input_file: พาธไฟล์ JSON ข้อมูลนำเข้า
        - output_dir: โฟลเดอร์ที่เก็บผลลัพธ์
        """
```

#### Methods สำคัญ:

**`solve_all()`** - แก้ปัญหาทุก Sheet
```python
def solve_all(self):
    """
    วนลูปแก้ปัญหา VRP สำหรับทุก sheet ในไฟล์ Excel
    เหมาะสำหรับกรณีมีหลายขนาดพื้นที่ (20, 30, 50, 80, 138 จุด)
    """
```

**`solve(sheet_name)`** - แก้ปัญหาเพียง Sheet เดียว
```python
def solve(self, sheet_name: str):
    """
    แก้ปัญหา VRP สำหรับ sheet ที่ระบุ
    คืนค่า Solution object
    """
```

**`solve_with_ortools()`** - ใช้อัลกอริทึม OR-Tools
```python
def solve_with_ortools(self, data, vehicle_type):
    """
    ใช้ Google OR-Tools แก้ปัญหา
    - สร้าง Routing Index Manager
    - กำหนด Constraints
    - ใช้ Guided Local Search
    """
```

**`solve_with_heuristic()`** - ใช้วิธี Heuristic
```python
def solve_with_heuristic(self, data, vehicle_type):
    """
    วิธีสำรองเมื่อ OR-Tools ล้มเหลว
    - ใช้ Nearest Neighbor
    - มั่นใจว่าได้ผลลัพธ์เสมอ
    """
```

### 2. Data Structures

#### Node - แทนแต่ละจุดเก็บขยะ
```python
@dataclass
class Node:
    id: int                      # หมายเลขจุด (1-indexed)
    name: str                    # ชื่อจุด
    general_demand: float        # ปริมาณขยะทั่วไป (กก.)
    recycle_demand: float        # ปริมาณขยะรีไซเคิล (กก.)
    is_depot: bool = False       # True ถ้าเป็นจุดรวมรถ
    is_checkpoint: bool = False  # True ถ้าเป็นจุดทิ้งขยะ
```

#### Vehicle - ข้อมูลรถ
```python
@dataclass
class Vehicle:
    type_id: str                  # รหัสประเภทรถ (เช่น "V1")
    general_capacity: float       # ความจุขยะทั่วไป (กก.)
    recycle_capacity: float       # ความจุขยะรีไซเคิล (กก.)
    fixed_cost: float             # ต้นทุนคงที่ (บาท)
    fuel_cost_per_km: float       # ต้นทุนน้ำมัน (บาท/กม.)
```

#### Route - เส้นทางเดินรถ
```python
@dataclass
class Route:
    vehicle_id: int               # หมายเลขรถ
    vehicle_type: str             # ประเภทรถ
    nodes: List[int]              # ลำดับจุดที่เยี่ยมชม
    distance_meters: float        # ระยะทางรวม (เมตร)
    distance_km: float            # ระยะทางรวม (กม.)
    general_load: float           # น้ำหนักขยะทั่วไป (กก.)
    recycle_load: float           # น้ำหนักขยะรีไซเคิล (กก.)
    general_utilization: float    # % การใช้งานความจุทั่วไป
    recycle_utilization: float    # % การใช้งานความจุรีไซเคิล
    fixed_cost: float             # ต้นทุนคงที่
    fuel_cost: float              # ต้นทุนน้ำมัน
    total_cost: float             # ต้นทุนรวม
```

#### Solution - ผลลัพธ์ทั้งหมด
```python
@dataclass
class Solution:
    sheet_name: str               # ชื่อ sheet
    status: str                   # สถานะ (OPTIMAL, FEASIBLE)
    routes: List[Route]           # รายการเส้นทางทั้งหมด
    total_distance: float         # ระยะทางรวม (กม.)
    total_fixed_cost: float       # ต้นทุนคงที่รวม
    total_fuel_cost: float        # ต้นทุนน้ำมันรวม
    total_cost: float             # ต้นทุนรวม
    num_vehicles: int             # จำนวนรถที่ใช้
    computation_time: float       # เวลาคำนวณ (วินาที)
```

### 3. Visualization Generator (`visualizations/generate_visualizations.py`)

ฟังก์ชันหลักในการสร้างกราฟ:

**`generate_cost_breakdown_charts()`** - กราฟต้นทุน
```python
def generate_cost_breakdown_charts(solution, output_dir):
    """
    สร้างกราฟแสดงสัดส่วนต้นทุนคงที่และต้นทุนน้ำมัน
    - Pie chart: แสดงสัดส่วนต้นทุน
    - Bar chart: เปรียบเทียบต้นทุนแต่ละคัน
    """
```

**`generate_route_visualization()`** - กราฟเส้นทาง
```python
def generate_route_visualization(route, output_dir):
    """
    สร้างกราฟแสดงเส้นทางการเดินรถ
    - ใช้สีแดง = Depot
    - ใช้สีเหลือง = จุดทิ้งขยะ
    - ใช้สีฟ้า = จุดเก็บขยะ
    """
```

**`generate_summary_comparison()`** - กราฟเปรียบเทียบ
```python
def generate_summary_comparison(solutions, output_dir):
    """
    สร้างกราฟเปรียบเทียบผลลัพธ์หลายขนาดพื้นที่
    - แสดงต้นทุนรวม
    - แสดงระยะทางรวม
    - แสดงจำนวนรถที่ใช้
    """
```

---

## อัลกอริทึมที่ใช้

### 1. Google OR-Tools (วิธีหลัก)

**คืออะไร?**
- ไลบรารีแก้ปัญหา Optimization จาก Google
- ใช้ Constraint Programming และ Metaheuristics

**วิธีทำงาน**:
```python
# 1. สร้าง Routing Index Manager
manager = pywrapcp.RoutingIndexManager(
    len(data['distance_matrix']),
    num_vehicles,
    data['depot']
)

# 2. สร้าง Routing Model
routing = pywrapcp.RoutingModel(manager)

# 3. กำหนด Distance Callback
def distance_callback(from_index, to_index):
    # คืนค่าระยะห่างระหว่างสองจุด
    return data['distance_matrix'][from_node][to_node]

# 4. กำหนด Capacity Constraints
# - สำหรับขยะทั่วไป
# - สำหรับขยะรีไซเคิล

# 5. กำหนด Checkpoint Constraint
# บังคับให้ผ่าน "จุดทิ้ง" ก่อนกลับ Depot

# 6. แก้ปัญหาด้วย Guided Local Search
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30  # จำกัดเวลา 30 วินาที
```

**ข้อดี**:
- ✅ ได้ผลลัพธ์ที่ดีมาก (Near-optimal)
- ✅ รวดเร็วสำหรับปัญหาขนาดเล็ก-กลาง

### 2. Heuristic Solver (วิธีสำรอง)

**คืออะไร?**
- วิธีแบบ Heuristic แบบง่าย
- ใช้เมื่อ OR-Tools ไม่สามารถหาคำตอบได้

**วิธีทำงาน (Nearest Neighbor)**:
```python
def solve_with_heuristic(data, vehicle_type):
    # 1. เริ่มต้นที่ Depot (Node 1)
    current_node = depot

    # 2. วนลูปจนกว่าจะเต็มความจุหรือไม่มีจุดเหลือ
    while can_add_more_nodes:
        # 3. หาจุดที่ใกล้ที่สุดที่ยังไม่ไป
        nearest = find_nearest_unvisited_node(current_node)

        # 4. ตรวจสอบว่าเพิ่มได้ไหม (เช็ค Capacity)
        if fits_capacity(nearest):
            route.append(nearest)
            current_node = nearest
        else:
            break

    # 5. บังคับให้ไป Checkpoint ("จุดทิ้ง")
    route.append(checkpoint_node)

    # 6. กลับ Depot
    route.append(depot)
```

**ข้อดี**:
- ✅ ได้ผลลัพธ์เสมอ (Guaranteed)
- ✅ เข้าใจง่าย
- ❌ ผลลัพธ์อาจไม่ดีเท่า OR-Tools

---

## ข้อจำกัดและเงื่อนไขพิเศษ

### 1. เงื่อนไขเส้นทางบังคับ

**รูปแบบเส้นทาง**:
```
Depot → จุดเก็บขยะ → จุดทิ้ง → Depot
  1   →  5, 8, 12   →   2   →  1
```

**เหตุผล**:
- รถเก็บขยะต้องไปทิ้งขยะก่อนจะกลับจุดรวมรถ
- เพื่อลดกลิ่นและความสกปรกในรถ

### 2. ข้อจำกัดความจุ (Capacity Constraints)

แต่ละรถมี **2 ความจุ** ที่ต้องเคร่งครัด:

```python
# ต้องไม่เกินความจุทั้ง 2 ชนิด
route.general_load ≤ vehicle.general_capacity
route.recycle_load ≤ vehicle.recycle_capacity
```

**ตัวอย่าง**:
```python
# รถ V1
general_capacity = 1000  # กก.
recycle_capacity = 500   # กก.

# เส้นทาง
route.general_load = 850    # กก. (85%)
route.recycle_load = 320    # กก. (64%)

# ✅ ผ่านทั้งคู่
```

### 3. เงื่อนไข Depot และ Checkpoint

```python
# Depot: Node 1 เสมอ
depot_node = 1

# Checkpoint: Node ที่มีชื่อ "จุดทิ้ง"
checkpoint_node = [node for node in nodes if node.name == "จุดทิ้ง"][0]
```

### 4. การเลือกประเภทรถ

ระบบจะเลือกประเภทรถที่ให้ **ต้นทุนต่ำสุด**:

```python
# ทดลองทุกประเภทรถ
best_cost = infinity
best_solution = None

for vehicle_type in vehicle_types:
    # แก้ปัญหาด้วยประเภทรถนี้
    solution = solve_with_vehicle(vehicle_type)

    # เลือกที่ต้นทุนต่ำสุด
    if solution.total_cost < best_cost:
        best_cost = solution.total_cost
        best_solution = solution
```

---

## การใช้งานระบบ

### ขั้นตอนที่ 1: สร้างเทมเพลต Excel

```bash
# รันคำสั่ง
python create_template.py

# ผลลัพธ์: ไฟล์ data/vrp_template.xlsx
```

### ขั้นตอนที่ 2: กรอกข้อมูล

เปิดไฟล์ `data/vrp_template.xlsx` แล้วกรอก:

1. **Distance Matrix**: ระยะห่างระหว่างจุดต่างๆ
2. **Demand**: ปริมาณขยะทั่วไปและรีไซเคิล
3. **Vehicle Info**: ความจุและต้นทุนรถ

### ขั้นตอนที่ 3: แปลงเป็น JSON

```bash
python convert_template_to_json.py --input data/vrp_template.xlsx --output inputs/
```

### ขั้นตอนที่ 4: แก้ปัญหา VRP

```bash
# แก้ปัญหาทุก Sheet
python solvers/vrp_solver_v2.py --input inputs/vrp_data.json --output outputs/

# หรือใช้โค้ด
from solvers.vrp_solver_v2 import VRPSolverV2

solver = VRPSolverV2("inputs/vrp_data.json", "outputs/")
solver.solve_all()
```

### ขั้นตอนที่ 5: สร้างกราฟ

```bash
python visualizations/generate_visualizations.py --solutions outputs/ --charts outputs/charts/
```

---

## ตัวอย่างผลลัพธ์

### 1. ไฟล์ Solution (`outputs/20_nodes_solution.json`)

```json
{
  "sheet_name": "20_nodes",
  "status": "OPTIMAL",
  "total_distance": 125.3,
  "total_cost": 3580.75,
  "routes": [
    {
      "vehicle_id": 1,
      "vehicle_type": "V2",
      "nodes": [1, 5, 8, 12, 3, 2, 1],
      "distance_km": 42.5,
      "general_load": 1850.5,
      "recycle_load": 420.0,
      "general_utilization": 92.53,
      "recycle_utilization": 56.0,
      "fixed_cost": 1500.0,
      "fuel_cost": 510.0,
      "total_cost": 2010.0
    },
    {
      "vehicle_id": 2,
      "vehicle_type": "V1",
      "nodes": [1, 15, 18, 20, 7, 2, 1],
      "distance_km": 38.2,
      "general_load": 850.0,
      "recycle_load": 180.0,
      "general_utilization": 85.0,
      "recycle_utilization": 36.0,
      "fixed_cost": 800.0,
      "fuel_cost": 458.4,
      "total_cost": 1258.4
    }
  ],
  "summary": {
    "num_vehicles": 2,
    "total_distance": 80.7,
    "total_fixed_cost": 2300.0,
    "total_fuel_cost": 968.4,
    "total_cost": 3268.4,
    "computation_time": 2.3
  }
}
```

### 2. กราฟที่สร้างได้

**Cost Breakdown Chart**:
- แสดงสัดส่วนต้นทุนคงที่ vs ต้นทุนน้ำมัน
- ประมาณ 70% ต้นทุนคงที่, 30% ต้นทุนน้ำมัน

**Capacity Utilization Chart**:
- แสดง % การใช้งานความจุรถแต่ละคัน
- ส่วนใหญ่ 80-95% (ใช้งานดี)

**Route Visualization**:
- แสดงลำดับการเดินรถเป็นวงกลม
- Depot = สีแดง, จุดทิ้ง = สีเหลือง, จุดเก็บ = สีฟ้า

**Summary Comparison**:
- เปรียบเทียบผลลัพธ์ 20, 30, 50, 80, 138 จุด
- พบว่า 20 จุดใช้รถ 2 คัน, 138 จุดใช้รถ 8 คัน

---

## สรุป

ระบบ VRP Solver v2 นี้ถูกออกแบบมาเพื่อแก้ปัญหาการวางแผนเส้นทางเก็บขยะของเทศบาล โดยมีจุดเด่นคือ:

1. **รองรับ 2 ชนิดขยะ**: แยกความจุสำหรับขยะทั่วไปและรีไซเคิล
2. **บังคับผ่านจุดทิ้ง**: เหมาะกับงานเก็บขยะจริง
3. **เลือกรถอัตโนมัติ**: ใช้ประเภทรถที่คุ้มค่าที่สุด
4. **ได้ผลลัพธ์คุณภาพ**: ใช้ Google OR-Tools ช่วย
5. **มีกราฟวิเคราะห์**: เข้าใจผลลัพธ์ง่าย

**ไฟล์ที่ต้องใช้จริง**:
- `solvers/vrp_solver_v2.py` - แก้ปัญหา
- `visualizations/generate_visualizations.py` - สร้างกราฟ
- `create_template.py` - สร้างเทมเพลต
- `convert_template_to_json.py` - แปลงข้อมูล

หากต้องการข้อมูลเพิ่มเติม กรุณาติดต่อทีมพัฒนา
