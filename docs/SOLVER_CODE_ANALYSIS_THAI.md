# การวิเคราะห์ Flow และการทำงานของ VRP Solver v2

**ไฟล์ต้นทางหลัก:** `solvers/vrp_solver_v2.py`

## สารบัญ

1. [ภาพรวมของระบบ](#1-ภาพรวมของระบบ)
2. [การออกแบบ Data Structures](#2-การออกแบบ-data-structures)
3. [Data Loading Flow](#3-data-loading-flow)
4. [OR-Tools Solver Flow](#4-or-tools-solver-flow)
5. [Heuristic Solver Flow](#5-heuristic-solver-flow)
6. [Validation Flow](#6-validation-flow)
7. [การคำนวณต้นทุน](#7-การคำนวณต้นทุน)
8. [Checkpoint Constraint Implementation](#8-checkpoint-constraint-implementation)
9. [Output Generation](#9-output-generation)

---

## 1. ภาพรวมของระบบ

### 1.1 สถาปัตยกรรมของ VRP Solver v2

ระบบ VRP Solver v2 ถูกออกแบบด้วยแนวคิด Object-Oriented Programming (OOP) โดยมีโครงสร้างหลักดังนี้:

```
VRPSolverV2 (Main Class)
├── Data Structures (Node, Vehicle, Route, Solution)
├── Data Loading Module
├── Solver Engine
│   ├── OR-Tools Solver (Primary)
│   └── Heuristic Solver (Fallback)
├── Validation Module
└── Output Generation Module
```

### 1.2 หลักการทำงานโดยรวม

**การทำงานของระบบเริ่มต้นด้วยการโหลดข้อมูลจาก Excel** ซึ่งประกอบด้วยระยะทางระหว่างจุดต่างๆ (distance matrix), ปริมาณขยะ (demands), และข้อมูลรถ (vehicle specifications) จากนั้นระบบจะประมวลผลโดยใช้ OR-Tools เป็นวิธีหลัก และ Heuristic เป็นวิธีสำรอง สุดท้ายจะทำการตรวจสอบความถูกต้องของคำตอบ และบันทึกผลลัพธ์ออกมาในรูปแบบ JSON

### 1.3 Flow Diagram ภาพรวม

```
┌─────────────────┐
│  Initialize     │
│  VRPSolverV2    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Load Excel     │
│  Data           │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Calculate      │
│  Min Vehicles   │
└────────┬────────┘
         ↓
    ┌────┴─────┐
    │          ↓
    │   ┌──────────────┐
    │   │ Try OR-Tools │
    │   └──────┬───────┘
    │          │ Failed?
    │          ↓
    │   ┌──────────────┐
    └──→│ Use Heuristic│
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  Validate    │
        │  Solution    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  Save JSON   │
        └──────────────┘
```

---

## 2. การออกแบบ Data Structures

### 2.1 Node Data Structure

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 39-47

**Node** คือ class ที่ใช้แทนแต่ละจุดในระบบ ซึ่งอาจเป็น Depot, จุดเก็บขยะ, หรือจุดทิ้งขยะ

```python
@dataclass
class Node:
    """Represents a node in the VRP"""
    id: int  # 1-indexed node ID
    name: str
    general_demand: float
    recycle_demand: float
    is_depot: bool = False
    is_checkpoint: bool = False
```

**คำอธิบายโค้ด:**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 39 | `@dataclass` - Python decorator สร้าง class ที่เก็บข้อมูล อัตโนมัติสร้าง `__init__`, `__repr__`, `__eq__` methods |
| 40 | `class Node:` - ประกาศ class ชื่อ Node |
| 41-47 | ประกาศ attributes ของ Node: |
| 42 | `id: int` - หมายเลขระบุจุด (ใช้ 1-indexing เพื่อความสอดคล้องกับข้อมูลจริง) |
| 43 | `name: str` - ชื่อของจุด เช่น "Depot (Node 1)" หรือ "Node 15" |
| 44 | `general_demand: float` - ปริมาณขยะทั่วไปที่ต้องเก็บ |
| 45 | `recycle_demand: float` - ปริมาณขยะ recycle ที่ต้องเก็บ |
| 46 | `is_depot: bool = False` - ค่า boolean ระบุว่าเป็นจุดจอดรถหรือไม่ (เฉพาะ Node 1) |
| 47 | `is_checkpoint: bool = False` - ค่า boolean ระบุว่าเป็นจุดทิ้งขยะหรือไม่ |

**การใช้งาน:**

```python
# สร้าง Node สำหรับ Depot
depot = Node(
    id=1,
    name="Depot (Node 1)",
    general_demand=100.0,
    recycle_demand=10.0,
    is_depot=True,
    is_checkpoint=False
)
```

### 2.2 Vehicle Data Structure

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 50-57

**Vehicle** คือ class ที่ใช้แทนประเภทรถที่ใช้ในการเก็บขยะ

```python
@dataclass
class Vehicle:
    """Represents a vehicle type"""
    type_id: str
    general_capacity: float
    recycle_capacity: float
    fixed_cost: float
    fuel_cost_per_km: float  # THB per kilometer
```

**คำอธิบายโค้ด:**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 50 | `@dataclass` - สร้าง dataclass สำหรับเก็บข้อมูลประเภทรถ |
| 51 | `class Vehicle:` - ประกาศ class ชื่อ Vehicle |
| 52-57 | ประกาศ attributes ของ Vehicle: |
| 53 | `type_id: str` - รหัสประเภทรถ (เช่น "A", "B", "C") |
| 54 | `general_capacity: float` - ความจุสูงสุดสำหรับขยะทั่วไป (หน่วยเดียวกับ demand) |
| 55 | `recycle_capacity: float` - ความจุสูงสุดสำหรับขยะ recycle |
| 56 | `fixed_cost: float` - ต้นทุนคงที่ต่อคัน (บาท) |
| 57 | `fuel_cost_per_km: float` - ต้นทุนน้ำมันต่อกิโลเมตร (บาท/กม.) |

**การเลือกใช้รถ (ใน `solve` method):**

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 232

```python
# บรรทัดที่ 232: เลือกประเภทรถที่มีต้นทุนน้ำมันต่ำสุด
best_vehicle = min(vehicles, key=lambda v: v.fuel_cost_per_km)
```

| ส่วนของโค้ด | คำอธิบาย |
|--------------|-----------|
| `min(vehicles, ...)` | ฟังก์ชันหาค่าต่ำสุดจาก list ของ vehicles |
| `key=lambda v: v.fuel_cost_per_km` | ใช้ lambda function เปรียบเทียบตามค่า fuel_cost_per_km |
| ผลลัพธ์ | เลือก vehicle ที่มีต้นทุนน้ำมันต่ำสุด เพื่อลดต้นทุนรวม |

### 2.3 Route Data Structure

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 60-72

**Route** คือ class ที่ใช้แทนเส้นทางของแต่ละรถ

```python
@dataclass
class Route:
    """Represents a vehicle route"""
    vehicle_id: int
    vehicle_type: str
    nodes: List[int]  # List of node IDs (1-indexed)
    distance_meters: float
    distance_km: float
    general_load: float
    recycle_load: float
    fixed_cost: float
    fuel_cost: float
    total_cost: float
```

**คำอธิบายโค้ด:**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 60-63 | ประกาศ Route dataclass พร้อม attributes: |
| 64 | `vehicle_id: int` - หมายเลขรถ (1-indexed) |
| 65 | `vehicle_type: str` - ประเภทรถ (เช่น "A", "B", "C") |
| 66 | `nodes: List[int]` - List ของ node IDs ที่เป็นลำดับการเดินทาง (เช่น [1, 2, 5, 8, 20, 1]) |
| 67 | `distance_meters: float` - ระยะทางรวมของเส้นทาง (เมตร) |
| 68 | `distance_km: float` - ระยะทางรวมของเส้นทาง (กิโลเมตร) |
| 69 | `general_load: float` - ปริมาณขยะทั่วไปรวมที่เก็บในเส้นทางนี้ |
| 70 | `recycle_load: float` - ปริมาณขยะ recycle รวมที่เก็บในเส้นทางนี้ |
| 71 | `fixed_cost: float` - ต้นทุนคงที่ของเส้นทางนี้ |
| 72 | `fuel_cost: float` - ต้นทุนน้ำมันของเส้นทางนี้ |
| 73 (implicit) | `total_cost: float` - ต้นทุนรวม (fixed_cost + fuel_cost) |

### 2.4 Solution Data Structure

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 75-88

**Solution** คือ class ที่เก็บผลลัพธ์ทั้งหมดของการแก้ปัญหา

```python
@dataclass
class Solution:
    """Represents a VRP solution"""
    status: str
    routes: List[Route]
    num_vehicles_used: int
    total_distance_meters: float
    total_distance_km: float
    total_fixed_cost: float
    total_fuel_cost: float
    total_cost: float
    all_nodes_visited: bool
    all_routes_valid: bool
    validation_errors: List[str]
```

**คำอธิบายโค้ด:**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 75-78 | ประกาศ Solution dataclass พร้อม attributes: |
| 79 | `status: str` - สถานะของคำตอบ ("OPTIMAL", "FEASIBLE", "INFEASIBLE") |
| 80 | `routes: List[Route]` - List ของเส้นทางทั้งหมด |
| 81 | `num_vehicles_used: int` - จำนวนรถที่ใช้ |
| 82 | `total_distance_meters: float` - ระยะทางรวมทั้งหมด (เมตร) |
| 83 | `total_distance_km: float` - ระยะทางรวมทั้งหมด (กิโลเมตร) |
| 84 | `total_fixed_cost: float` - ต้นทุนคงที่รวมทั้งหมด |
| 85 | `total_fuel_cost: float` - ต้นทุนน้ำมันรวมทั้งหมด |
| 86 | `total_cost: float` - ต้นทุนรวมทั้งหมด |
| 87 | `all_nodes_visited: bool` - ระบุว่าไปครบทุก node หรือไม่ |
| 88 | `all_routes_valid: bool` - ระบุว่าทุกเส้นทางถูกต้องตามเงื่อนไขหรือไม่ |
| 89 (implicit) | `validation_errors: List[str]` - List ของข้อความ error ถ้ามี |

---

## 3. Data Loading Flow

### 3.1 การเริ่มต้นระบบ (Initialization)

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 91-108

```python
class VRPSolverV2:
    """
    VRP Solver implementing the corrected requirements.

    Route structure: Node 1 (depot, collect) → collection nodes → checkpoint → Node 1 (no collect)
    """

    def __init__(self, excel_file: str):
        self.excel_file = excel_file
        self.base_dir = Path(excel_file).parent.parent

        # Load sheet names
        wb = openpyxl.load_workbook(excel_file)
        self.sheet_names = wb.sheetnames
        wb.close()

        print(f"VRP Solver v2 initialized")
        print(f"Found sheets: {self.sheet_names}")
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 91 | `class VRPSolverV2:` - ประกาศ class หลักของระบบ |
| 92-95 | Docstring อธิบายว่า class นี้ทำหน้าที่แก้ปัญหา VRP ตาม requirements ที่กำหนด |
| 98 | `def __init__(self, excel_file: str):` - Constructor method ที่ถูกเรียกเมื่อสร้าง instance ใหม่ |
| 99 | `self.excel_file = excel_file` - เก็บ path ของ Excel file ไว้ใน instance variable |
| 100 | `self.base_dir = Path(excel_file).parent.parent` - กำหนด base directory โดยใช้ `Path()` ของ pathlib และขึ้นไป 2 ระดับ |
| 103 | `wb = openpyxl.load_workbook(excel_file)` - โหลด Excel file ด้วย openpyxl library |
| 104 | `self.sheet_names = wb.sheetnames` - ดึงชื่อ sheets ทั้งหมดจาก workbook |
| 105 | `wb.close()` - ปิด workbook เพื่อปล่อย resource |
| 107-108 | `print(...)` - แสดงข้อมูลเริ่มต้นเพื่อยืนยันว่าระบบทำงาน |

**Flow การทำงาน:**
```
เรียก VRPSolverV2(excel_file)
    ↓
เก็บ excel_file path
    ↓
คำนวณ base_dir (ขึ้น 2 ระดับจาก file)
    ↓
โหลด Excel ด้วย openpyxl
    ↓
ดึงชื่อ sheets ทั้งหมด
    ↓
ปิด workbook
    ↓
แสดงข้อมูลเริ่มต้น
```

### 3.2 การโหลดข้อมูลจาก Excel Sheet

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 110-124

```python
def load_sheet_data(self, sheet_name: str) -> Tuple[List[Node], List[Vehicle], np.ndarray, int]:
    """
    Load data from an Excel sheet.

    Returns:
        nodes: List of Node objects
        vehicles: List of Vehicle objects
        distance_matrix: NxN numpy array (in meters)
        checkpoint_id: ID of the checkpoint node (1-indexed)
    """
    print(f"\nLoading sheet: {sheet_name}")

    df = pd.read_excel(self.excel_file, sheet_name=sheet_name)

    num_nodes = int(df['Destination'].max())
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 110 | `def load_sheet_data(self, sheet_name: str) -> ...` - ประกาศ method สำหรับโหลดข้อมูลจาก sheet ที่ระบุ |
| 111-118 | Docstring อธิบายค่าที่ return: |
| 112 | `nodes: List[Node]` - List ของ Node objects |
| 113 | `vehicles: List[Vehicle]` - List ของ Vehicle objects |
| 114 | `distance_matrix: np.ndarray` - Distance matrix N×N (หน่วยเมตร) |
| 115 | `checkpoint_id: int` - ID ของ checkpoint node (1-indexed) |
| 120 | `print(f"\nLoading sheet: {sheet_name}")` - แสดงชื่อ sheet ที่กำลังโหลด |
| 122 | `df = pd.read_excel(self.excel_file, sheet_name=sheet_name)` - อ่าน Excel sheet ด้วย pandas สร้างเป็น DataFrame |
| 124 | `num_nodes = int(df['Destination'].max())` - หาค่าสูงสุดของคอลัมน์ 'Destination' เพื่อกำหนดจำนวน nodes |

**Flow การทำงาน:**
```
เรียก load_sheet_data(sheet_name)
    ↓
แสดงชื่อ sheet
    ↓
อ่าน Excel sheet ด้วย pandas
    ↓
หาค่าสูงสุดของ Destination
    ↓
กำหนดจำนวน nodes
```

### 3.3 การสร้าง Distance Matrix

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 131-149

```python
# Build distance matrix (in meters)
distance_matrix = np.zeros((num_nodes, num_nodes))

for idx, row in df.iterrows():
    if pd.isna(row['Destination']):
        continue

    node_id = int(row['Destination'])
    i = node_id - 1  # 0-indexed

    # Extract distances
    for j in range(1, num_nodes + 1):
        origin_col = f'Origin_{j}'
        if origin_col in df.columns:
            dist = row[origin_col]
            if pd.notna(dist):
                j_idx = j - 1
                distance_matrix[i][j_idx] = float(dist)
                distance_matrix[j_idx][i] = float(dist)  # Symmetric
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 132 | `distance_matrix = np.zeros((num_nodes, num_nodes))` - สร้าง NumPy array ขนาด N×N เต็มไปด้วยศูนย์ |
| 134 | `for idx, row in df.iterrows():` - วนลูปผ่านทุกแถวใน DataFrame ได้ (index, row) |
| 135-136 | `if pd.isna(row['Destination']): continue` - ถ้า Destination เป็น NaN ให้ข้ามแถวนี้ |
| 138 | `node_id = int(row['Destination'])` - แปลงค่า Destination เป็น integer (1-indexed) |
| 139 | `i = node_id - 1` - แปลงเป็น 0-indexed สำหรับใช้กับ NumPy array |
| 142 | `for j in range(1, num_nodes + 1):` - วนลูป j จาก 1 ถึง num_nodes |
| 143 | `origin_col = f'Origin_{j}'` - สร้างชื่อคอลัมน์ เช่น 'Origin_1', 'Origin_2' |
| 144 | `if origin_col in df.columns:` - ตรวจสอบว่ามีคอลัมน์นี้ใน DataFrame หรือไม่ |
| 145 | `dist = row[origin_col]` - อ่านระยะทางจากคอลัมน์ Origin_j |
| 146 | `if pd.notna(dist):` - ถ้าระยะทางไม่เป็น NaN ให้ประมวลผล |
| 147 | `j_idx = j - 1` - แปลง j เป็น 0-indexed |
| 148 | `distance_matrix[i][j_idx] = float(dist)` - เติมระยะทางใน matrix ที่ตำแหน่ง [i][j_idx] |
| 149 | `distance_matrix[j_idx][i] = float(dist)` - เติมระยะทางในตำแหน่งตรงข้าม (สมมาตร) |

**ตัวอย่าง Distance Matrix:**
```
      Node1  Node2  Node3  Node4
Node1   0    100    250    180
Node2  100     0    300    220
Node3  250   300      0    150
Node4  180   220    150      0
```

**Flow การทำงาน:**
```
สร้าง zero matrix N×N
    ↓
วนลูปผ่านทุกแถวใน DataFrame
    ↓
แปลง node_id เป็น 0-indexed (i)
    ↓
วนลูป j จาก 1 ถึง num_nodes
    ↓
อ่านระยะทางจากคอลัมน์ Origin_j
    ↓
เติมระยะทางลง matrix[i][j] และ matrix[j][i]
```

**ตัวอย่าง Distance Matrix:**

```
      Node1  Node2  Node3  Node4
Node1   0    100    250    180
Node2  100     0    300    220
Node3  250   300      0    150
Node4  180   220    150      0
```

### 3.4 การสร้าง Node Objects

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 151-183

```python
# Extract node information
general_str = str(row.get('ขยะทั่วไป', 0)).strip()
recycle_str = str(row.get('ขยะ recycle', 0)).strip()

# Check if this is the checkpoint node
is_checkpoint = (general_str == 'จุดทิ้ง')
if is_checkpoint:
    checkpoint_id = node_id
    general_demand = 0.0
    recycle_demand = 0.0
else:
    try:
        general_demand = float(general_str) if general_str not in ['nan', ''] else 0.0
    except ValueError:
        general_demand = 0.0

    try:
        recycle_demand = float(recycle_str) if recycle_str not in ['nan', '', 'จุดทิ้ง'] else 0.0
    except ValueError:
        recycle_demand = 0.0

# Node 1 is always the depot
is_depot = (node_id == 1)

node = Node(
    id=node_id,
    name=f"Node {node_id}" if not is_depot else "Depot (Node 1)",
    general_demand=general_demand,
    recycle_demand=recycle_demand,
    is_depot=is_depot,
    is_checkpoint=is_checkpoint
)
nodes.append(node)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 152 | `general_str = str(row.get('ขยะทั่วไป', 0)).strip()` - อ่านค่าจากคอลัมน์ 'ขยะทั่วไป' แปลงเป็น string และลบ spaces |
| 153 | `recycle_str = str(row.get('ขยะ recycle', 0)).strip()` - อ่านค่าจากคอลัมน์ 'ขยะ recycle' แปลงเป็น string และลบ spaces |
| 156 | `is_checkpoint = (general_str == 'จุดทิ้ง')` - ตรวจสอบว่าเป็น checkpoint node หรือไม่ |
| 157-160 | `if is_checkpoint:` - ถ้าเป็น checkpoint: กำหนด checkpoint_id และ demand = 0 |
| 162-165 | `try: ... except ValueError:` - พยายามแปลง general_demand เป็น float หรือ 0.0 ถ้าแปลงไม่ได้ |
| 167-170 | `try: ... except ValueError:` - พยายามแปลง recycle_demand เป็น float หรือ 0.0 ถ้าแปลงไม่ได้ |
| 173 | `is_depot = (node_id == 1)` - กำหนดว่า Node 1 เป็น depot เสมอ |
| 175-182 | `node = Node(...)` - สร้าง Node object ด้วยข้อมูลทั้งหมด |
| 183 | `nodes.append(node)` - เพิ่ม node เข้าไปใน list |

**Flow การทำงาน:**
```
อ่านค่า general_demand และ recycle_demand จาก Excel
    ↓
ตรวจสอบว่าเป็น checkpoint ('จุดทิ้ง') หรือไม่
    ↓
ถ้าเป็น checkpoint → demand = 0
    ↓
ถ้าไม่ใช่ checkpoint → แปลงค่าเป็น float (หรือ 0.0 ถ้า error)
    ↓
ตรวจสอบว่าเป็น depot (Node 1) หรือไม่
    ↓
สร้าง Node object
    ↓
เพิ่ม node เข้าไปใน list
```

### 3.5 การสร้าง Vehicle Objects

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 185-195

```python
# Extract vehicle information
v_type = str(row.get('รถคันที่', '')).strip()
if pd.notna(v_type) and v_type and v_type != 'nan':
    if v_type not in vehicles_dict:
        vehicles_dict[v_type] = Vehicle(
            type_id=v_type,
            general_capacity=float(row.get('cap for gereral ', row.get('cap for general', 2000))),
            recycle_capacity=float(row.get('cap for recycle', 200)),
            fixed_cost=float(row.get('fix cost', 2400)),
            fuel_cost_per_km=float(row.get('variable cost', 8))  # THB per km
        )
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 186 | `v_type = str(row.get('รถคันที่', '')).strip()` - อ่านค่าจากคอลัมน์ 'รถคันที่' แปลงเป็น string และลบ spaces |
| 187 | `if pd.notna(v_type) and v_type and v_type != 'nan':` - ตรวจสอบว่า v_type มีค่าและไม่ใช่ 'nan' |
| 188 | `if v_type not in vehicles_dict:` - ตรวจสอบว่ายังไม่มี vehicle ประเภทนี้ใน dictionary (ป้องกันการสร้างซ้ำ) |
| 189-195 | `vehicles_dict[v_type] = Vehicle(...)` - สร้าง Vehicle object ใหม่: |
| 190 | `type_id=v_type` - รหัสประเภทรถ |
| 191 | `general_capacity=float(...)` - ความจุขยะทั่วไป (มี fallback 2 ชั้นเผื่อ typo) |
| 192 | `recycle_capacity=float(...)` - ความจุขยะ recycle (default 200) |
| 193 | `fixed_cost=float(...)` - ต้นทุนคงที่ (default 2400 บาท) |
| 194 | `fuel_cost_per_km=float(...)` - ต้นทุนน้ำมันต่อกม. (default 8 บาท/กม.) |

**Flow การทำงาน:**
```
อ่านค่าประเภทรถจาก Excel
    ↓
ตรวจสอบว่าค่าถูกต้อง (ไม่ใช่ NaN หรือค่าว่าง)
    ↓
ตรวจสอบว่ายังไม่เคยสร้าง vehicle ประเภทนี้
    ↓
สร้าง Vehicle object พร้อม capacity และ costs
    ↓
เก็บไว้ใน vehicles_dict
```

---

## 4. OR-Tools Solver Flow

### 4.1 การเริ่มต้น OR-Tools Solver (solve method)

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 213-232

```python
def solve(self, sheet_name: str, time_limit: int = 60) -> Solution:
    """
    Solve VRP for a specific sheet.

    Args:
        sheet_name: Name of the Excel sheet
        time_limit: Solver time limit in seconds

    Returns:
        Solution object
    """
    # Load data
    nodes, vehicles, distance_matrix, checkpoint_id = self.load_sheet_data(sheet_name)

    num_nodes = len(nodes)
    depot_idx = 0  # Node 1 is always depot (0-indexed)
    checkpoint_idx = checkpoint_id - 1  # Convert to 0-indexed

    # Use the best vehicle type (lowest fuel cost)
    best_vehicle = min(vehicles, key=lambda v: v.fuel_cost_per_km)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 213 | `def solve(self, sheet_name: str, time_limit: int = 60) -> Solution:` - ประกาศ method หลักสำหรับแก้ปัญหา |
| 214-222 | Docstring อธิบาย parameters และ return value |
| 225 | `nodes, vehicles, distance_matrix, checkpoint_id = self.load_sheet_data(sheet_name)` - เรียก method โหลดข้อมูลจาก Excel |
| 227 | `num_nodes = len(nodes)` - นับจำนวน nodes ทั้งหมด |
| 228 | `depot_idx = 0` - กำหนด index ของ depot (Node 1 = index 0 เพราะ 0-indexed) |
| 229 | `checkpoint_idx = checkpoint_id - 1` - แปลง checkpoint_id เป็น 0-indexed |
| 232 | `best_vehicle = min(vehicles, key=lambda v: v.fuel_cost_per_km)` - เลือกประเภทรถที่มีต้นทุนน้ำมันต่ำสุด |

**Flow การทำงาน:**
```
เรียก solve(sheet_name, time_limit)
    ↓
โหลดข้อมูลจาก Excel (load_sheet_data)
    ↓
แปลง checkpoint_id เป็น 0-indexed
    ↓
เลือกประเภทรถที่มีต้นทุนน้ำมันต่ำสุด
    ↓
คำนวณจำนวนรถขั้นต่ำที่ต้องใช้
    ↓
เรียก OR-Tools solver หรือ Heuristic solver
    ↓
ตรวจสอบความถูกต้องของคำตอบ
    ↓
คืนค่า Solution object
```

### 4.2 การคำนวณจำนวนรถขั้นต่ำ

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 234-247

```python
# Calculate total demand (including depot's demand which is collected at start)
total_general = sum(n.general_demand for n in nodes)
total_recycle = sum(n.recycle_demand for n in nodes)

print(f"\nSolving sheet: {sheet_name}")
print(f"  Total demand: {total_general} general, {total_recycle} recycle")
print(f"  Vehicle capacity: {best_vehicle.general_capacity} general, {best_vehicle.recycle_capacity} recycle")

# Calculate minimum vehicles needed
min_veh_gen = math.ceil(total_general / best_vehicle.general_capacity) if best_vehicle.general_capacity > 0 else 1
min_veh_rec = math.ceil(total_recycle / best_vehicle.recycle_capacity) if best_vehicle.recycle_capacity > 0 else 1
num_vehicles = max(min_veh_gen, min_veh_rec, 1)

print(f"  Minimum vehicles needed: {num_vehicles}")
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 235 | `total_general = sum(n.general_demand for n in nodes)` - รวม demand ขยะทั่วไปทั้งหมด (generator expression) |
| 236 | `total_recycle = sum(n.recycle_demand for n in nodes)` - รวม demand ขยะ recycle ทั้งหมด |
| 238-240 | `print(...)` - แสดงข้อมูล demand ทั้งหมด |
| 243 | `min_veh_gen = math.ceil(...)` - คำนวณจำนวนรถขั้นต่ำสำหรับขยะทั่วไป (ใช้ ceiling) |
| 244 | `min_veh_rec = math.ceil(...)` - คำนวณจำนวนรถขั้นต่ำสำหรับขยะ recycle |
| 245 | `num_vehicles = max(min_veh_gen, min_veh_rec, 1)` - เลือกค่าสูงสุดเพื่อให้มีรถเพียงพอทั้งสองประเภท |
| 247 | `print(f"  Minimum vehicles needed: {num_vehicles}")` - แสดงจำนวนรถขั้นต่ำที่ต้องใช้ |

**ตัวอย่างการคำนวณ:**
```
Total general demand: 2,849 units
Vehicle general capacity: 2,000 units
min_veh_gen = ceil(2849 / 2000) = ceil(1.4245) = 2 vehicles

Total recycle demand: 174 units
Vehicle recycle capacity: 200 units
min_veh_rec = ceil(174 / 200) = ceil(0.87) = 1 vehicle

num_vehicles = max(2, 1, 1) = 2 vehicles
```

**ตัวอย่างการคำนวณ:**
```
Total general demand: 2,849 units
Vehicle general capacity: 2,000 units
min_veh_gen = ceil(2849 / 2000) = ceil(1.4245) = 2 vehicles

Total recycle demand: 174 units
Vehicle recycle capacity: 200 units
min_veh_rec = ceil(174 / 200) = ceil(0.87) = 1 vehicle

num_vehicles = max(2, 1, 1) = 2 vehicles
```

### 4.3 การสร้าง OR-Tools Model

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 274-291

```python
def _solve_ortools(self, nodes: List[Node], vehicle: Vehicle,
                   distance_matrix: np.ndarray, depot_idx: int,
                   checkpoint_idx: int, num_vehicles: int,
                   time_limit: int) -> Solution:
    """
    Solve using OR-Tools with the checkpoint natively integrated into optimization.

    The checkpoint constraint mirrors the AMPL formulation:
        x[checkpoint, depot, k] = vehicle_used[k]

    Implementation:
    - A BIG_PENALTY is added to any arc going directly from a non-checkpoint node
      to the depot, making it far cheaper for OR-Tools to always route via the
      checkpoint last.
    - For multi-vehicle problems, phantom checkpoint copies are added to the
      distance matrix so every active vehicle visits the checkpoint before depot.
    """
    num_nodes = len(nodes)
    BIG_PENALTY = 10_000_000

    # For multi-vehicle, expand distance matrix with (num_vehicles-1) checkpoint copies
    total_nodes = num_nodes + (num_vehicles - 1)
    checkpoint_copies = [checkpoint_idx] + list(range(num_nodes, total_nodes))

    manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 274-277 | `def _solve_ortools(self, ...)` - ประกาศ method สำหรับใช้ OR-Tools solver (private method) |
| 278-287 | Docstring อธิบาย approach ใหม่ที่ integrate checkpoint เข้า optimization โดยตรง |
| 288 | `BIG_PENALTY = 10_000_000` - ค่า penalty ขนาดใหญ่ (10 ล้าน เมตร) ใหญ่กว่าระยะทางจริงใด ๆ |
| 290-292 | ขยาย distance matrix สำหรับ checkpoint copies (กรณี multi-vehicle) |
| 294 | `manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, depot_idx)` - สร้าง index manager |
| 295 | `routing = pywrapcp.RoutingModel(manager)` - สร้าง routing optimization model |

**คำอธิบาย RoutingIndexManager:**
```
RoutingIndexManager ทำหน้าที่แปลงระหว่าง:
- User indices: Node IDs (1, 2, 3, ...) ที่ user เห็น
- Internal indices: Indices ภายในของ OR-Tools solver

Parameters:
- total_nodes: จำนวน nodes ทั้งหมด (รวม checkpoint copies)
- num_vehicles: จำนวนรถที่ใช้
- depot_idx: index ของ depot (เสมอ 0 สำหรับ Node 1)

ตัวอย่างการแปลง (20 nodes, 1 vehicle):
User Node 1 (depot) → Internal Index 0
User Node 5 → Internal Index 4
User Node 20 (checkpoint) → Internal Index 19
```

### 4.4 Distance Callback พร้อม BIG_PENALTY

**ไฟล์:** `solvers/vrp_solver_v2.py`

```python
# Distance callback: penalize any non-checkpoint → depot arc
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    dist = int(expanded_dist[from_node][to_node])
    if to_node == depot_idx and from_node not in checkpoint_set:
        dist += BIG_PENALTY
    return dist

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| `def distance_callback(from_index, to_index):` | ประกาศ callback function ที่ OR-Tools เรียกเพื่อทราบระยะทาง |
| `from_node = manager.IndexToNode(from_index)` | แปลง from_index (internal) → user node (0-indexed) |
| `to_node = manager.IndexToNode(to_index)` | แปลง to_index (internal) → user node (0-indexed) |
| `dist = int(expanded_dist[from_node][to_node])` | ดึงระยะทางจาก expanded distance matrix |
| `if to_node == depot_idx and from_node not in checkpoint_set:` | ตรวจสอบว่า arc นี้เป็น non-checkpoint → depot หรือไม่ |
| `dist += BIG_PENALTY` | ถ้าใช่ บวก penalty 10 ล้าน เมตร เพื่อบังคับให้ผ่าน checkpoint ก่อน |
| `routing.RegisterTransitCallback(...)` | ลงทะเบียน callback function กับ solver |
| `routing.SetArcCostEvaluatorOfAllVehicles(...)` | กำหนดให้ใช้ callback นี้สำหรับทุกคัน |

**หลักการ BIG_PENALTY:**
```
ปัญหา: จะบังคับให้รถผ่าน checkpoint ก่อนกลับ depot ได้อย่างไร?

วิธีแก้: ปรับ cost ของ arcs ที่ไม่ผ่าน checkpoint ให้แพงมาก
├─ arc: collection_node → depot  →  cost = actual_dist + 10,000,000
└─ arc: checkpoint → depot       →  cost = actual_dist (ปกติ)

ผลลัพธ์: OR-Tools จะเลือก route ที่ผ่าน checkpoint เสมอ
เพราะถูกกว่ามาก (หลีก BIG_PENALTY)

ตัวอย่าง:
Route ที่ไม่ดี: ... → Node 14 → Depot     cost = 500 + 10,000,000
Route ที่ดี:    ... → Node 19 → Checkpoint → Depot  cost = 200 + 300 = 500
```

**Flow การเรียกใช้ Callback:**
```
Solver ต้องการ cost ของ arc A → B
    ↓
เรียก distance_callback(A_index, B_index)
    ↓
แปลง indices → user nodes
    ↓
ดึงระยะทางจาก expanded_dist[A][B]
    ↓
ถ้า B = depot และ A ≠ checkpoint → บวก BIG_PENALTY
    ↓
คืนค่าให้ Solver
    ↓
Solver ใช้ค่านี้เพื่อหาเส้นทางที่ cost ต่ำสุด
```

### 4.5 Capacity Constraints

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 302-334

```python
# Demand callbacks for capacity constraints
# Note: Depot (Node 1) demand is collected at start, so include it
# Checkpoint has zero demand
general_demands = [int(n.general_demand) for n in nodes]
recycle_demands = [int(n.recycle_demand) for n in nodes]

def demand_general_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return general_demands[from_node]

def demand_recycle_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return recycle_demands[from_node]

# Add general capacity dimension
demand_gen_callback_index = routing.RegisterUnaryTransitCallback(demand_general_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_gen_callback_index,
    0,  # no slack
    [int(vehicle.general_capacity)] * num_vehicles,
    True,  # start cumul to zero
    'GeneralCapacity'
)

# Add recycle capacity dimension
demand_rec_callback_index = routing.RegisterUnaryTransitCallback(demand_recycle_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_rec_callback_index,
    0,
    [int(vehicle.recycle_capacity)] * num_vehicles,
    True,
    'RecycleCapacity'
)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 305 | `general_demands = [int(n.general_demand) for n in nodes]` - สร้าง list ของ general demands จากทุก node |
| 306 | `recycle_demands = [int(n.recycle_demand) for n in nodes]` - สร้าง list ของ recycle demands |
| 308-310 | `def demand_general_callback(from_index):` - callback คืนค่า general demand ของ node |
| 309 | `from_node = manager.IndexToNode(from_index)` - แปลง internal index → user node |
| 310 | `return general_demands[from_node]` - คืนค่า general demand |
| 312-314 | `def demand_recycle_callback(from_index):` - callback คืนค่า recycle demand (โครงสร้างเหมือนกัน) |
| 317 | `demand_gen_callback_index = routing.RegisterUnaryTransitCallback(...)` - ลงทะเบียน callback สำหรับ general demand |
| 318-324 | `routing.AddDimensionWithVehicleCapacity(...)` - เพิ่ม dimension สำหรับ general capacity constraints: |
| 319 | Callback index สำหรับ general demand |
| 320 | `0` - Slack (ไม่อนุญาตให้มีค่าสะสมติดลบ) |
| 321 | `[int(vehicle.general_capacity)] * num_vehicles` - List ของ capacities สำหรับทุกคัน |
| 322 | `True` - Start cumul to zero (เริ่มนับจาก 0) |
| 323 | `'GeneralCapacity'` - ชื่อ dimension |
| 327-334 | ทำแบบเดียวกันสำหรับ Recycle Capacity dimension |

**หมายเหตุสำคัญ**:
- Node 1 (Depot) มี demand ที่ถูกเก็บที่จุดเริ่มต้น ดังนั้นจึงถูกรวมในการคำนวณ
- Checkpoint มี demand = 0 เพราะไม่มีการเก็บขยะที่นั่น

### 4.6 Checkpoint Assignment ต่อ Vehicle

**ไฟล์:** `solvers/vrp_solver_v2.py`

```python
# Assign each checkpoint copy to exactly one vehicle.
# High penalty ensures every active vehicle visits its checkpoint.
for v, ckpt_node in enumerate(checkpoint_copies):
    ckpt_routing_idx = manager.NodeToIndex(ckpt_node)
    routing.AddDisjunction([ckpt_routing_idx], BIG_PENALTY)
    routing.SetAllowedVehiclesForIndex([v], ckpt_routing_idx)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| `for v, ckpt_node in enumerate(checkpoint_copies):` | วนลูปผ่าน checkpoint node ทุกตัว (1 ตัวต่อ 1 vehicle) |
| `ckpt_routing_idx = manager.NodeToIndex(ckpt_node)` | แปลง checkpoint node index → internal OR-Tools index |
| `routing.AddDisjunction([ckpt_routing_idx], BIG_PENALTY)` | กำหนด penalty สูงมากถ้าไม่ไปเยี่ยม (บังคับให้ไป) |
| `routing.SetAllowedVehiclesForIndex([v], ckpt_routing_idx)` | กำหนดให้เฉพาะรถ `v` เท่านั้นที่ไปเยี่ยม checkpoint copy นี้ได้ |

**เปรียบเทียบวิธีเก่า vs วิธีใหม่:**
```
วิธีเก่า (Post-Processing Hack):
├─ AddDisjunction([checkpoint], penalty=0)  ← ไม่มี penalty → OR-Tools ข้ามเสมอ
├─ OR-Tools แก้ปัญหาโดยไม่สนใจ checkpoint
└─ หลังได้คำตอบ: ยัด checkpoint ต่อท้ายทุก route โดยไม่สนว่าไกลแค่ไหน

วิธีใหม่ (Native Constraint):
├─ AddDisjunction([checkpoint], penalty=BIG_PENALTY)  ← penalty สูงมาก → OR-Tools ต้องไป
├─ distance_callback: arc ที่ไม่ผ่าน checkpoint → depot มี cost สูงมาก
├─ SetAllowedVehiclesForIndex: กำหนดว่ารถใดต้องไป checkpoint copy ใด
└─ OR-Tools วางแผนเส้นทางทั้งหมด "รู้" ว่าต้องผ่าน checkpoint ก่อนกลับ depot

ผลลัพธ์:
├─ วิธีเก่า (20 nodes): 5.298 km, 2,442.38 THB
└─ วิธีใหม่ (20 nodes): 4.162 km, 2,433.30 THB  ✓ ตรงกับ AMPL optimal
```

**การจัดการ Multi-Vehicle (checkpoint copies):**
```
ปัญหา: OR-Tools บังคับว่าแต่ละ mandatory node ถูกเยี่ยมโดยรถเพียงคันเดียว
แต่ทุกรถต้องไป checkpoint

วิธีแก้: สร้าง "phantom checkpoint copies"
├─ num_vehicles = 2
├─ checkpoint_copies = [checkpoint_idx, num_nodes]  (2 copies)
├─ Copy 0 (index 19): เฉพาะรถ 0, ตำแหน่งเดียวกับ checkpoint
└─ Copy 1 (index 20): เฉพาะรถ 1, ตำแหน่งเดียวกับ checkpoint

ผล: รถทุกคันมี "checkpoint ของตัวเอง" → ทั้งคู่ต้องผ่านก่อนกลับ depot
```

### 4.7 Search Parameters

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 341-352

```python
# Search parameters
search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_params.time_limit.seconds = time_limit

# Solve
assignment = routing.SolveWithParameters(search_params)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 342 | `search_params = pywrapcp.DefaultRoutingSearchParameters()` - สร้าง parameters object พร้อมค่าเริ่มต้น |
| 343-344 | `search_params.first_solution_strategy = ...` - กำหนด strategy สำหรับสร้างคำตอบเริ่มต้น |
| 345 | `PATH_CHEAPEST_ARC` - เริ่มจาก depot เลือก node ถัดไปที่มีระยะทางน้อยที่สุด ทำซ้ำจนครบ |
| 346-347 | `search_params.local_search_metaheuristic = ...` - กำหนด metaheuristic สำหรับปรับปรุงคำตอบ |
| 348 | `GUIDED_LOCAL_SEARCH` - ใช้ local search พร้อม penalties เพื่อหลีกเลี่ยง local optima |
| 349 | `search_params.time_limit.seconds = time_limit` - จำกัดเวลาการแก้ปัญหา (default 60 วินาที) |
| 352 | `assignment = routing.SolveWithParameters(search_params)` - เริ่มการแก้ปัญหาและคืนค่า assignment object |

**คำอธิบาย Search Strategies:**
```
PATH_CHEAPEST_ARC:
├─ เริ่มจาก depot
├─ เลือก node ถัดไปที่มีระยะทางน้อยที่สุด
└─ ทำซ้ำจนครบทุก node

GUIDED_LOCAL_SEARCH:
├─ เริ่มจากคำตอบเริ่มต้น (จาก PATH_CHEAPEST_ARC)
├─ ปรับปรุงด้วย local search
├─ ใช้ penalties เพื่อหลีกเลี่ยง local optima
└─ หาคำตอบที่ดีกว่าเดิม
```

### 4.8 การดึงคำตอบ

```python
if not assignment:
    raise Exception("OR-Tools could not find a solution")

routes = []
total_distance_m = 0
checkpoint_node_id = nodes[checkpoint_idx].id  # 1-indexed

for vehicle_id in range(num_vehicles):
    index = routing.Start(vehicle_id)
    route_nodes = []
    general_load = 0.0
    recycle_load = 0.0

    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        if node_idx < num_nodes:
            node = nodes[node_idx]
            route_nodes.append(node.id)
            general_load += node.general_demand
            recycle_load += node.recycle_demand
        else:
            # Phantom checkpoint copy — map to real checkpoint node ID
            route_nodes.append(checkpoint_node_id)
        index = assignment.Value(routing.NextVar(index))

    route_nodes.append(1)  # End at depot

    # Skip trivial routes (only depot + checkpoint + depot, no collections)
    collection_nodes_in_route = [n for n in route_nodes[1:-1] if n != checkpoint_node_id]
    if not collection_nodes_in_route:
        continue
```

**คำอธิบายโค้ด:**

**Solution Extraction** คือการดึงเส้นทางจาก assignment:

1. **ตรวจสอบ Solution**: ถ้า `assignment` เป็น None แสดงว่าไม่พบคำตอบ

2. **วนลูปผ่าน Vehicles**: สำหรับแต่ละ vehicle_id จาก 0 ถึง num_vehicles-1

3. **เริ่มจาก Start Node**: `routing.Start(vehicle_id)` คืนค่า index ของจุดเริ่มต้น

4. **วนลูปผ่าน Path**: `while not routing.IsEnd(index)` วนจนถึงจุดสิ้นสุด (depot)

5. **แยก Node จริง vs Checkpoint Copy**:
   - `node_idx < num_nodes` → เป็น node จริง ดึงข้อมูลจาก `nodes[node_idx]`
   - `node_idx >= num_nodes` → เป็น phantom checkpoint copy → map กลับไปเป็น `checkpoint_node_id`

6. **ไม่มีการ skip checkpoint ระหว่าง extraction** เพราะ OR-Tools จัดตำแหน่งให้ถูกต้องแล้วผ่าน BIG_PENALTY

7. **คำนวณ actual distance**: ใช้ `distance_matrix` จริง (ไม่มี penalty) เพื่อให้ได้ระยะทางที่ถูกต้อง

---

## 5. Heuristic Solver Flow

### 5.1 ภาพรวมของ Heuristic Solver

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 430-447

```python
def _solve_heuristic(self, nodes: List[Node], vehicle: Vehicle,
                     distance_matrix: np.ndarray, depot_idx: int,
                     checkpoint_idx: int) -> Solution:
    """
    Heuristic solver using nearest neighbor with mandatory checkpoint.

    For each route:
    1. Start at depot (Node 1), collect its trash
    2. Visit nearest feasible nodes
    3. When capacity is nearly full or no more nodes, go to checkpoint
    4. Return to depot
    """
    num_nodes = len(nodes)

    # Nodes to visit (excluding depot and checkpoint)
    unvisited = set(range(num_nodes))
    unvisited.discard(depot_idx)  # Don't revisit depot in middle
    unvisited.discard(checkpoint_idx)  # Checkpoint is visited at end of each route
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 430-434 | `def _solve_heuristic(self, ...)` - ประกาศ fallback solver method (ใช้เมื่อ OR-Tools ล้มเหลว) |
| 435-440 | Docstring อธิบาย approach: |
| 436 | 1. เริ่มที่ depot (Node 1) เก็บขยะที่นั่น |
| 437 | 2. เยี่ยม nodes ที่ใกล้ที่สุดที่เป็นไปได้ |
| 438 | 3. เมื่อ capacity เต็ม หรือไม่มี nodes แล้ว ไป checkpoint |
| 439 | 4. กลับ depot |
| 442 | `num_nodes = len(nodes)` - นับจำนวน nodes ทั้งหมด |
| 445 | `unvisited = set(range(num_nodes))` - สร้าง set ของ indices ทั้งหมด (0, 1, 2, ..., n-1) |
| 446 | `unvisited.discard(depot_idx)` - ลบ depot ออกจาก set (ไม่ต้องเยี่ยมซ้ำ) |
| 447 | `unvisited.discard(checkpoint_idx)` - ลบ checkpoint ออกจาก set (จะเยี่ยมท้ายเส้นทาง) |

**หลักการทำงาน:**
```
ใช้เมื่อ: OR-Tools ล้มเหลว หรือไม่มี OR-Tools
Algorithm: Nearest Neighbor Heuristic
├─ เลือก node ที่ใกล้ที่สุดที่เป็นไปได้ (feasible)
├─ ตรวจสอบ capacity constraints
├─ เมื่อเต็ม หรือไม่มี nodes แล้ว ไป checkpoint แล้วกลับ depot
└─ ทำซ้ำจนครบทุก node
```

### 5.2 การสร้างเส้นทาง

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 449-462

```python
routes = []
vehicle_id = 0

while unvisited:
    vehicle_id += 1

    # Start new route at depot
    current = depot_idx
    route_nodes = [nodes[depot_idx].id]  # Start with Node 1
    route_distance = 0.0

    # Collect depot's trash at start
    general_load = nodes[depot_idx].general_demand
    recycle_load = nodes[depot_idx].recycle_demand
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 449 | `routes = []` - สร้าง empty list สำหรับเก็บ routes |
| 450 | `vehicle_id = 0` - เริ่มนับ vehicle_id จาก 0 |
| 452 | `while unvisited:` - วนลูปจนกว่า set จะว่าง (ไม่มี nodes ที่ยังไม่ได้เยี่ยม) |
| 453 | `vehicle_id += 1` - เพิ่ม vehicle_id สำหรับ route ใหม่ |
| 456 | `current = depot_idx` - เริ่มเส้นทางที่ depot |
| 457 | `route_nodes = [nodes[depot_idx].id]` - สร้าง list เริ่มต้นด้วย Node ID ของ depot (Node 1) |
| 458 | `route_distance = 0.0` - เริ่มระยะทางรวมที่ 0 |
| 461 | `general_load = nodes[depot_idx].general_demand` - เริ่มสะสม general load จาก depot |
| 462 | `recycle_load = nodes[depot_idx].recycle_demand` - เริ่มสะสม recycle load จาก depot |

**Flow การสร้างเส้นทาง:**
```
while unvisited (ยังมี nodes ที่ไม่ได้เยี่ยม)
    ↓
เริ่ม route ใหม่
    ↓
ตั้งค่าเริ่มต้น: current = depot, loads = depot demands
    ↓
วนลูปเลือก nodes ด้วย Nearest Neighbor
    ↓
เมื่อเต็ม หรือไม่มี nodes แล้ว → ไป checkpoint → กลับ depot
    ↓
เพิ่ม route เข้า list
    ↓
ทำซ้ำจนกว่าจะเยี่ยมครบทุก node
```

### 5.3 การเลือก Node ถัดไป (Nearest Neighbor)

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 464-484

```python
    # Visit nodes until capacity is reached
    while True:
        best_node = None
        best_dist = float('inf')

        for node_idx in unvisited:
            node = nodes[node_idx]

            # Check capacity constraints
            if (general_load + node.general_demand > vehicle.general_capacity or
                recycle_load + node.recycle_demand > vehicle.recycle_capacity):
                continue

            # Find nearest feasible node
            dist = distance_matrix[current][node_idx]
            if dist < best_dist:
                best_dist = dist
                best_node = node_idx

        if best_node is None:
            break
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 465 | `while True:` - วนลูปไม่สิ้นสุด (จะ break เมื่อไม่มี nodes ที่เป็นไปได้) |
| 466 | `best_node = None` - กำหนดค่าเริ่มต้นสำหรับ node ที่ดีที่สุด |
| 467 | `best_dist = float('inf')` - กำหนดระยะทางเริ่มต้นเป็นอนันต์ |
| 469 | `for node_idx in unvisited:` - วนลูปผ่านทุก node ใน unvisited set |
| 470 | `node = nodes[node_idx]` - ดึง Node object จาก index |
| 473-475 | `if (general_load + node.general_demand > ...)` - ตรวจสอบ capacity constraints |
| 476 | `continue` - ข้าม node นี้ ถ้าเกินความจุ |
| 479 | `dist = distance_matrix[current][node_idx]` - ดึงระยะทางจาก current node ไป node นี้ |
| 480-481 | `if dist < best_dist:` - ถ้าระยะทางน้อยกว่าค่าที่ดีที่สุด ให้ update |
| 482 | `best_dist = dist` - อัปเดตระยะทางที่ดีที่สุด |
| 483 | `best_node = node_idx` - อัปเดต node ที่ดีที่สุด |
| 485 | `if best_node is None:` - ถ้าไม่มี node ที่เป็นไปได้ |
| 486 | `break` - หยุดการเยี่ยม nodes ใน route นี้ |

**ตรรกะการเลือก Node:**
```
สำหรับทุก node ใน unvisited:
    ↓
    ตรวจสอบ capacity constraints
    ↓
    ถ้าเกิน capacity → ข้าม (continue)
    ↓
    ถ้าไม่เกิน → เปรียบเทียบระยะทาง
    ↓
    ถ้าระยะทางน้อยกว่าค่าที่ดีที่สุด → update
    ↓
เลือก node ที่มีระยะทางน้อยที่สุด
    ↓
ถ้าไม่มี node ที่เป็นไปได้ → หยุด (break)
```

### 5.4 การเดินทางไป Checkpoint และกลับ Depot

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 487-501

```python
    # Visit best node
    route_nodes.append(nodes[best_node].id)
    route_distance += best_dist
    general_load += nodes[best_node].general_demand
    recycle_load += nodes[best_node].recycle_demand
    unvisited.remove(best_node)
    current = best_node

# Go to checkpoint before returning to depot
route_nodes.append(nodes[checkpoint_idx].id)
route_distance += distance_matrix[current][checkpoint_idx]
current = checkpoint_idx

# Return to depot
route_nodes.append(nodes[depot_idx].id)
route_distance += distance_matrix[checkpoint_idx][depot_idx]
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 488 | `route_nodes.append(nodes[best_node].id)` - เพิ่ม node ID ของ node ที่เลือกลงใน route |
| 489 | `route_distance += best_dist` - เพิ่มระยะทางจาก current node ไป node ที่เลือก |
| 490 | `general_load += nodes[best_node].general_demand` - เพิ่ม general load |
| 491 | `recycle_load += nodes[best_node].recycle_demand` - เพิ่ม recycle load |
| 492 | `unvisited.remove(best_node)` - ลบ node ออกจาก unvisited set |
| 493 | `current = best_node` - ย้าย current position ไป node ใหม่ |
| 496 | `route_nodes.append(nodes[checkpoint_idx].id)` - เพิ่ม checkpoint node ID ลงใน route |
| 497 | `route_distance += distance_matrix[current][checkpoint_idx]` - เพิ่มระยะทางจาก current ไป checkpoint |
| 498 | `current = checkpoint_idx` - ย้าย current position ไป checkpoint |
| 501 | `route_nodes.append(nodes[depot_idx].id)` - เพิ่ม depot node ID ลงใน route (จบที่ depot) |
| 502 | `route_distance += distance_matrix[checkpoint_idx][depot_idx]` - เพิ่มระยะทางจาก checkpoint กลับ depot |

**โครงสร้างเส้นทางที่สมบูรณ์:**
```
[Depot] → [Collection Node 1] → ... → [Collection Node N] → [Checkpoint] → [Depot]
   ↑                ↑                      ↑                      ↑              ↑
 เริ่ม          เก็บขยะ               เก็บขยะ              ทิ้งขยะ       จบ
```

**Flow การทำงาน:**
```
เยี่ยม node ที่เลือก → อัปเดต loads, distance, current
    ↓
ลบ node ออกจาก unvisited
    ↓
ทำซ้ำจนไม่มี node ที่เป็นไปได้
    ↓
ไป checkpoint → อัปเดต distance, current
    ↓
กลับ depot → อัปเดต distance
    ↓
สร้าง Route object
```

---

## 6. Validation Flow

### 6.1 ภาพรวมของ Validation Module

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 541-559

```python
def _validate_solution(self, solution: Solution, nodes: List[Node],
                      checkpoint_idx: int) -> Solution:
    """
    Validate solution against requirements.

    Checks:
    1. Every route starts at Node 1
    2. Every route visits checkpoint before returning to Node 1
    3. Every route ends at Node 1
    4. Trash at Node 1 is collected only once per vehicle (at start)
    5. No non-depot node is visited more than once (across all routes)
    6. No vehicle exceeds capacity
    7. All nodes are visited
    """
    errors = []
    checkpoint_node_id = nodes[checkpoint_idx].id

    # Track visited nodes
    all_visited = set()
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 541-543 | `def _validate_solution(self, solution, nodes, checkpoint_idx)` - ประกาศ validation method |
| 544-553 | Docstring อธิบาย 7 checks หลัก: |
| 546 | 1. ทุก route เริ่มที่ Node 1 |
| 547 | 2. ทุก route ไป checkpoint ก่อนกลับ Node 1 |
| 548 | 3. ทุก route จบที่ Node 1 |
| 549 | 4. เก็บขยะที่ Node 1 ครั้งเดียว (ตอนเริ่ม) |
| 550 | 5. ไม่ซ้ำ nodes (ยกเว้น depot) |
| 551 | 6. ไม่เกิน capacity |
| 552 | 7. เยี่ยมครบทุก node |
| 555 | `errors = []` - สร้าง empty list สำหรับเก็บ error messages |
| 556 | `checkpoint_node_id = nodes[checkpoint_idx].id` - ดึง checkpoint node ID (1-indexed) |
| 559 | `all_visited = set()` - สร้าง empty set สำหรับติดตาม nodes ที่ถูกเยี่ยม |

**7 Checks หลัก:**
```
Check 1: เริ่มที่ Node 1
├─ route_nodes[0] == 1

Check 2: ผ่าน Checkpoint ก่อนกลับ Node 1
├─ route_nodes[-2] == checkpoint_node_id

Check 3: จบที่ Node 1
├─ route_nodes[-1] == 1

Check 4: เก็บขยะที่ Node 1 ครั้งเดียว
├─ เก็บตอนเริ่ม route (เช็คจาก loads)

Check 5: ไม่ซ้ำ nodes (ยกเว้น depot และ checkpoint)
├─ ตรวจสอบว่า node ปรากฏครั้งเดียวในทุก routes

Check 6: ไม่เกิน capacity
├─ load <= capacity

Check 7: เยี่ยมครบทุก node
├─ all_visited == collection_nodes
```

### 6.2 การตรวจสอบโครงสร้างเส้นทาง

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 561-577

```python
for route in solution.routes:
    route_nodes = route.nodes

    # Check 1: Starts at Node 1
    if route_nodes[0] != 1:
        errors.append(f"Route {route.vehicle_id}: Does not start at Node 1 (starts at {route_nodes[0]})")

    # Check 3: Ends at Node 1
    if route_nodes[-1] != 1:
        errors.append(f"Route {route.vehicle_id}: Does not end at Node 1 (ends at {route_nodes[-1]})")

    # Check 2: Visits checkpoint before returning to Node 1
    # Find position of checkpoint and final depot
    if len(route_nodes) >= 3:
        # Second-to-last should be checkpoint (last is depot)
        if route_nodes[-2] != checkpoint_node_id:
            errors.append(f"Route {route.vehicle_id}: Does not visit checkpoint (Node {checkpoint_node_id}) before returning to Node 1")
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 561 | `for route in solution.routes:` - วนลูปผ่านทุก route ใน solution |
| 562 | `route_nodes = route.nodes` - ดึง list ของ node IDs จาก route |
| 565 | `if route_nodes[0] != 1:` - Check 1: ตรวจสอบว่าเริ่มที่ Node 1 หรือไม่ |
| 566 | `errors.append(...)` - ถ้าไม่ใช่ เพิ่ม error message |
| 569 | `if route_nodes[-1] != 1:` - Check 3: ตรวจสอบว่าจบที่ Node 1 หรือไม่ |
| 570 | `errors.append(...)` - ถ้าไม่ใช่ เพิ่ม error message |
| 574 | `if len(route_nodes) >= 3:` - Check 2: ตรวจสอบว่ามีอย่างน้อย 3 nodes (Depot → Checkpoint → Depot) |
| 576 | `if route_nodes[-2] != checkpoint_node_id:` - ตรวจสอบว่า element รองสุดท้ายเป็น checkpoint หรือไม่ |
| 577 | `errors.append(...)` - ถ้าไม่ใช่ เพิ่ม error message |

**Python Indexing ใน List:**
```
route_nodes = [1, 2, 5, 8, 20, 1]
               ^           ^     ^  ^
Index:        0           ...  -2  -1
              |           ...    |   |
           start        ...  checkpoint  end
```

**ตัวอย่างเส้นทางที่ถูกต้อง:**
```python
route_nodes = [1, 2, 3, 5, 8, 20, 1]
#              ^           ^     ^  ^
#            start    ...  checkpoint  end
```

**ตัวอย่างเส้นทางที่ถูกต้อง**:
```python
route_nodes = [1, 2, 3, 5, 8, 20, 1]
#              ^           ^     ^  ^
#            start    ...  checkpoint  end
```

### 6.3 การตรวจสอบการซ้ำ Nodes

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 579-584

```python
    # Check 4 & 5: Track visited nodes (excluding depot and checkpoint)
    for node_id in route_nodes[1:-1]:  # Exclude start and end depot
        if node_id != checkpoint_node_id:  # Checkpoint can be visited by each vehicle
            if node_id in all_visited:
                errors.append(f"Node {node_id} visited more than once across routes")
            all_visited.add(node_id)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 580 | `for node_id in route_nodes[1:-1]:` - วนลูปผ่าน nodes ระหว่าง start และ end depot (slice notation) |
| 581 | `if node_id != checkpoint_node_id:` - ข้าม checkpoint (เพราะทุกรถต้องไป) |
| 582 | `if node_id in all_visited:` - ตรวจสอบว่า node นี้ถูกเยี่ยมไปแล้วหรือยัง |
| 583 | `errors.append(...)` - ถ้าซ้ำ เพิ่ม error message |
| 584 | `all_visited.add(node_id)` - เพิ่ม node ID ลงใน set ของ nodes ที่เยี่ยมแล้ว |

**Python Slice Notation:**
```
route_nodes = [1, 2, 5, 8, 20, 1]
               [1:-1] = [2, 5, 8, 20]

เฉพาะ: 2, 5, 8, 20 (collection nodes + checkpoint)
ข้าม: depot เริ่ม (index 0) และ depot จบ (index -1)
```

**ตรรกะการตรวจสอบ:**
```
สำหรับทุก node_id ใน route_nodes[1:-1]:
    ↓
    ถ้า node_id != checkpoint:
        ↓
        ถ้า node_id อยู่ใน all_visited แล้ว:
            ↓
            ERROR: node ซ้ำ!
        ↓
        เพิ่ม node_id ลงใน all_visited
```

**ตัวอย่าง:**
```python
# Route 1: [1, 2, 3, 4, 20, 1]
# Route 2: [1, 5, 3, 6, 20, 1]
#                  ^
#                Node 3 ซ้ำ → Error!
```

**ตัวอย่าง**:
```python
# Route 1: [1, 2, 3, 4, 20, 1]
# Route 2: [1, 5, 3, 6, 20, 1]
#                  ^
#                Node 3 ซ้ำ → Error!
```

### 6.4 การตรวจสอบความครบถ้วน

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 586-605

```python
# Check 7: All collection nodes visited
collection_nodes = {n.id for n in nodes
                  if not n.is_depot and not n.is_checkpoint
                  and (n.general_demand > 0 or n.recycle_demand > 0)}

missing = collection_nodes - all_visited
if missing:
    errors.append(f"Nodes not visited: {sorted(missing)}")

solution.all_nodes_visited = len(missing) == 0
solution.all_routes_valid = len([e for e in errors if "Does not" in e]) == 0
solution.validation_errors = errors

return solution
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 587-589 | `collection_nodes = {n.id for n in nodes if ...}` - Set comprehension สร้าง set ของ node IDs ที่ต้องเยี่ยม: |
| 588 | `not n.is_depot` - ไม่ใช่ depot |
| 588 | `not n.is_checkpoint` - ไม่ใช่ checkpoint |
| 589 | `n.general_demand > 0 or n.recycle_demand > 0` - มีขยะที่ต้องเก็บ |
| 591 | `missing = collection_nodes - all_visited` - Set difference: nodes ที่ไม่ได้เยี่ยม |
| 592-593 | `if missing:` - ถ้ามี nodes ที่ไม่ได้เยี่ยม |
| 593 | `errors.append(...)` - เพิ่ม error message พร้อมรายชื่อ nodes ที่ขาด |
| 601 | `solution.all_nodes_visited = len(missing) == 0` - True ถ้าไม่มี missing nodes |
| 602 | `solution.all_routes_valid = len([e for e in errors if "Does not" in e]) == 0` - True ถ้าไม่มี structure errors |
| 603 | `solution.validation_errors = errors` - เก็บ error messages ทั้งหมด |
| 605 | `return solution` - คืนค่า solution ที่ถูก validate |

**Set Comprehension:**
```python
collection_nodes = {n.id for n in nodes
                  if not n.is_depot
                  and not n.is_checkpoint
                  and (n.general_demand > 0 or n.recycle_demand > 0)}

ผลลัพธ์: {2, 3, 4, 5, 6, 7, 8, ...}
(เฉพาะ nodes ที่มีขยะ ไม่ใช่ depot/checkpoint)
```

**Set Difference:**
```python
collection_nodes - all_visited = missing

ตัวอย่าง:
collection_nodes = {2, 3, 4, 5, 6, 7, 8}
all_visited = {2, 3, 5, 6, 8}
missing = {4, 7} → Error: "Nodes not visited: [4, 7]"
```

---

## 7. การคำนวณต้นทุน

### 7.1 โครงสร้างต้นทุน

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 503-518 (ใน Heuristic Solver)

```python
# Calculate costs
distance_km = route_distance / 1000.0
fuel_cost = distance_km * vehicle.fuel_cost_per_km

route = Route(
    vehicle_id=vehicle_id,
    vehicle_type=vehicle.type_id,
    nodes=route_nodes,
    distance_meters=route_distance,
    distance_km=distance_km,
    general_load=general_load,
    recycle_load=recycle_load,
    fixed_cost=vehicle.fixed_cost,
    fuel_cost=fuel_cost,
    total_cost=vehicle.fixed_cost + fuel_cost
)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 504 | `distance_km = route_distance / 1000.0` - แปลงระยะทางจากเมตรเป็นกิโลเมตร |
| 505 | `fuel_cost = distance_km * vehicle.fuel_cost_per_km` - คำนวณต้นทุนน้ำมัน = ระยะทาง(กม.) × อัตรา(บาท/กม.) |
| 507-518 | `route = Route(...)` - สร้าง Route object พร้อมข้อมูลทั้งหมด: |
| 508 | `vehicle_id=vehicle_id` - หมายเลขรถ |
| 509 | `vehicle_type=vehicle.type_id` - ประเภทรถ |
| 510 | `nodes=route_nodes` - List ของ node IDs ในเส้นทาง |
| 511 | `distance_meters=route_distance` - ระยะทาง (เมตร) |
| 512 | `distance_km=distance_km` - ระยะทาง (กิโลเมตร) |
| 513 | `general_load=general_load` - ปริมาณขยะทั่วไปที่เก็บ |
| 514 | `recycle_load=recycle_load` - ปริมาณขยะ recycle ที่เก็บ |
| 515 | `fixed_cost=vehicle.fixed_cost` - ต้นทุนคงที่ |
| 516 | `fuel_cost=fuel_cost` - ต้นทุนน้ำมัน |
| 517 | `total_cost=vehicle.fixed_cost + fuel_cost` - ต้นทุนรวม |

**ตัวอย่างการคำนวณ (20 nodes, หลัง refactor):**
```
route_distance = 4,162 เมตร
distance_km = 4,162 / 1,000 = 4.162 กม.

fuel_cost = 4.162 × 8 = 33.30 บาท

fixed_cost = 2,400 บาท
total_cost = 2,400 + 33.30 = 2,433.30 บาท  ← ตรงกับ AMPL optimal
```

### 7.2 การคำนวณระยะทางจริง (ไม่มี Penalty)

```python
# Recalculate actual distance using original distance matrix (no penalties)
route_distance = 0.0
for i in range(len(route_nodes) - 1):
    from_idx = route_nodes[i] - 1  # 1-indexed → 0-indexed
    to_idx = route_nodes[i + 1] - 1
    route_distance += distance_matrix[from_idx][to_idx]

distance_km = route_distance / 1000.0
fuel_cost = distance_km * vehicle.fuel_cost_per_km
```

**คำอธิบายโค้ด:**

**Actual Distance Recalculation** คำนวณระยะทางจริงโดยใช้ `distance_matrix` เดิม (ไม่ใช่ `expanded_dist` ที่มี BIG_PENALTY):

1. **ใช้ distance_matrix จริง**: ระยะทางที่แสดงในผลลัพธ์ต้องสะท้อนระยะทางจริง ไม่ใช่ค่า penalty
   - `distance_matrix` (ไม่มี penalty) → ใช้คำนวณ route_distance
   - `expanded_dist` (มี BIG_PENALTY) → ใช้เฉพาะภายใน OR-Tools optimizer

2. **Checkpoint อยู่ใน route_nodes แล้ว**: OR-Tools จัดตำแหน่ง checkpoint ให้ถูกต้องโดยอัตโนมัติ
   ไม่ต้อง append เพิ่มเหมือนวิธีเก่า

3. **Pair-wise Distance Calculation**:
   - วนลูปผ่านทุกคู่ของ nodes ที่ติดกัน
   - แปลง 1-indexed → 0-indexed
   - ดึงระยะทางจาก distance_matrix
   - สะสมระยะทางรวม

**ตัวอย่าง (หลัง refactor)**:
```python
# OR-Tools จัด route ให้เส้นทางผ่านใกล้ checkpoint โดยธรรมชาติ
route_nodes = [1, 2, 3, ..., 19, 20, 1]  # 20 = checkpoint (ตำแหน่งถูกต้อง)

# Pairs: (1,2), (2,3), ..., (19,20), (20,1)
# Distance = dist[0][1] + dist[1][2] + ... + dist[18][19] + dist[19][0]
# ผลลัพธ์: 4,162 เมตร (ไม่ใช่ 5,298 เมตรแบบเก่า)
```

### 7.3 การคำนวณต้นทุนรวม

```python
total_distance_km = total_distance_m / 1000.0
total_fixed = sum(r.fixed_cost for r in routes)
total_fuel = sum(r.fuel_cost for r in routes)

return Solution(
    status='OPTIMAL' if assignment else 'INFEASIBLE',
    routes=routes,
    num_vehicles_used=len(routes),
    total_distance_meters=total_distance_m,
    total_distance_km=total_distance_km,
    total_fixed_cost=total_fixed,
    total_fuel_cost=total_fuel,
    total_cost=total_fixed + total_fuel,
    all_nodes_visited=False,  # Will be validated
    all_routes_valid=False,
    validation_errors=[]
)
```

**คำอธิบายโค้ด:**

**Total Cost Calculation** คำนวณต้นทุนรวมทั้งหมด:

1. **Total Distance**: แปลงเมตรเป็นกิโลเมตร

2. **Total Fixed Cost**: `sum(r.fixed_cost for r in routes)`
   - ใช้ generator expression
   - รวมต้นทุนคงที่ของทุกเส้นทาง

3. **Total Fuel Cost**: `sum(r.fuel_cost for r in routes)`
   - รวมต้นทุนน้ำมันของทุกเส้นทาง

4. **Return Solution**: สร้าง Solution object พร้อมข้อมูลทั้งหมด

**ตัวอย่างการคำนวณ (138 nodes, 2 vehicles, หลัง refactor)**:
```
Route 1: 2,400 fixed + 133.64 fuel = 2,533.64
Route 2: 2,400 fixed + 133.64 fuel = 2,533.64

Total Fixed: 4,800.00
Total Fuel: 267.28
Total Cost: 5,067.28  ← ลดลงจาก 5,099.44 (−32.16 THB)
```

---

## 8. Checkpoint Constraint Implementation

### 8.1 ภาพรวมของ Checkpoint Constraint

ระบบนี้ implement constraint ที่ว่า **"ทุกรถที่ active ต้องผ่าน checkpoint (จุดทิ้ง) ก่อนกลับ depot"** โดยตรงใน OR-Tools optimizer ซึ่งเทียบเท่ากับ AMPL constraint:

```
x[checkpoint_node, depot_node, k] = vehicle_used[k]
```

แนวทางที่ใช้มี 2 ส่วนหลัก:

### 8.2 ส่วนที่ 1: BIG_PENALTY Distance Callback

```python
BIG_PENALTY = 10_000_000  # >> ระยะทางจริงใด ๆ (เมตร)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    dist = int(expanded_dist[from_node][to_node])
    # ทำให้ arc ที่ข้าม checkpoint แพงมาก
    if to_node == depot_idx and from_node not in checkpoint_set:
        dist += BIG_PENALTY
    return dist
```

**หลักการทำงาน:**
```
OR-Tools ต้องการ minimize total cost
├─ Arc: collection_node → depot     = actual_dist + 10,000,000  (แพงมาก)
└─ Arc: checkpoint → depot          = actual_dist               (ราคาปกติ)

∴ OR-Tools จะเลือก route ที่ผ่าน checkpoint ก่อนกลับ depot เสมอ
เพราะ cost ต่ำกว่าอย่างชัดเจน
```

**ผลลัพธ์**: OR-Tools "รู้" ตั้งแต่ต้นว่าต้องผ่าน checkpoint จึงวางแผนเส้นทาง
เพื่อให้ node สุดท้ายก่อน checkpoint อยู่ใกล้ checkpoint ที่สุด

### 8.3 ส่วนที่ 2: Phantom Checkpoint Copies (Multi-Vehicle)

```python
# สร้าง checkpoint copies สำหรับ multi-vehicle
total_nodes = num_nodes + (num_vehicles - 1)
checkpoint_copies = [checkpoint_idx] + list(range(num_nodes, total_nodes))

# Expanded distance matrix สำหรับ copies
for copy_idx in range(num_nodes, total_nodes):
    for j in range(num_nodes):
        d = int(distance_matrix[checkpoint_idx][j])
        expanded_dist[copy_idx][j] = d
        expanded_dist[j][copy_idx] = d

# กำหนด checkpoint copy ต่อ vehicle
for v, ckpt_node in enumerate(checkpoint_copies):
    ckpt_routing_idx = manager.NodeToIndex(ckpt_node)
    routing.AddDisjunction([ckpt_routing_idx], BIG_PENALTY)
    routing.SetAllowedVehiclesForIndex([v], ckpt_routing_idx)
```

**เหตุผลที่ต้องใช้ Checkpoint Copies:**
```
ข้อจำกัดของ OR-Tools:
└─ Mandatory node ถูกเยี่ยมได้โดยรถ 1 คันเท่านั้น

ปัญหา: ถ้า num_vehicles = 2 ทั้งสองคันต้องไป checkpoint
แต่ OR-Tools อนุญาตให้ 1 คันเท่านั้น

วิธีแก้: สร้าง "phantom copies" ที่มีตำแหน่งเดียวกับ checkpoint
├─ Copy 0 = checkpoint จริง (สำหรับรถ 0)
└─ Copy 1 = node พิเศษ ระยะทาง = checkpoint จริง (สำหรับรถ 1)

ผล: ทุกรถมี checkpoint ของตัวเอง ระยะทางถูกต้อง
```

### 8.4 การเปรียบเทียบผลลัพธ์

| Sheet | วิธีเก่า (Post-Processing) | วิธีใหม่ (Native Constraint) | ปรับปรุง |
|-------|---------------------------|------------------------------|---------|
| 20 nodes | 5.298 km / 2,442.38 THB | **4.162 km / 2,433.30 THB** | −1.136 km |
| 30 nodes | 5.879 km / 2,447.03 THB | **5.477 km / 2,443.82 THB** | −0.402 km |
| 50 nodes | 8.394 km / 2,467.15 THB | **7.695 km / 2,461.56 THB** | −0.699 km |
| 80 nodes | 14.163 km / 2,513.30 THB | **13.767 km / 2,510.14 THB** | −0.396 km |
| 138 nodes | 37.430 km / 5,099.44 THB | **33.410 km / 5,067.28 THB** | −4.020 km |

**หมายเหตุ**: ผลลัพธ์ 20 nodes ตรงกับ AMPL/NEOS optimal (2,433.296 THB) แทบทั้งหมด

---

## 9. Output Generation

### 9.1 การบันทึก Solution เป็น JSON

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 607-644

```python
def save_solution(self, solution: Solution, sheet_name: str, output_dir: Path) -> str:
    """Save solution to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sheet_clean = sheet_name.strip().replace(' ', '_')
    output_file = output_dir / f"vrp_solution_v2_{sheet_clean}.json"

    # Convert solution to dict
    solution_dict = {
        "status": solution.status,
        "num_vehicles_used": solution.num_vehicles_used,
        "total_distance_meters": solution.total_distance_meters,
        "total_distance_km": round(solution.total_distance_km, 3),
        "total_fixed_cost": solution.total_fixed_cost,
        "total_fuel_cost": round(solution.total_fuel_cost, 2),
        "total_cost": round(solution.total_cost, 2),
        "validation": {
            "all_nodes_visited": solution.all_nodes_visited,
            "all_routes_valid": solution.all_routes_valid,
            "errors": solution.validation_errors
        },
        "routes": [
            {
                "vehicle_id": r.vehicle_id,
                "vehicle_type": r.vehicle_type,
                "route": r.nodes,
                "distance_meters": r.distance_meters,
                "distance_km": round(r.distance_km, 3),
                "general_load": r.general_load,
                "recycle_load": r.recycle_load,
                "fixed_cost": r.fixed_cost,
                "fuel_cost": round(r.fuel_cost, 2),
                "total_cost": round(r.total_cost, 2)
            }
            for r in solution.routes
        ],
        "sheet_name": sheet_name
    }
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 607 | `def save_solution(self, solution, sheet_name, output_dir)` - ประกาศ method สำหรับบันทึก solution เป็น JSON |
| 609 | `output_dir.mkdir(parents=True, exist_ok=True)` - สร้าง output directory (พร้อม parent directories) |
| 611 | `sheet_clean = sheet_name.strip().replace(' ', '_')` - ทำความสะอาดชื่อ sheet (ลบ spaces, แทนที่ด้วย underscores) |
| 612 | `output_file = output_dir / f"vrp_solution_v2_{sheet_clean}.json"` - สร้าง path สำหรับ output file |
| 615-644 | `solution_dict = {...}` - แปลง Solution object เป็น dictionary: |
| 616 | `"status": solution.status` - สถานะของคำตอบ |
| 617 | `"num_vehicles_used": solution.num_vehicles_used` - จำนวนรถที่ใช้ |
| 618-622 | ข้อมูลระยะทางและต้นทุน (ใช้ `round()` สำหรับความแม่นยำ) |
| 623-627 | `"validation": {...}` - ผลการตรวจสอบ |
| 628-641 | `"routes": [...]` - List comprehension สร้าง list ของ routes แต่ละเส้นทาง |
| 643 | `"sheet_name": sheet_name` - ชื่อ sheet ต้นทาง |

**ตัวอย่าง JSON Output:**
```json
{
  "status": "OPTIMAL",
  "num_vehicles_used": 2,
  "total_distance_km": 33.41,
  "total_cost": 5067.28,
  "validation": {
    "all_nodes_visited": true,
    "all_routes_valid": true,
    "errors": []
  },
  "routes": [...]
}
```

### 9.2 การเขียน JSON File

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 646-649

```python
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(solution_dict, f, indent=2, ensure_ascii=False)

    return str(output_file)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 646 | `with open(output_file, 'w', encoding='utf-8') as f:` - เปิด file สำหรับเขียน: |
|  | `'w'` - write mode (สร้าง file ใหม่ หรือเขียนทับ file เดิม) |
|  | `'utf-8'` - ใช้ UTF-8 encoding เพื่อรองรับภาษาไทย |
| 647 | `json.dump(solution_dict, f, indent=2, ensure_ascii=False)` - เขียน dictionary ลง file: |
|  | `solution_dict` - dictionary ที่จะบันทึก |
|  | `f` - file object |
|  | `indent=2` - format ด้วย 2 spaces indentation (อ่านง่าย) |
|  | `ensure_ascii=False` - อนุญาต non-ASCII characters (ภาษาไทย) |
| 649 | `return str(output_file)` - คืนค่า path ของ file ที่บันทึก (เป็น string) |

**ตัวอย่าง JSON Output (formatted, 20 nodes):**
```json
{
  "status": "OPTIMAL",
  "num_vehicles_used": 1,
  "total_distance_km": 4.162,
  "total_cost": 2433.3,
  "routes": [
    {
      "vehicle_id": 1,
      "route": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 1],
      "distance_km": 4.162,
      ...
    }
  ]
}
```

### 9.3 การแก้ปัญหาทุก Sheets (solve_all method)

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 651-677

```python
def solve_all(self, time_limit_per_sheet: int = 60) -> Dict[str, Solution]:
    """Solve all sheets and save results."""
    print("\n" + "=" * 80)
    print("VRP SOLVER v2 - MULTI-SHEET PROCESSING")
    print("=" * 80)

    results = {}
    output_dir = self.base_dir / "results_v2"

    for sheet_name in self.sheet_names:
        try:
            solution = self.solve(sheet_name, time_limit_per_sheet)
            results[sheet_name] = solution

            # Save solution
            output_file = self.save_solution(solution, sheet_name, output_dir)
            print(f"  Solution saved: {output_file}")

            # Print summary
            print(f"\n  Result for '{sheet_name}':")
            print(f"    Status: {solution.status}")
            print(f"    Vehicles: {solution.num_vehicles_used}")
            print(f"    Distance: {solution.total_distance_km:.2f} km ({solution.total_distance_meters:.0f} m)")
            print(f"    Fixed cost: {solution.total_fixed_cost:.2f} THB")
            print(f"    Fuel cost: {solution.total_fuel_cost:.2f} THB")
            print(f"    Total cost: {solution.total_cost:.2f} THB")
            print(f"    Valid: {solution.all_routes_valid}, All nodes: {solution.all_nodes_visited}")
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 651 | `def solve_all(self, time_limit_per_sheet=60)` - ประกาศ method สำหรับแก้ปัญหาทุก sheets |
| 653-655 | `print(...)` - แสดง header ของ multi-sheet processing |
| 657 | `results = {}` - สร้าง empty dictionary สำหรับเก็บผลลัพธ์ |
| 658 | `output_dir = self.base_dir / "results_v2"` - กำหนด output directory path |
| 660 | `for sheet_name in self.sheet_names:` - วนลูปผ่านทุก sheet name |
| 661 | `try:` - เริ่ม try block สำหรับ error handling |
| 662 | `solution = self.solve(sheet_name, time_limit_per_sheet)` - เรียก solve() สำหรับ sheet นี้ |
| 663 | `results[sheet_name] = solution` - เก็บ solution ใน dictionary |
| 666 | `output_file = self.save_solution(...)` - บันทึก solution เป็น JSON file |
| 667 | `print(f"  Solution saved: {output_file}")` - แสดง path ของ file ที่บันทึก |
| 670-677 | `print(...)` - แสดง summary statistics: |
| 671 | ชื่อ sheet |
| 672 | สถานะ (OPTIMAL/FEASIBLE) |
| 673 | จำนวนรถที่ใช้ |
| 674 | ระยะทางรวม (กม. และ เมตร) |
| 675-677 | ต้นทุน (fixed, fuel, total) |

**Flow การทำงาน:**
```
เรียก solve_all()
    ↓
สร้าง results dictionary และ output directory
    ↓
วนลูปผ่านทุก sheet:
    ↓
    เรียก solve() → ได้ solution
    ↓
    บันทึก JSON file
    ↓
    แสดง summary statistics
    ↓
สร้าง summary file รวมทุก sheets
    ↓
คืนค่า results dictionary
```

### 9.4 การสร้าง Summary File

**ไฟล์:** `solvers/vrp_solver_v2.py`
**บรรทัด:** 689-702

```python
        # Save combined summary
        summary_file = output_dir / "vrp_all_sheets_summary_v2.json"
        summary = {}
        for name, sol in results.items():
            summary[name] = {
                "status": sol.status,
                "vehicles": sol.num_vehicles_used,
                "distance_km": round(sol.total_distance_km, 3),
                "total_cost": round(sol.total_cost, 2),
                "valid": sol.all_routes_valid and sol.all_nodes_visited
            }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
```

**คำอธิบายโค้ดแบบละเอียด (Line-by-Line):**

| บรรทัด | คำอธิบาย |
|--------|-----------|
| 690 | `summary_file = output_dir / "vrp_all_sheets_summary_v2.json"` - กำหนด path สำหรับ summary file |
| 691 | `summary = {}` - สร้าง empty dictionary สำหรับสรุปผลลัพธ์ |
| 692 | `for name, sol in results.items():` - วนลูปผ่านทุก sheet และ solution |
| 693-698 | `summary[name] = {...}` - สร้าง entry สำหรับแต่ละ sheet: |
| 694 | `"status": sol.status` - สถานะ (OPTIMAL/FEASIBLE) |
| 695 | `"vehicles": sol.num_vehicles_used` - จำนวนรถที่ใช้ |
| 696 | `"distance_km": round(...)` - ระยะทางรวม (กม.) |
| 697 | `"total_cost": round(...)` - ต้นทุนรวม (บาท) |
| 698 | `"valid": sol.all_routes_valid and sol.all_nodes_visited` - ความถูกต้องของคำตอบ |
| 701 | `with open(summary_file, 'w', encoding='utf-8') as f:` - เปิด file สำหรับเขียน |
| 702 | `json.dump(summary, f, indent=2, ensure_ascii=False)` - เขียน summary ลง file |

**ตัวอย่าง Summary Output:**
```json
{
  "20": {
    "status": "OPTIMAL",
    "vehicles": 1,
    "distance_km": 5.298,
    "total_cost": 2442.38,
    "valid": true
  },
  "30": {
    "status": "OPTIMAL",
    "vehicles": 1,
    "distance_km": 5.879,
    "total_cost": 2447.03,
    "valid": true
  }
}
```

**โครงสร้าง Output Files:**
```
results_v2/
├── vrp_solution_v2_20.json
├── vrp_solution_v2_30.json
├── vrp_solution_v2_40.json
└── vrp_all_sheets_summary_v2.json
```

---

## บทสรุป

### สรุปการทำงานของระบบ

VRP Solver v2 เป็นระบบที่ออกแบบมาเพื่อแก้ปัญหา Vehicle Routing Problem ที่มีความซับซ้อน โดยมีขั้นตอนการทำงานหลักดังนี้:

1. **Data Loading** (บรรทัด 110-211): อ่านและแปลงข้อมูลจาก Excel เป็น data structures ที่เหมาะสม
2. **Problem Formulation** (บรรทัด 274-352): สร้าง OR-Tools model พร้อม constraints และ objective function
3. **Optimization** (บรรทัด 274-539): ใช้ OR-Tools หรือ Heuristic ในการหาคำตอบ
4. **Post-Processing** (บรรทัด 380-392): เพิ่ม checkpoint เข้าไปในเส้นทาง
5. **Validation** (บรรทัด 541-605): ตรวจสอบความถูกต้องของคำตอบ
6. **Output Generation** (บรรทัด 607-709): บันทึกผลลัพธ์ในรูปแบบ JSON

### จุดเด่นของการออกแบบ

1. **Modular Design**: แยกส่วนการทำงานออกเป็น modules ที่ชัดเจน
   - Data structures: บรรทัด 39-88
   - Data loading: บรรทัด 110-211
   - OR-Tools solver: บรรทัด 274-428
   - Heuristic solver: บรรทัด 430-539
   - Validation: บรรทัด 541-605
   - Output generation: บรรทัด 607-709

2. **Robust Error Handling**: มี fallback solver เมื่อ OR-Tools ล้มเหลว (บรรทัด 256-267)

3. **Comprehensive Validation**: ตรวจสอบความถูกต้องอย่างละเอียด (7 checks หลัก)

4. **Clear Data Structures**: ใช้ dataclasses สำหรับความชัดเจน (Node, Vehicle, Route, Solution)

5. **Flexible Output**: บันทึกผลลัพธ์ในรูปแบบ JSON ที่อ่านง่าย (พร้อม summary file)

### แนวทางการพัฒนาต่อ

1. สามารถเพิ่มประเภทรถหลายแบบในการแก้ปัญหา (ปัจจุบันใช้รถที่มีต้นทุนน้ำมันต่ำสุด)
2. สามารถเพิ่ม constraints ประเภท time windows
3. สามารถปรับปรุง heuristic algorithm ให้มีประสิทธิภาพมากขึ้น
4. สามารถเพิ่ม visualization ของเส้นทางที่ได้

---

## อ้างอิงไฟล์ต้นทาง

- **ไฟล์หลัก:** `solvers/vrp_solver_v2.py`
- **จำนวนบรรทัด:** 756 บรรทัด
- **Data Structures:** บรรทัด 39-88
- **Class VRPSolverV2:** บรรทัด 91-709
- **Methods หลัก:**
  - `__init__`: บรรทัด 98-108
  - `load_sheet_data`: บรรทัด 110-211
  - `solve`: บรรทัด 213-272
  - `_solve_ortools`: บรรทัด 274-428
  - `_solve_heuristic`: บรรทัด 430-539
  - `_validate_solution`: บรรทัด 541-605
  - `save_solution`: บรรทัด 607-649
  - `solve_all`: บรรทัด 651-709
