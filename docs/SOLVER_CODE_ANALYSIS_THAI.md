# การวิเคราะห์ Flow และการทำงานของ VRP Solver v2

## สารบัญ

1. [ภาพรวมของระบบ](#1-ภาพรวมของระบบ)
2. [การออกแบบ Data Structures](#2-การออกแบบ-data-structures)
3. [Data Loading Flow](#3-data-loading-flow)
4. [OR-Tools Solver Flow](#4-or-tools-solver-flow)
5. [Heuristic Solver Flow](#5-heuristic-solver-flow)
6. [Validation Flow](#6-validation-flow)
7. [การคำนวณต้นทุน](#7-การคำนวณต้นทุน)
8. [Post-Processing](#8-post-processing)
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

Class `Node` ถูกกำหนดโดยใช้ Python decorator `@dataclass` ซึ่งเป็น feature ของ Python 3.7+ ที่ช่วยลดความซับซ้อนในการสร้าง class ที่เก็บข้อมูล โดยอัตโนมัติสร้าง `__init__`, `__repr__`, และ `__eq__` methods

**Attributes:**
- `id`: หมายเลขระบุจุด (ใช้ 1-indexing เพื่อความสอดคล้องกับข้อมูลจริง)
- `name`: ชื่อของจุด เช่น "Depot (Node 1)" หรือ "Node 15"
- `general_demand`: ปริมาณขยะทั่วไปที่ต้องเก็บ
- `recycle_demand`: ปริมาณขยะ recycle ที่ต้องเก็บ
- `is_depot`: ค่า boolean ระบุว่าเป็นจุดจอดรถหรือไม่ (เฉพาะ Node 1)
- `is_checkpoint`: ค่า boolean ระบุว่าเป็นจุดทิ้งขยะหรือไม่

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

Class `Vehicle` เก็บข้อมูลคุณสมบัติของแต่ละประเภทรถ โดยระบบอนุญาตให้มีหลายประเภทรถ แต่ในการแก้ปัญหาจะเลือกใช้รถที่มีต้นทุนน้ำมันต่ำที่สุด

**Attributes:**
- `type_id`: รหัสประเภทรถ (เช่น "A", "B", "C")
- `general_capacity`: ความจุสูงสุดสำหรับขยะทั่วไป (หน่วยเดียวกับ demand)
- `recycle_capacity`: ความจุสูงสุดสำหรับขยะ recycle
- `fixed_cost`: ต้นทุนคงที่ต่อคัน (บาท)
- `fuel_cost_per_km`: ต้นทุนน้ำมันต่อกิโลเมตร (บาท/กม.)

**การเลือกใช้รถ:**

```python
# บรรทัดที่ 232: เลือกประเภทรถที่มีต้นทุนน้ำมันต่ำสุด
best_vehicle = min(vehicles, key=lambda v: v.fuel_cost_per_km)
```

โค้ดนี้ใช้ฟังก์ชัน `min()` กับ `key` parameter เพื่อเลือก vehicle ที่มี `fuel_cost_per_km` ต่ำที่สุด ซึ่งเป็นกลยุทธ์ที่เหมาะสมเพื่อลดต้นทุนรวม

### 2.3 Route Data Structure

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

Class `Route` เก็บข้อมูลครบถ้วนของแต่ละเส้นทาง รวมทั้งข้อมูลการเดินทางและต้นทุน ซึ่งจะถูกใช้ในการแสดงผลและวิเคราะห์

**Attributes สำคัญ:**
- `nodes`: List ของ node IDs ที่เป็นลำดับการเดินทาง (เช่น [1, 2, 5, 8, 20, 1])
- `distance_meters` และ `distance_km`: ระยะทางรวมของเส้นทาง
- `general_load` และ `recycle_load`: ปริมาณขยะรวมที่เก็บในเส้นทางนี้
- `fixed_cost`, `fuel_cost`, `total_cost`: ต้นทุนแยกตามประเภทและรวม

### 2.4 Solution Data Structure

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

Class `Solution` เป็น container สำหรับเก็บผลลัพธ์ทั้งหมด ซึ่งสามารถใช้ในการวิเคราะห์และเปรียบเทียบวิธีการแก้ปัญหาต่างๆ ได้

**Attributes ที่ใช้ในการตรวจสอบ:**
- `all_nodes_visited`: ระบุว่าไปครบทุก node หรือไม่
- `all_routes_valid`: ระบุว่าทุกเส้นทางถูกต้องตามเงื่อนไขหรือไม่
- `validation_errors`: List ของข้อความ error ถ้ามี

---

## 3. Data Loading Flow

### 3.1 การเริ่มต้นระบบ (Initialization)

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

**คำอธิบายโค้ด:**

Method `__init__` คือ constructor ของ class ที่ถูกเรียกเมื่อสร้าง instance ใหม่ โค้ดนี้ทำหน้าที่:

1. **รับ path ของ Excel file** และเก็บไว้ใน `self.excel_file`
2. **กำหนด base directory** โดยใช้ `Path(excel_file).parent.parent` ซึ่งจะได้ directory 2 ระดับขึ้นจาก file
3. **โหลดชื่อ sheets ทั้งหมด** จาก Excel file โดยใช้ `openpyxl.load_workbook()`
4. **แสดงข้อมูลเริ่มต้น** เพื่อยืนยันว่าระบบทำงาน

### 3.2 การโหลดข้อมูลจาก Excel Sheet

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

**คำอธิบายโค้ด:**

Method `load_sheet_data` ทำหน้าที่อ่านและแปลงข้อมูลจาก Excel sheet ให้อยู่ในรูปแบบที่ระบบใช้งานได้:

1. **อ่าน Excel sheet** โดยใช้ `pandas.read_excel()` ซึ่งเป็น library ที่มีประสิทธิภาพสูงในการจัดการข้อมูล tabular
2. **กำหนดจำนวน nodes** โดยหาค่าสูงสุดของคอลัมน์ 'Destination' ซึ่งเป็นหมายเลข node สูงสุด

### 3.3 การสร้าง Distance Matrix

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

**คำอธิบายโค้ด:**

**Distance Matrix** คือเมทริกซ์สมมาตร N x N ที่เก็บระยะทางระหว่างทุกคู่ของ nodes:

1. **สร้าง zero matrix** โดยใช้ `np.zeros((num_nodes, num_nodes))` ซึ่งเป็น NumPy array ขนาด N x N เต็มไปด้วยศูนย์

2. **วนลูปผ่านทุกแถว** ใน DataFrame โดยใช้ `df.iterrows()` ซึ่งคืนค่า (index, row) สำหรับแต่ละแถว

3. **แปลงเป็น 0-indexed** โดยลบ 1 จาก node_id เพราะ NumPy array ใช้ 0-indexing

4. **อ่านระยะทาง** จากคอลัมน์ `Origin_j` สำหรับทุก j (1 ถึง num_nodes)

5. **เติมข้อมูลลง matrix** ทั้งสองทิศทางเพราะ matrix สมมาตร (distance[A][B] = distance[B][A])

**ตัวอย่าง Distance Matrix:**

```
      Node1  Node2  Node3  Node4
Node1   0    100    250    180
Node2  100     0    300    220
Node3  250   300      0    150
Node4  180   220    150      0
```

### 3.4 การสร้าง Node Objects

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

**คำอธิบายโค้ด:**

โค้ดนี้ทำหน้าที่สร้าง Node objects จากข้อมูลใน Excel:

1. **ตรวจสอบ Checkpoint Node**: ถ้าค่าในคอลัมน์ 'ขยะทั่วไป' เป็น 'จุดทิ้ง' แสดงว่า node นี้คือจุดทิ้งขยะ

2. **จัดการ Missing Values**: ใช้ `try-except` block เพื่อจัดการกรณีที่ค่าไม่สามารถแปลงเป็น float ได้

3. **ระบุ Depot**: Node 1 ถูกกำหนดให้เป็น depot เสมอ (`is_depot = (node_id == 1)`)

4. **สร้าง Node Object**: ใช้ Node dataclass constructor พร้อมข้อมูลทั้งหมด

### 3.5 การสร้าง Vehicle Objects

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

**คำอธิบายโค้ด:**

โค้ดนี้สร้าง Vehicle objects จากข้อมูลใน Excel:

1. **ใช้ Dictionary**: `vehicles_dict` ใช้เพื่อป้องกันการสร้าง vehicle ซ้ำ (check `if v_type not in vehicles_dict`)

2. **Fallback Values**: ใช้ `row.get()` พร้อม default values เพื่อรองรับกรณีที่ข้อมูลไม่ครบ

3. **Typo Handling**: มีการจัดการ typo ในชื่อคอลัมน์ 'cap for gereral ' (มี space และสะกดผิด)

---

## 4. OR-Tools Solver Flow

### 4.1 การเริ่มต้น OR-Tools Solver

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

**คำอธิบายโค้ด:**

Method `solve` เป็น entry point หลักสำหรับการแก้ปัญหา:

1. **โหลดข้อมูล**: เรียก `load_sheet_data()` เพื่ออ่านข้อมูลจาก Excel
2. **แปลงเป็น 0-indexed**: OR-Tools ใช้ 0-indexing ภายใน จึงต้องแปลงจาก 1-indexed
3. **เลือกประเภทรถ**: เลือก vehicle ที่มีต้นทุนน้ำมันต่ำสุด

### 4.2 การคำนวณจำนวนรถขั้นต่ำ

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

**คำอธิบายโค้ด:**

การคำนวณจำนวนรถขั้นต่ำที่ต้องใช้:

1. **คำนวณ Total Demand**: รวม demand ของทุก node โดยใช้ `sum()` กับ generator expression

2. **คำนวณ Minimum Vehicles**:
   - `min_veh_gen`: จำนวนรถขั้นต่ำสำหรับขยะทั่วไป = ceiling(total_general / capacity)
   - `min_veh_rec`: จำนวนรถขั้นต่ำสำหรับขยะ recycle = ceiling(total_recycle / capacity)

3. **เลือกค่าสูงสุด**: `num_vehicles = max(min_veh_gen, min_veh_rec, 1)` เพื่อให้มีรถเพียงพอทั้งสองประเภท

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

```python
def _solve_ortools(self, nodes: List[Node], vehicle: Vehicle,
                   distance_matrix: np.ndarray, depot_idx: int,
                   checkpoint_idx: int, num_vehicles: int,
                   time_limit: int) -> Solution:
    """
    Solve using OR-Tools and post-process to ensure checkpoint visits.

    The approach:
    1. Solve VRP normally (checkpoint is excluded from collection)
    2. Post-process: Insert checkpoint visit before each route's return to depot
    3. Recalculate distances with checkpoint included
    """
    num_nodes = len(nodes)

    # OR-Tools setup - exclude checkpoint from collection nodes
    # We'll add it back during post-processing
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
```

**คำอธิบายโค้ด:**

Method `_solve_ortools` เป็น core logic ของ OR-Tools solver:

1. **RoutingIndexManager**: แปลงระหว่าง user indices (1,2,3,...) กับ internal indices ของ solver
   - `num_nodes`: จำนวน nodes ทั้งหมด
   - `num_vehicles`: จำนวนรถที่ใช้
   - `depot_idx`: index ของ depot (เสมอ 0)

2. **RoutingModel**: สร้าง routing optimization model ที่มี constraints และ objective function

**หมายเหตุ**: ระบบใช้ approach ที่แยก checkpoint ออกจากการ optimization แล้วเพิ่มกลับมาใน post-processing ซึ่งทำให้การ optimize เร็วขึ้น

### 4.4 Distance Callback

```python
# Distance callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(distance_matrix[from_node][to_node])

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
```

**คำอธิบายโค้ด:**

**Distance Callback** คือฟังก์ชันที่ OR-Tools เรียกใช้เพื่อทราบระยะทางระหว่างสอง nodes:

1. **รับ Indices**: ฟังก์ชันรับ `from_index` และ `to_index` ซึ่งเป็น internal indices ของ OR-Tools

2. **แปลงเป็น User Nodes**: ใช้ `manager.IndexToNode()` เพื่อแปลงเป็น node IDs ปกติ (0-indexed)

3. **คืนค่าระยะทาง**: ดึงค่าจาก `distance_matrix` และแปลงเป็น integer

4. **ลงทะเบียน Callback**: `RegisterTransitCallback()` ลงทะเบียน callback กับ solver และคืนค่า index

5. **ตั้งค่า Cost Evaluator**: `SetArcCostEvaluatorOfAllVehicles()` กำหนดให้ใช้ callback นี้สำหรับทุกคัน

**Flow ของการเรียกใช้ Callback:**

```
Solver ต้องการระยะทาง Node A → Node B
    ↓
เรียก distance_callback(A_index, B_index)
    ↓
แปลง indices → user nodes
    ↓
ดึงระยะทางจาก distance_matrix[A][B]
    ↓
คืนค่าให้ Solver
    ↓
Solver ใช้ค่านี้ในการคำนวณ cost
```

### 4.5 Capacity Constraints

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
```

**คำอธิบายโค้ด:**

**Capacity Constraints** ใช้เพื่อให้แน่ใจว่ารถไม่บรรทุกเกินความจุ:

1. **สร้าง Demand Arrays**: แปลง Node objects เป็น arrays ของ demands

2. **Demand Callbacks**: สองฟังก์ชัน `demand_general_callback` และ `demand_recycle_callback` คืนค่า demand ของแต่ละ node

3. **Register Unary Transit Callback**: `RegisterUnaryTransitCallback()` ใช้สำหรับ callbacks ที่รับ parameter เดียว (from_index เท่านั้น)

4. **AddDimensionWithVehicleCapacity**: เพิ่ม dimension สำหรับ capacity constraints:
   - Callback index
   - Slack (0 = ไม่อนุญาตให้มีค่าสะสมติดลบ)
   - Capacities ของแต่ละรถ (list ความยาว num_vehicles)
   - Start cumul to zero (เริ่มนับจาก 0)
   - ชื่อ dimension

**หมายเหตุสำคัญ**:
- Node 1 (Depot) มี demand ที่ถูกเก็บที่จุดเริ่มต้น ดังนั้นจึงถูกรวมในการคำนวณ
- Checkpoint มี demand = 0 เพราะไม่มีการเก็บขยะที่นั่น

### 4.6 Checkpoint Handling

```python
# Make checkpoint visit optional during optimization (we'll add it in post-processing)
# This allows OR-Tools to focus on optimizing collection routes
checkpoint_routing_idx = manager.NodeToIndex(checkpoint_idx)
routing.AddDisjunction([checkpoint_routing_idx], 0)  # Zero penalty for not visiting
```

**คำอธิบายโค้ด:**

**Disjunction** คือ feature ของ OR-Tools ที่อนุญาตให้ "ข้าม" nodes บาง nodes ได้:

1. **แปลง Checkpoint Index**: ใช้ `manager.NodeToIndex()` เพื่อแปลง checkpoint index เป็น internal index

2. **Add Disjunction**: `routing.AddDisjunction([checkpoint_routing_idx], 0)` ทำให้ checkpoint เป็น optional
   - `[checkpoint_routing_idx]`: List ของ indices ที่เป็น disjunction
   - `0`: Penalty สำหรับไม่เยี่ยม (0 = ไม่มี penalty)

**เหตุผลของการทำเช่นนี้**:
- OR-Tools จะ focus กับการ optimize collection routes
- Checkpoint จะถูกเพิ่มกลับมาใน post-processing ด้วยตำแหน่งที่เหมาะสม (ก่อนกลับ depot)

### 4.7 Search Parameters

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

**คำอธิบายโค้ด:**

**Search Parameters** คือการตั้งค่าวิธีการค้นหาคำตอบ:

1. **DefaultRoutingSearchParameters**: สร้าง parameters object พร้อมค่าเริ่มต้น

2. **First Solution Strategy**: `PATH_CHEAPEST_ARC`
   - เริ่มจาก depot
   - เลือก node ถัดไปที่มีระยะทางน้อยที่สุด
   - ทำซ้ำจนครบทุก node

3. **Local Search Metaheuristic**: `GUIDED_LOCAL_SEARCH`
   - ปรับปรุงคำตอบเริ่มต้นด้วย local search
   - ใช้ penalties เพื่อหลีกเลี่ยง local optima

4. **Time Limit**: จำกัดเวลาการแก้ปัญหา (default 60 วินาที)

5. **Solve**: `routing.SolveWithParameters()` เริ่มการแก้ปัญหาและคืนค่า assignment object

### 4.8 การดึงคำตอบ

```python
if not assignment:
    raise Exception("OR-Tools could not find a solution")

# Extract solution and post-process to add checkpoint visits
routes = []
total_distance_m = 0
checkpoint_node_id = nodes[checkpoint_idx].id  # 1-indexed

for vehicle_id in range(num_vehicles):
    index = routing.Start(vehicle_id)
    route_nodes = []
    general_load = 0
    recycle_load = 0

    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        node = nodes[node_idx]

        # Skip checkpoint if OR-Tools included it (we'll add it at the right position)
        if node_idx != checkpoint_idx:
            route_nodes.append(node.id)  # 1-indexed
            general_load += node.general_demand
            recycle_load += node.recycle_demand

        index = assignment.Value(routing.NextVar(index))
```

**คำอธิบายโค้ด:**

**Solution Extraction** คือการดึงเส้นทางจาก assignment:

1. **ตรวจสอบ Solution**: ถ้า `assignment` เป็น None แสดงว่าไม่พบคำตอบ

2. **วนลูปผ่าน Vehicles**: สำหรับแต่ละ vehicle_id จาก 0 ถึง num_vehicles-1

3. **เริ่มจาก Start Node**: `routing.Start(vehicle_id)` คืนค่า index ของจุดเริ่มต้น

4. **วนลูปผ่าน Path**: `while not routing.IsEnd(index)` วนจนถึงจุดสิ้นสุด (depot)

5. **แปลง Index → Node**: ใช้ `manager.IndexToNode(index)` เพื่อแปลงเป็น user node

6. **สะสม Loads**: เพิ่ม demand ของแต่ละ node เข้ากับ loads ปัจจุบัน

7. **เลื่อนไป Node ถัดไป**: `assignment.Value(routing.NextVar(index))` คืนค่า index ถัดไป

---

## 5. Heuristic Solver Flow

### 5.1 ภาพรวมของ Heuristic Solver

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

**คำอธิบายโค้ด:**

Method `_solve_heuristic` เป็น fallback solver ที่ใช้ **Nearest Neighbor Heuristic**:

1. **วัตถุประสงค์**: ใช้เมื่อ OR-Tools ล้มเหลว หรือไม่มี OR-Tools

2. **หลักการ**: เลือก node ที่ใกล้ที่สุดที่เป็นไปได้ (feasible)

3. **Unvisited Set**: ใช้ Python `set` เพื่อเก็บ nodes ที่ยังไม่ได้เยี่ยม
   - `discard()` ลบ element ถ้ามี ไม่ error ถ้าไม่มี
   - Depot และ checkpoint ถูก exclude เพราะถูกจัดการแยก

### 5.2 การสร้างเส้นทาง

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

**คำอธิบายโค้ด:**

**Route Construction** สร้างเส้นทางทีละเส้นจนกว่าจะเยี่ยมครบทุก node:

1. **While Loop**: `while unvisited:` ทำซ้ำจนกว่า set จะว่าง

2. **เริ่มเส้นทางใหม่**:
   - เพิ่ม vehicle_id
   - เริ่มที่ depot (`current = depot_idx`)
   - เริ่ม route_nodes พร้อม node ID ของ depot
   - เริ่มสะสม loads จาก depot

### 5.3 การเลือก Node ถัดไป (Nearest Neighbor)

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

**คำอธิบายโค้ด:**

**Nearest Neighbor Selection** เลือก node ที่ใกล้ที่สุดที่เป็นไปได้:

1. **Initialize**: `best_node = None` และ `best_dist = float('inf')` เพื่อเริ่มการเปรียบเทียบ

2. **วนลูปผ่าน Unvisited**: สำหรับทุก node ใน unvisited set

3. **ตรวจสอบ Capacity Constraints**:
   ```python
   if (general_load + node.general_demand > vehicle.general_capacity or
       recycle_load + node.recycle_demand > vehicle.recycle_capacity):
       continue  # ข้าม node นี้ ถ้าเกินความจุ
   ```

4. **เปรียบเทียบระยะทาง**: ถ้าระยะทางน้อยกว่าค่าที่ดีที่สุด ให้ update

5. **ตรวจสอบการหยุด**: `if best_node is None: break` หยุดเมื่อไม่มี node ที่เป็นไปได้อีก

### 5.4 การเดินทางไป Checkpoint และกลับ Depot

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

**คำอธิบายโค้ด:**

**Completion of Route** ปิดท้ายเส้นทาง:

1. **Visit Best Node**: เพิ่ม node ที่เลือกลงใน route และอัปเดต:
   - `route_nodes`: เพิ่ม node ID
   - `route_distance`: เพิ่มระยะทาง
   - `general_load`, `recycle_load`: เพิ่ม loads
   - `unvisited`: ลบ node ออกจาก set
   - `current`: ย้ายไป node ใหม่

2. **Go to Checkpoint**: หลังจากเก็บขยะครบแล้ว ไป checkpoint:
   - เพิ่ม checkpoint node ID ลง route
   - เพิ่มระยะทางจาก current node ไป checkpoint
   - อัปเดต current เป็น checkpoint

3. **Return to Depot**: จาก checkpoint กลับ depot:
   - เพิ่ม depot node ID ลง route
   - เพิ่มระยะทางจาก checkpoint ไป depot

**โครงสร้างเส้นทางที่สมบูรณ์**:
```
[Depot] → [Collection Node 1] → ... → [Collection Node N] → [Checkpoint] → [Depot]
```

---

## 6. Validation Flow

### 6.1 ภาพรวมของ Validation Module

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

**คำอธิบายโค้ด:**

Method `_validate_solution` ทำหน้าที่ตรวจสอบความถูกต้องของคำตอบ:

1. **วัตถุประสงค์**: มั่นใจว่าคำตอบสอดคล้องกับ requirements ทั้งหมด

2. **7 Checks หลัก**:
   - เริ่มที่ Node 1
   - ผ่าน Checkpoint ก่อนกลับ Node 1
   - จบที่ Node 1
   - เก็บขยะที่ Node 1 ครั้งเดียว (ตอนเริ่ม)
   - ไม่ซ้ำ nodes (ยกเว้น depot)
   - ไม่เกิน capacity
   - เยี่ยมครบทุก node

3. **Tracking**: ใช้ `all_visited` set เพื่อติดตาม nodes ที่ถูกเยี่ยม

### 6.2 การตรวจสอบโครงสร้างเส้นทาง

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

**คำอธิบายโค้ด:**

**Route Structure Validation** ตรวจสอบโครงสร้างของแต่ละเส้นทาง:

1. **Check 1 - Start at Node 1**: `route_nodes[0] != 1`
   - `route_nodes[0]` คือ element แรกของ list
   - ต้องเป็น 1 (Depot) เสมอ

2. **Check 3 - End at Node 1**: `route_nodes[-1] != 1`
   - `route_nodes[-1]` คือ element สุดท้ายของ list (Python indexing)
   - ต้องเป็น 1 (Depot) เสมอ

3. **Check 2 - Checkpoint Before Return**: `route_nodes[-2] != checkpoint_node_id`
   - `route_nodes[-2]` คือ element รองสุดท้าย
   - ต้องเป็น checkpoint node ID
   - โครงสร้าง: [..., checkpoint, depot]

**ตัวอย่างเส้นทางที่ถูกต้อง**:
```python
route_nodes = [1, 2, 3, 5, 8, 20, 1]
#              ^           ^     ^  ^
#            start    ...  checkpoint  end
```

### 6.3 การตรวจสอบการซ้ำ Nodes

```python
    # Check 4 & 5: Track visited nodes (excluding depot and checkpoint)
    for node_id in route_nodes[1:-1]:  # Exclude start and end depot
        if node_id != checkpoint_node_id:  # Checkpoint can be visited by each vehicle
            if node_id in all_visited:
                errors.append(f"Node {node_id} visited more than once across routes")
            all_visited.add(node_id)
```

**คำอธิบายโค้ด:**

**Duplicate Node Detection** ตรวจสอบว่าไม่มี node ที่ถูกเยี่ยมซ้ำ:

1. **Slice Notation**: `route_nodes[1:-1]`
   - `[1:]` เริ่มจาก index 1 (ข้าม depot เริ่ม)
   - `[:-1]` จบถึง index -2 (ข้าม depot จบ)
   - ผลลัพธ์: nodes ระหว่าง start และ end depot

2. **Checkpoint Exclusion**: `if node_id != checkpoint_node_id`
   - Checkpoint สามารถถูกเยี่ยมโดยทุกรถ (ไม่ผิด)
   - Collection nodes ต้องถูกเยี่ยมครั้งเดียว

3. **Duplicate Detection**: `if node_id in all_visited`
   - ถ้า node อยู่ใน set แล้ว แสดงว่าซ้ำ
   - เพิ่ม error message

**ตัวอย่าง**:
```python
# Route 1: [1, 2, 3, 4, 20, 1]
# Route 2: [1, 5, 3, 6, 20, 1]
#                  ^
#                Node 3 ซ้ำ → Error!
```

### 6.4 การตรวจสอบความครบถ้วน

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

**คำอธิบายโค้ด:**

**Completeness Check** ตรวจสอบว่าทุก node ถูกเยี่ยม:

1. **Set Comprehension**: สร้าง `collection_nodes` set จาก nodes ที่มี:
   - `not n.is_depot`: ไม่ใช่ depot
   - `not n.is_checkpoint`: ไม่ใช่ checkpoint
   - `demand > 0`: มีขยะที่ต้องเก็บ

2. **Set Difference**: `collection_nodes - all_visited`
   - คืนค่า elements ที่อยู่ใน collection_nodes แต่ไม่อยู่ใน all_visited
   - ผลลัพธ์คือ nodes ที่ไม่ได้เยี่ยม

3. **Update Solution Flags**:
   - `all_nodes_visited`: True ถ้าไม่มี missing nodes
   - `all_routes_valid`: True ถ้าไม่มี errors ที่มี "Does not"

---

## 7. การคำนวณต้นทุน

### 7.1 โครงสร้างต้นทุน

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

**คำอธิบายโค้ด:**

**Cost Calculation** คำนวณต้นทุนของแต่ละเส้นทาง:

1. **Distance Conversion**: `route_distance / 1000.0`
   - แปลงเมตรเป็นกิโลเมตร
   - ใช้ float division ใน Python 3

2. **Fuel Cost**: `distance_km * vehicle.fuel_cost_per_km`
   - ต้นทุนน้ำมัน = ระยะทาง(กม.) × อัตรา(บาท/กม.)
   - เช่น: 5.298 km × 8 บาท/กม. = 42.38 บาท

3. **Total Cost**: `vehicle.fixed_cost + fuel_cost`
   - ต้นทุนรวม = ต้นทุนคงที่ + ต้นทุนน้ำมัน
   - เช่น: 2,400 + 42.38 = 2,442.38 บาท

### 7.2 การคำนวณระยะทางพร้อม Checkpoint

```python
# Only process non-trivial routes (has collection nodes besides depot)
if len(route_nodes) > 1:
    # POST-PROCESSING: Insert checkpoint before returning to depot
    # Route structure: [depot, ...collections..., checkpoint, depot]
    route_nodes.append(checkpoint_node_id)
    route_nodes.append(1)  # End at depot

    # Calculate actual distance with checkpoint included
    route_distance = 0.0
    for i in range(len(route_nodes) - 1):
        from_idx = route_nodes[i] - 1  # Convert to 0-indexed
        to_idx = route_nodes[i + 1] - 1
        route_distance += distance_matrix[from_idx][to_idx]
```

**คำอธิบายโค้ด:**

**Distance Calculation with Checkpoint** คำนวณระยะทางจริงหลังจากเพิ่ม checkpoint:

1. **Non-trivial Routes Check**: `if len(route_nodes) > 1`
   - ข้าม routes ที่ว่างเปลย (เฉพาะ depot)
   - เพื่อประหยัดการคำนวณ

2. **Post-Processing**:
   - `route_nodes.append(checkpoint_node_id)`: เพิ่ม checkpoint
   - `route_nodes.append(1)`: เพิ่ม depot ท้ายสุด

3. **Pair-wise Distance Calculation**:
   - วนลูปผ่านทุกคู่ของ nodes ที่ติดกัน
   - แปลง 1-indexed → 0-indexed
   - ดึงระยะทางจาก distance_matrix
   - สะสมระยะทางรวม

**ตัวอย่าง**:
```python
route_nodes = [1, 2, 5, 20, 1]  # 20 = checkpoint

# Pairs: (1,2), (2,5), (5,20), (20,1)
# Distance = dist[0][1] + dist[1][4] + dist[4][19] + dist[19][0]
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

**ตัวอย่างการคำนวณ**:
```
Route 1: 2,400 fixed + 137.50 fuel = 2,537.50
Route 2: 2,400 fixed + 161.94 fuel = 2,561.94

Total Fixed: 4,800.00
Total Fuel: 299.44
Total Cost: 5,099.44
```

---

## 8. Post-Processing

### 8.1 วัตถุประสงค์ของ Post-Processing

```python
# Only process non-trivial routes (has collection nodes besides depot)
if len(route_nodes) > 1:
    # POST-PROCESSING: Insert checkpoint before returning to depot
    # Route structure: [depot, ...collections..., checkpoint, depot]
    route_nodes.append(checkpoint_node_id)
    route_nodes.append(1)  # End at depot
```

**คำอธิบายโค้ด:**

**Post-Processing** คือขั้นตอนที่สำคัญในการปรับปรุงคำตอบจาก OR-Tools:

1. **เหตุผล**: OR-Tools ไม่รองรับ checkpoint constraint โดยตรง
   - VRP มาตรฐาน: depot → collection → depot
   - VRP นี้: depot → collection → checkpoint → depot

2. **Approach**:
   - ให้ OR-Tools แก้ปัญหาโดยไม่มี checkpoint constraint
   - หลังได้คำตอบ ใส่ checkpoint ก่อนกลับ depot
   - คำนวณระยะทางใหม่

3. **ข้อดี**:
   - OR-Tools focus กับ collection routes
   - Checkpoint ถูกเพิ่มในตำแหน่งที่ถูกต้องเสมอ
   - ลดความซับซ้อนของ constraints

### 8.2 การคำนวณระยะทางใหม่

```python
# Calculate actual distance with checkpoint included
route_distance = 0.0
for i in range(len(route_nodes) - 1):
    from_idx = route_nodes[i] - 1  # Convert to 0-indexed
    to_idx = route_nodes[i + 1] - 1
    route_distance += distance_matrix[from_idx][to_idx]

distance_km = route_distance / 1000.0
fuel_cost = distance_km * vehicle.fuel_cost_per_km
```

**คำอธิบายโค้ด:**

**Re-calculation of Distance** หลังจากเพิ่ม checkpoint:

1. **Loop Through Pairs**: วนลูปผ่านทุกคู่ของ nodes ที่ติดกัน
   - `range(len(route_nodes) - 1)`: 0, 1, 2, ..., n-2
   - เชื่อม node[i] → node[i+1]

2. **Index Conversion**: แปลง 1-indexed → 0-indexed
   - `route_nodes[i] - 1`: จาก user node ID เป็น matrix index

3. **Distance Accumulation**: สะสมระยะทางจาก distance matrix

**ตัวอย่าง Flow**:
```
route_nodes = [1, 2, 5, 20, 1]

i=0: from=0, to=1  → dist[0][1]
i=1: from=1, to=4  → dist[1][4]
i=2: from=4, to=19 → dist[4][19]
i=3: from=19, to=0 → dist[19][0]

Total = Σ distances
```

---

## 9. Output Generation

### 9.1 การบันทึก Solution เป็น JSON

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

**คำอธิบายโค้ด:**

Method `save_solution` แปลง Solution object เป็น JSON format:

1. **Create Directory**: `output_dir.mkdir(parents=True, exist_ok=True)`
   - `parents=True`: สร้าง parent directories ด้วย
   - `exist_ok=True`: ไม่ error ถ้ามีอยู่แล้ว

2. **Clean Sheet Name**: `sheet_name.strip().replace(' ', '_')`
   - ลบ spaces ที่ต้น/ท้าย
   - แทนที่ spaces ด้วย underscores

3. **Build Dictionary**: แปลง Solution เป็น dict พร้อม:
   - Summary statistics (status, vehicles, distance, costs)
   - Validation results
   - Routes detail (list comprehension)

4. **Round Values**: ใช้ `round()` เพื่อความแม่นยำในการแสดงผล
   - 3 decimal places สำหรับระยะทาง
   - 2 decimal places สำหรับต้นทุน

### 9.2 การเขียน JSON File

```python
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(solution_dict, f, indent=2, ensure_ascii=False)

    return str(output_file)
```

**คำอธิบายโค้ด:**

**JSON Writing** บันทึก dictionary ลง file:

1. **Open File**: `with open(output_file, 'w', encoding='utf-8')`
   - `'w'`: write mode
   - `'utf-8'`: รองรับภาษาไทย

2. **JSON Dump**: `json.dump(solution_dict, f, indent=2, ensure_ascii=False)`
   - `solution_dict`: dictionary ที่จะบันทึก
   - `f`: file object
   - `indent=2`: format ด้วย 2 spaces indentation
   - `ensure_ascii=False`: อนุญาต non-ASCII characters (ภาษาไทย)

3. **Return Path**: คืนค่า path ของ file ที่บันทึก

### 9.3 การแก้ปัญหาทุก Sheets

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

**คำอธิบายโค้ด:**

Method `solve_all` แก้ปัญหาทุก sheets ใน Excel file:

1. **Initialization**: สร้าง empty dictionary และ output directory

2. **Loop Through Sheets**: วนลูปผ่านทุก sheet name
   - เรียก `self.solve()` เพื่อแก้ปัญหา
   - เก็บผลลัพธ์ใน `results` dict

3. **Save and Display**: บันทึกและแสดงผล
   - บันทึก JSON file
   - แสดง summary statistics

4. **Error Handling**: ใช้ try-except เพื่อจัดการ errors

### 9.4 การสร้าง Summary File

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

**คำอธิบายโค้ด:**

**Summary File Generation** สร้าง file สรุปผลลัพธ์ทุก sheets:

1. **Create Summary Dict**: สร้าง dictionary ที่มีข้อมูลสำคัญ:
   - Status: OPTIMAL/FEASIBLE
   - Vehicles: จำนวนรถที่ใช้
   - Distance: ระยะทางรวม (กม.)
   - Total Cost: ต้นทุนรวม (บาท)
   - Valid: ความถูกต้องของคำตอบ

2. **Save JSON**: บันทึกเป็น `vrp_all_sheets_summary_v2.json`

**ตัวอย่าง Output**:
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

---

## บทสรุป

### สรุปการทำงานของระบบ

VRP Solver v2 เป็นระบบที่ออกแบบมาเพื่อแก้ปัญหา Vehicle Routing Problem ที่มีความซับซ้อน โดยมีขั้นตอนการทำงานหลักดังนี้:

1. **Data Loading**: อ่านและแปลงข้อมูลจาก Excel เป็น data structures ที่เหมาะสม
2. **Problem Formulation**: สร้าง OR-Tools model พร้อม constraints และ objective function
3. **Optimization**: ใช้ OR-Tools หรือ Heuristic ในการหาคำตอบ
4. **Post-Processing**: เพิ่ม checkpoint เข้าไปในเส้นทาง
5. **Validation**: ตรวจสอบความถูกต้องของคำตอบ
6. **Output Generation**: บันทึกผลลัพธ์ในรูปแบบ JSON

### จุดเด่นของการออกแบบ

1. **Modular Design**: แยกส่วนการทำงานออกเป็น modules ที่ชัดเจน
2. **Robust Error Handling**: มี fallback solver เมื่อ OR-Tools ล้มเหลว
3. **Comprehensive Validation**: ตรวจสอบความถูกต้องอย่างละเอียด
4. **Clear Data Structures**: ใช้ dataclasses สำหรับความชัดเจน
5. **Flexible Output**: บันทึกผลลัพธ์ในรูปแบบ JSON ที่อ่านง่าย

### แนวทางการพัฒนาต่อ

1. สามารถเพิ่มประเภทรถหลายแบบในการแก้ปัญหา
2. สามารถเพิ่ม constraints ประเภท time windows
3. สามารถปรับปรุง heuristic algorithm ให้มีประสิทธิภาพมากขึ้น
4. สามารถเพิ่ม visualization ของเส้นทางที่ได้
