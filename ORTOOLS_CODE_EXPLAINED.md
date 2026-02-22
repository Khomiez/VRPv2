# คู่มือโค้ด OR-Tools สำหรับ VRP Solver

## สารบัญ
1. [ภาพรวม OR-Tools](#ภาพรวม-or-tools)
2. [การติดตั้งและนำเข้า](#การติดตั้งและนำเข้า)
3. [โครงสร้างหลักของ OR-Tools](#โครงสร้างหลักของ-or-tools)
4. [ขั้นตอนการแก้ปัญหา VRP](#ขั้นตอนการแก้ปัญหา-vrp)
5. [โค้ดสำคัญแต่ละส่วน](#โค้ดสำคัญแต่ละส่วน)
6. [การปรับแต่งพารามิเตอร์](#การปรับแต่งพารามิเตอร์)
7. [ตัวอย่างการนำไปใช้](#ตัวอย่างการนำไปใช้)

---

## ภาพรวม OR-Tools

### OR-Tools คืออะไร?

**OR-Tools (Operations Research Tools)** คือไลบรารีสำหรับแก้ปัญหา Optimization จาก Google ที่ใช้:
- **Constraint Programming**: กำหนดเงื่อนไขและข้อจำกัด
- **Metaheuristics**: อัลกอริทึมการค้นหาคำตอบที่ดีที่สุด

### ทำไมต้องใช้ OR-Tools สำหรับ VRP?

ปัญหา Vehicle Routing Problem (VRP) เป็นปัญหา **NP-Hard** ที่:
- ยากที่จะหาคำตอบที่ดีที่สุด (Optimal) ในเวลาอันสั้น
- มีเงื่อนไขจำนวนมาก (Capacity, Distance, Time, etc.)
- OR-Tools ช่วยหาคำตอบที่ **ดีมาก (Near-Optimal)** ในเวลาสั้น

---

## การติดตั้งและนำเข้า

### การติดตั้ง

```bash
pip install ortools
```

### การนำเข้าในโค้ด (บรรทัด 30-36)

```python
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: OR-Tools not installed. Install with: pip install ortools")
```

**อธิบาย**:
- `routing_enums_pb2`: Enumerations สำหรับกลยุทธ์การแก้ปัญหา
- `pywrapcp`: Python wrapper สำหรับ Constraint Solver
- ใช้ `try-except` เพื่อให้ระบบทำงานได้แม้ไม่มี OR-Tools (จะใช้ Heuristic แทน)

---

## โครงสร้างหลักของ OR-Tools

### ส่วนประกอบสำคัญ

```
OR-Tools VRP Structure:
├── RoutingIndexManager    → จัดการการแปลง Node Index
├── RoutingModel           → โมเดลปัญหา VRP
├── Callbacks              → ฟังก์ชันคำนวณระยะห่างและความจุ
├── Dimensions             → กำหนด Constraints (Capacity, Time, etc.)
├── Search Parameters     → ตั้งค่าอัลกอริทึม
└── Assignment             → คำตอบที่ได้
```

### Index System ของ OR-Tools

⚠️ **สำคัญมาก**: OR-Tools ใช้ 2 ระบบ Index

```python
# Node Index = ตำแหน่งใน distance_matrix (0, 1, 2, ..., N-1)
# Routing Index = Index ภายใน OR-Tools (แปลงโดย Manager)

# ตัวอย่าง:
node_index = 5          # ใน distance_matrix
routing_index = manager.NodeToIndex(5)    # แปลงเป็น routing index
node_index = manager.IndexToNode(routing_index)  # แปลงกลับ
```

---

## ขั้นตอนการแก้ปัญหา VRP

### ภาพรวมการทำงาน (Flowchart)

```
1. สร้าง RoutingIndexManager
   ↓
2. สร้าง RoutingModel
   ↓
3. กำหนด Distance Callback
   ↓
4. กำหนด Demand Callbacks (General + Recycle)
   ↓
5. เพิ่ม Capacity Dimensions
   ↓
6. กำหนด Search Parameters
   ↓
7. Solve() → ได้ Assignment
   ↓
8. ดึงคำตอบจาก Assignment
```

---

## โค้ดสำคัญแต่ละส่วน

### ส่วนที่ 1: สร้าง RoutingIndexManager (บรรทัด 290)

```python
# สร้าง Manager สำหรับจัดการ Index
manager = pywrapcp.RoutingIndexManager(
    num_nodes,      # จำนวน Node ทั้งหมด
    num_vehicles,   # จำนวนรถ
    depot_idx       # Index ของ Depot (Node 1 = Index 0)
)
```

**อธิบาย**:
- `num_nodes`: จำนวนจุดเก็บขยะทั้งหมด (เช่น 20, 30, 50, 80, 138)
- `num_vehicles`: จำนวนรถที่คำนวณได้จากความจุรวม
- `depot_idx`: จุดเริ่มต้นและสิ้นสุด (เสมอเป็น Node 1 = Index 0)

**ทำไมต้องมี Manager?**
```python
# Manager ช่วยแปลงระหว่าง 2 ระบบ Index:
# 1. Node Index (0, 1, 2, ..., N-1) → ใช้กับ distance_matrix
# 2. Routing Index → ใช้ภายใน OR-Tools

# ตัวอย่างการใช้งาน:
routing_idx = manager.NodeToIndex(5)     # Node 5 → Routing Index
node_idx = manager.IndexToNode(routing_idx)  # Routing Index → Node 5
```

---

### ส่วนที่ 2: สร้าง RoutingModel (บรรทัด 291)

```python
# สร้าง Model หลัก
routing = pywrapcp.RoutingModel(manager)
```

**อธิบาย**:
- `RoutingModel` เป็นโมเดลหลักที่เก็บปัญหา VRP ทั้งหมด
- เก็บ Variables, Constraints, และ Objective Function
- ทุกอย่างต้องผ่าน `routing` object นี้

---

### ส่วนที่ 3: Distance Callback (บรรทัด 293-300)

```python
# กำหนดฟังก์ชันคำนวณระยะห่าง
def distance_callback(from_index, to_index):
    """
    คำนวณระยะห่างระหว่างสองจุด

    Args:
        from_index: Routing Index ของจุดเริ่มต้น
        to_index: Routing Index ของจุดปลายทาง

    Returns:
        ระยะห่าง (เมตร)
    """
    # แปลง Routing Index → Node Index
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)

    # คืนค่าระยะห่างจาก distance_matrix
    return int(distance_matrix[from_node][to_node])

# ลงทะเบียน Callback กับ OR-Tools
transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# กำหนดให้ใช้ Distance Callback เป็นต้นทุน (Arc Cost)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
```

**อธิบายแต่ละบรรทัด**:

1. **`def distance_callback(from_index, to_index)`**:
   - OR-Tools เรียกฟังก์ชันนี้เมื่อต้องการรู้ระยะห่าง
   - รับค่าเป็น **Routing Index** ไม่ใช่ Node Index

2. **`manager.IndexToNode(from_index)`**:
   - แปลง Routing Index → Node Index
   - เพื่อใช้กับ `distance_matrix`

3. **`distance_matrix[from_node][to_node]`**:
   - ดึงระยะห่างจาก Matrix
   - ค่าเป็นเมตร (meters)

4. **`routing.RegisterTransitCallback(distance_callback)`**:
   - ลงทะเบียนฟังก์ชันกับ OR-Tools
   - คืนค่า Index ของ Callback

5. **`routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)`**:
   - กำหนดให้ Distance = Cost (Objective Function)
   - OR-Tools จะพยายาม **ลดระยะทางรวม**

**⚠️ ข้อควรระวัง**:
```python
# ❌ ผิด: ใช้ Routing Index กับ distance_matrix โดยตรง
return distance_matrix[from_index][to_index]

# ✅ ถูก: แปลงเป็น Node Index ก่อน
from_node = manager.IndexToNode(from_index)
to_node = manager.IndexToNode(to_index)
return distance_matrix[from_node][to_node]
```

---

### ส่วนที่ 4: Demand Callbacks (บรรทัด 302-314)

#### เตรียมข้อมูล Demand (บรรทัด 305-306)

```python
# แยกข้อมูล Demand เป็น 2 ประเภท
general_demands = [int(n.general_demand) for n in nodes]
recycle_demands = [int(n.recycle_demand) for n in nodes]

# ตัวอย่าง:
# general_demands = [100, 50, 80, 0, 120, ...]
# recycle_demands = [20, 10, 15, 0, 25, ...]
#                   ↑   ↑   ↑  ↑   ↑
#                   1   2   3  4   5  (Node IDs)
```

#### General Demand Callback (บรรทัด 308-310)

```python
def demand_general_callback(from_index):
    """
    คำนวณปริมาณขยะทั่วไปที่รถเก็บได้ที่จุดนั้น

    Args:
        from_index: Routing Index ของจุดที่กำลังไป

    Returns:
        ปริมาณขยะทั่วไป (กก.)
    """
    from_node = manager.IndexToNode(from_index)
    return general_demands[from_node]
```

#### Recycle Demand Callback (บรรทัด 312-314)

```python
def demand_recycle_callback(from_index):
    """
    คำนวณปริมาณขยะรีไซเคิลที่รถเก็บได้ที่จุดนั้น
    """
    from_node = manager.IndexToNode(from_index)
    return recycle_demands[from_node]
```

**ทำไมต้องมี 2 Callback?**
- เพราะมี **2 ความจุ** ที่ต้องเคร่งครัด:
  - `general_capacity`: ความจุขยะทั่วไป
  - `recycle_capacity`: ความจุขยะรีไซเคิล
- รถต้องไม่เกินทั้ง 2 ความจุพร้อมกัน

---

### ส่วนที่ 5: Capacity Dimensions (บรรทัด 316-334)

#### General Capacity Dimension (บรรทัด 317-324)

```python
# ลงทะเบียน General Demand Callback
demand_gen_callback_index = routing.RegisterUnaryTransitCallback(
    demand_general_callback
)

# เพิ่ม Dimension สำหรับความจุขยะทั่วไป
routing.AddDimensionWithVehicleCapacity(
    demand_gen_callback_index,    # Callback ที่คำนวณ Demand
    0,                             # Slack (ไม่อนุญาตให้เกินความจุ)
    [int(vehicle.general_capacity)] * num_vehicles,  # ความจุรถแต่ละคัน
    True,                          # Start cumulative to zero (เริ่มจาก 0)
    'GeneralCapacity'              # ชื่อ Dimension
)
```

**อธิบายแต่ละพารามิเตอร์**:

| พารามิเตอร์ | ค่า | อธิบาย |
|------------|-----|--------|
| `callback` | `demand_gen_callback_index` | ฟังก์ชันคำนวณปริมาณขยะ |
| `slack` | `0` | ไม่อนุญาตให้มี slack (เกินความจุไม่ได้) |
| `capacity` | `[2000, 2000, ...]` | ความจุรถแต่ละคัน (กก.) |
| `fix_start_cumul_to_zero` | `True` | เริ่มต้น cumulative จาก 0 |
| `name` | `'GeneralCapacity'` | ชื่อ Dimension |

#### Recycle Capacity Dimension (บรรทัด 327-334)

```python
# ลงทะเบียน Recycle Demand Callback
demand_rec_callback_index = routing.RegisterUnaryTransitCallback(
    demand_recycle_callback
)

# เพิ่ม Dimension สำหรับความจุขยะรีไซเคิล
routing.AddDimensionWithVehicleCapacity(
    demand_rec_callback_index,
    0,
    [int(vehicle.recycle_capacity)] * num_vehicles,
    True,
    'RecycleCapacity'
)
```

**⚠️ สำคัญ: Slack Parameter**

```python
# Slack = 0 หมายถึง: ห้ามเกินความจุอย่างเด็ดขาด
routing.AddDimensionWithVehicleCapacity(
    callback,
    0,  # ❌ ไม่มี slack = เกินไม่ได้เลย
    capacity,
    True,
    'Capacity'
)

# Slack > 0 หมายถึง: อนุญาตให้เกินความจุได้บางส่วน
routing.AddDimensionWithVehicleCapacity(
    callback,
    100,  # ✅ มี slack = อนุญาตเกินได้ 100 กก.
    capacity,
    True,
    'Capacity'
)
```

---

### ส่วนที่ 6: Checkpoint Handling (บรรทัด 336-339)

```python
# ทำให้ Checkpoint เป็น optional node
checkpoint_routing_idx = manager.NodeToIndex(checkpoint_idx)
routing.AddDisjunction([checkpoint_routing_idx], 0)
```

**อธิบาย**:

1. **`manager.NodeToIndex(checkpoint_idx)`**:
   - แปลง Node Index ของจุดทิ้งขยะ → Routing Index

2. **`routing.AddDisjunction([checkpoint_routing_idx], 0)`**:
   - `AddDisjunction`: สร้างกลุ่ม node ที่ OR-Tools เลือกไปได้หรือไม่ได้
   - `[checkpoint_routing_idx]`: กลุ่มที่มีเพียง checkpoint
   - `0`: Penalty = 0 (ไม่มีโทษเมื่อไม่ไป)

**ทำไมต้องทำ?**
```
เราต้องการให้ OR-Tools เน้นเก็บขยะก่อน
แล้วค่อยแทรก Checkpoint ในภายหลัง (Post-processing)

เพราะ:
- OR-Tools อาจยากที่จะบังคับ Checkpoint
- วิธีนี้ทำให้ได้คำตอบที่ดีกว่า
- เราจะแทรก Checkpoint เองในขั้นตอนถัดไป
```

---

### ส่วนที่ 7: Search Parameters (บรรทัด 341-349)

```python
# สร้าง Search Parameters เริ่มต้น
search_params = pywrapcp.DefaultRoutingSearchParameters()

# กำหนดกลยุทธ์คำตอบเริ่มต้น
search_params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

# กำหนด Metaheuristic สำหรับ Local Search
search_params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)

# กำหนดเวลาสูงสุด
search_params.time_limit.seconds = time_limit  # เช่น 60 วินาที
```

#### First Solution Strategies

```python
# กลยุทธ์สำหรับสร้างคำตอบแรก:

# 1. PATH_CHEAPEST_ARC (ใช้ในโค้ด)
#    → เริ่มจาก Depot แล้วไปจุดที่ใกล้สุดเสมอ
#    → เร็ว แต่อาจไม่ดีที่สุด

# 2. SAVINGS
#    → ใช้วิธี Savings Algorithm
#    → เหมาะกับปัญหา VRP แบบดั้งเดิม

# 3. CHRISTOFIDES
#    → ใช้ Christofides Algorithm
#    → ดีสำหรับ TSP (รถเดียว)

# 4. AUTOMATIC
#    → ให้ OR-Tools เลือกเอง
#    → ปลอดภัยที่สุด
```

#### Local Search Metaheuristics

```python
# กลยุทธ์สำหรับปรับปรุงคำตอบ:

# 1. GUIDED_LOCAL_SEARCH (ใช้ในโค้ด)
#    → ใช้ Guided Local Search
#    → ดุจยาก แต่ได้ผลดีมาก

# 2. SIMULATED_ANNEALING
#    → ใช้ Simulated Annealing
#    → หลีกเลี่ยง Local Optima ได้ดี

# 3. TABU_SEARCH
#    → ใช้ Tabu Search
#    → หลีกเลี่ยงการกลับไปที่เดิม

# 4. GREEDY_DESCENT
#    → ใช้วิธี Greedy
#    → เร็วที่สุด แต่อาจติด Local Optima
```

#### Time Limit

```python
# กำหนดเวลาสูงสุดในการแก้ปัญหา
search_params.time_limit.seconds = 60  # 60 วินาที

# ถ้าใส่ 0 = ไม่จำกัดเวลา (อาจใช้เวลานานมาก)
search_params.time_limit.seconds = 0

# แนะนำ:
# - ปัญหาเล็ก (20-30 nodes): 30-60 วินาที
# - ปัญหากลาง (50-80 nodes): 60-120 วินาที
# - ปัญหาใหญ่ (138 nodes): 120-300 วินาที
```

---

### ส่วนที่ 8: Solve (บรรทัด 351-355)

```python
# เริ่มแก้ปัญหา
assignment = routing.SolveWithParameters(search_params)

# ตรวจสอบว่าได้คำตอบหรือไม่
if not assignment:
    raise Exception("OR-Tools could not find a solution")
```

**อธิบาย**:

1. **`routing.SolveWithParameters(search_params)`**:
   - เริ่มกระบวนการแก้ปัญหา
   - ใช้ Search Parameters ที่กำหนด
   - คืนค่า `assignment` object

2. **`if not assignment:`**:
   - ตรวจสอบว่าได้คำตอบหรือไม่
   - ถ้าไม่ได้ ให้โยน Exception

**สิ่งที่เกิดขึ้นภายใน OR-Tools**:
```
1. สร้างคำตอบเริ่มต้น (Initial Solution)
   → ใช้ First Solution Strategy

2. ปรับปรุงคำตอบ (Local Search)
   → ใช้ Guided Local Search
   → ย้าย nodes ระหว่าง routes
   → สลับลำดับ nodes
   → พยายามลดระยะทางรวม

3. หยุดเมื่อ:
   → หมดเวลา (Time Limit)
   → หาคำตอบที่ดีที่สุดแล้ว
   → ไม่สามารถปรับปรุงได้อีก
```

---

### ส่วนที่ 9: ดึงคำตอบ (บรรทัด 357-428)

#### วนลูปแต่ละรถ (บรรทัด 362-410)

```python
routes = []
total_distance_m = 0
checkpoint_node_id = nodes[checkpoint_idx].id  # 1-indexed

# วนลูปแต่ละรถ
for vehicle_id in range(num_vehicles):
    # เริ่มจากต้นทางของรถคันนี้
    index = routing.Start(vehicle_id)
    route_nodes = []
    general_load = 0
    recycle_load = 0

    # เดินตามเส้นทางจนกว่าจะถึงปลายทาง
    while not routing.IsEnd(index):
        # แปลง Routing Index → Node Index
        node_idx = manager.IndexToNode(index)
        node = nodes[node_idx]

        # เพิ่ม node ลงใน route (ยกเว้น checkpoint)
        if node_idx != checkpoint_idx:
            route_nodes.append(node.id)  # 1-indexed
            general_load += node.general_demand
            recycle_load += node.recycle_demand

        # ไปยัง node ถัดไป
        index = assignment.Value(routing.NextVar(index))

    # ประมวลผลเฉพาะ route ที่ไม่ว่างเปล่า
    if len(route_nodes) > 1:
        # POST-PROCESSING: เพิ่ม checkpoint ก่อนกลับ depot
        route_nodes.append(checkpoint_node_id)
        route_nodes.append(1)  # กลับ Depot

        # คำนวณระยะทางจริง (รวม checkpoint)
        route_distance = 0.0
        for i in range(len(route_nodes) - 1):
            from_idx = route_nodes[i] - 1  # แปลงเป็น 0-indexed
            to_idx = route_nodes[i + 1] - 1
            route_distance += distance_matrix[from_idx][to_idx]

        # คำนวณต้นทุน
        distance_km = route_distance / 1000.0
        fuel_cost = distance_km * vehicle.fuel_cost_per_km

        # สร้าง Route object
        route = Route(
            vehicle_id=vehicle_id + 1,
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
        routes.append(route)
        total_distance_m += route_distance
```

**อธิบายสำคัญ**:

1. **`routing.Start(vehicle_id)`**:
   - คืนค่า Routing Index ของจุดเริ่มต้น (Depot)

2. **`while not routing.IsEnd(index)`**:
   - วนลูปจนกว่าจะถึงปลายทาง (Depot อีกครั้ง)

3. **`assignment.Value(routing.NextVar(index))`**:
   - คืนค่า Routing Index ถัดไป
   - `NextVar(index)` = Variable ที่เก็บ node ถัดไป
   - `assignment.Value(...)` = ค่าที่ OR-Tools แก้ได้

4. **Post-processing (บรรทัด 382-392)**:
   ```python
   # แทรก checkpoint ก่อนกลับ depot
   route_nodes.append(checkpoint_node_id)
   route_nodes.append(1)  # กลับ depot

   # คำนวณระยะทางใหม่ (รวม checkpoint)
   route_distance = 0.0
   for i in range(len(route_nodes) - 1):
       from_idx = route_nodes[i] - 1
       to_idx = route_nodes[i + 1] - 1
       route_distance += distance_matrix[from_idx][to_idx]
   ```

---

### ส่วนที่ 10: สรุปผลลัพธ์ (บรรทัด 412-428)

```python
# คำนวณระยะทางรวม
total_distance_km = total_distance_m / 1000.0

# คำนวณต้นทุนรวม
total_fixed = sum(r.fixed_cost for r in routes)
total_fuel = sum(r.fuel_cost for r in routes)

# สร้าง Solution object
return Solution(
    status='OPTIMAL' if assignment else 'INFEASIBLE',
    routes=routes,
    num_vehicles_used=len(routes),
    total_distance_meters=total_distance_m,
    total_distance_km=total_distance_km,
    total_fixed_cost=total_fixed,
    total_fuel_cost=total_fuel,
    total_cost=total_fixed + total_fuel,
    all_nodes_visited=False,  # จะตรวจสอบในภายหลัง
    all_routes_valid=False,
    validation_errors=[]
)
```

---

## การปรับแต่งพารามิเตอร์

### 1. เลือก First Solution Strategy

```python
# สำหรับปัญหา VRP แบบเก็บขยะ:
search_params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

# ตัวเลือกอื่น:
# - AUTOMATIC: ให้ OR-Tools เลือก
# - SAVINGS: ใช้ Savings Algorithm
# - CHRISTOFIDES: สำหรับ TSP
# - BEST_INSERTION: แทรก node ที่ดีที่สุด
```

### 2. เลือก Local Search Metaheuristic

```python
# สำหรับปัญหา VRP ขนาดกลาง-ใหญ่:
search_params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)

# ตัวเลือกอื่น:
# - SIMULATED_ANNEALING: หลีกเลี่ยง Local Optima
# - TABU_SEARCH: หลีกเลี่ยงการกลับไปเดิม
# - GREEDY_DESCENT: เร็วแต่อาจติด Local Optima
```

### 3. กำหนด Time Limit

```python
# ขนาดเล็ก (20-30 nodes):
search_params.time_limit.seconds = 30

# ขนาดกลาง (50-80 nodes):
search_params.time_limit.seconds = 60

# ขนาดใหญ่ (138 nodes):
search_params.time_limit.seconds = 120
```

### 4. ปรับ Slack Variable

```python
# ห้ามเกินความจุ:
routing.AddDimensionWithVehicleCapacity(
    callback, 0, capacity, True, 'Capacity'
)

# อนุญาตเกินได้ 10%:
slack = int(capacity * 0.1)
routing.AddDimensionWithVehicleCapacity(
    callback, slack, capacity + slack, True, 'Capacity'
)
```

---

## ตัวอย่างการนำไปใช้

### ตัวอย่างที่ 1: แก้ปัญหา VRP พื้นฐาน

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

# ข้อมูล
distance_matrix = np.array([
    [0, 1500, 2000, 3000],
    [1500, 0, 1200, 2500],
    [2000, 1200, 0, 1800],
    [3000, 2500, 1800, 0]
])
demands = [0, 100, 150, 80]
vehicle_capacity = 300
num_vehicles = 2
depot = 0

# สร้าง Manager
manager = pywrapcp.RoutingIndexManager(
    len(distance_matrix), num_vehicles, depot
)

# สร้าง Model
routing = pywrapcp.RoutingModel(manager)

# Distance Callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Demand Callback
def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return demands[from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index, 0, [vehicle_capacity] * num_vehicles, True, 'Capacity'
)

# Search Parameters
search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

# Solve
assignment = routing.SolveWithParameters(search_params)

# ดึงคำตอบ
if assignment:
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = assignment.Value(routing.NextVar(index))
        print(f"Vehicle {vehicle_id}: {route}")
```

### ตัวอย่างที่ 2: VRP พร้อม Checkpoint

```python
# ... โค้ดเดิมจนถึงสร้าง Model ...

checkpoint = 2  # Node 2 เป็น checkpoint

# ทำ checkpoint ให้เป็น optional
checkpoint_idx = manager.NodeToIndex(checkpoint)
routing.AddDisjunction([checkpoint_idx], 0)

# ... แก้ปัญหาเหมือนเดิม ...

# Post-processing: เพิ่ม checkpoint
if assignment:
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != checkpoint:  # ข้าม checkpoint
                route.append(node)
            index = assignment.Value(routing.NextVar(index))

        # เพิ่ม checkpoint กลับเข้าไป
        if len(route) > 1:
            route.append(checkpoint)
            route.append(0)  # กลับ depot
        print(f"Vehicle {vehicle_id}: {route}")
```

### ตัวอย่างที่ 3: VRM (Vehicle Routing with Multiple Capacities)

```python
# ... โค้ดเดิม ...

# 2 Capacities: General + Recycle
general_demands = [0, 100, 150, 80]
recycle_demands = [0, 20, 30, 15]
general_capacity = 300
recycle_capacity = 100

# General Capacity
def general_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return general_demands[from_node]

general_idx = routing.RegisterUnaryTransitCallback(general_callback)
routing.AddDimensionWithVehicleCapacity(
    general_idx, 0, [general_capacity] * num_vehicles, True, 'General'
)

# Recycle Capacity
def recycle_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return recycle_demands[from_node]

recycle_idx = routing.RegisterUnaryTransitCallback(recycle_callback)
routing.AddDimensionWithVehicleCapacity(
    recycle_idx, 0, [recycle_capacity] * num_vehicles, True, 'Recycle'
)

# ... แก้ปัญหาต่อ ...
```

---

## สรุป

### ขั้นตอนหลักในการใช้ OR-Tools สำหรับ VRP

1. **สร้าง RoutingIndexManager** → จัดการ Index
2. **สร้าง RoutingModel** → สร้างโมเดล
3. **กำหนด Distance Callback** → คำนวณระยะห่าง
4. **กำหนด Demand Callbacks** → คำนวณปริมาณขยะ
5. **เพิ่ม Capacity Dimensions** → กำหนดข้อจำกัดความจุ
6. **กำหนด Search Parameters** → ตั้งค่าอัลกอริทึม
7. **Solve()** → แก้ปัญหา
8. **ดึงคำตอบจาก Assignment** → ได้เส้นทาง

### จุดสำคัญที่ต้องจำ

1. **Index System**: OR-Tools ใช้ 2 ระบบ Index ที่ต้องแปลงกัน
2. **Callbacks**: ฟังก์ชันที่ OR-Tools เรียกเพื่อคำนวณค่าต่างๆ
3. **Dimensions**: กำหนด Constraints ต่างๆ (Capacity, Time, etc.)
4. **Search Parameters**: ควบคุมความเร็วและคุณภาพคำตอบ
5. **Post-processing**: แก้ไขคำตอบหลังจากได้จาก OR-Tools

### เอกสารอ้างอิงเพิ่มเติม

- [OR-Tools Documentation](https://developers.google.com/optimization)
- [OR-Tools VRP Examples](https://developers.google.com/optimization/routing/vrp)
- [Python API Reference](https://google.github.io/or-tools/python/annotated.html)

---

**หมายเหตุ**: เอกสารนี้อธิบายโค้ด OR-Tools ในไฟล์ `solvers/vrp_solver_v2.py` บรรทัด 274-428
