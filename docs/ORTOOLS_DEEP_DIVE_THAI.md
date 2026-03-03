# Google OR-Tools: ความเข้าใจเชิงลึกสำหรับปัญหา Vehicle Routing

## สารบัญ

1. [ภาพรวมของ OR-Tools](#1-ภาพรวมของ-or-tools)
2. [สถาปัตยกรรมของ OR-Tools](#2-สถาปัตยกรรมของ-or-tools)
3. [Routing API อย่างละเอียด](#3-routing-api-อย่างละเอียด)
4. [Constraint Programming](#4-constraint-programming)
5. [First Solution Strategies](#5-first-solution-strategies)
6. [Local Search Metaheuristics](#6-local-search-metaheuristics)
7. [Search Parameters](#7-search-parameters)
8. [การทำงานภายในของ OR-Tools](#8-การทำงานภายในของ-or-tools)
9. [การปรับแต่งประสิทธิภาพ](#9-การปรับแต่งประสิทธิภาพ)
10. [ข้อจำกัดและข้อดี](#10-ข้อจำกัดและข้อดี)

---

## 1. ภาพรวมของ OR-Tools

### 1.1 OR-Tools คืออะไร?

**Google OR-Tools** (Operations Research Tools) คือ Open-source software suite สำหรับแก้ปัญหา Combinatorial Optimization ที่พัฒนาโดย Google

```
OR-Tools = Optimization + Constraint Programming + Routing
```

### 1.2 ความสามารถหลัก

| ความสามารถ | คำอธิบาย | ใช้ใน VRP หรือไม่ |
|------------|----------|-----------------|
| Linear Optimization | แก้ปัญหา Linear Programming | ❌ |
| Constraint Programming | แก้ปัญหาด้วย Constraints | ✅ |
| Vehicle Routing | แก้ปัญหาเส้นทางรถ | ✅ |
| Graph Algorithms | อัลกอริทึมกราฟ | ✅ (ภายใน) |

### 1.3 ทำไมต้องใช้ OR-Tools สำหรับ VRP?

**ข้อดี:**
- ⚡ **ประสิทธิภาพสูง**: เขียนด้วย C++ มี Python binding
- 🎯 **เฉพาะทาง**: ออกแบบมาสำหรับ Routing Problem
- 🔧 **ยืดหยุ่น**: รองรับ Constraints หลากหลาย
- 📊 **Scalable**: รองรับปัญหาขนาดใหญ่ (1,000+ nodes)
- 🆓 **ฟรี**: Open-source ใช้งานได้ฟรี

**ข้อเปรียบเทียบ:**

| วิธี | ความเร็ว | คุณภาพ | ความยาก |
|------|---------|-------|---------|
| OR-Tools | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Genetic Algorithm | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Simulated Annealing | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Ant Colony | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 2. สถาปัตยกรรมของ OR-Tools

### 2.1 โครงสร้างหลัก

```
OR-Tools Routing Engine
├── Routing Model (core)
│   ├── Node Index Manager
│   ├── Dimension (capacity, time, distance)
│   └── Callback Functions
├── Constraint Solver
│   ├── Decision Builder
│   ├── Search Monitor
│   └── Local Search Operators
└── Search Strategy
    ├── First Solution Strategy
    └── Local Search Metaheuristic
```

### 2.2 การสร้าง Routing Model

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# สร้าง Routing Model
manager = pywrapcp.RoutingIndexManager(
    num_nodes,      # จำนวน node ทั้งหมด
    num_vehicles,   # จำนวนรถ
    depot           # node จุดเริ่มต้น/สิ้นสุด
)

routing = pywrapcp.RoutingModel(manager)
```

**สิ่งที่เกิดขึ้นภายใน:**

1. **Index Manager**: แปลงระหว่าง node indices (0,1,2,...) กับ internal indices
2. **Routing Model**: สร้างโครงสร้างกราฟและ variables
3. **Solver**: สร้าง constraint solver instance

### 2.3 Node Index Manager

**ทำไมต้องมี Index Manager?**

```
User Indices (External):     [0, 1, 2, 3, 4, ..., N]
                               ↓ แปลง
Internal Indices (Solver):    [0, 1, 2, 3, 4, ..., N+starts-1]
```

**เหตุผล:**
- จัดการ multiple depots
- จัดการ รถหลายคันที่เริ่มต้นคนละจุด
- Optimizations ภายใน solver

**ตัวอย่างการใช้งาน:**

```python
# User node 5 -> Internal index
index = manager.NodeToIndex(5)

# Internal index -> User node
node = manager.IndexToNode(index)
```

---

## 3. Routing API อย่างละเอียด

### 3.1 Distance Callback

**Distance Callback** คือฟังก์ชันที่บอกระยะทางระหว่างสอง node

```python
def distance_callback(from_index, to_index):
    # แปลง internal index เป็น user node
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)

    # คืนค่าระยะทางจาก distance matrix
    return distance_matrix[from_node][to_node]

# ลงทะเบียน callback กับ routing model
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
```

**ขั้นตอนภายใน OR-Tools:**

```
1. Solver เรียกใช้ callback เมื่อต้องการระยะทาง
2. Callback คำนวณระยะทาง
3. ผลลัพธ์ถูกเก็บใน cache (ถ้ามี)
4. Solver ใช้ระยะทางนี้ในการคำนวณ constraints
```

### 3.2 Dimension คืออะไร?

**Dimension** คือค่าที่สะสมขณะเดินทาง เช่น:
- **Distance Dimension**: ระยะทางรวม
- **Capacity Dimension**: ปริมาณของบนรถ
- **Time Dimension**: เวลาที่ใช้

```python
# สร้าง Distance Dimension
routing.AddDimension(
    transit_callback_index,    # callback สำหรับระยะทาง
    0,                         # slack สูงสุด (0 = ไม่มี)
    3000,                      # ระยะทางสูงสุดต่อรถ
    True,                      # start cumul to zero
    'Distance'                 # ชื่อ dimension
)

# สร้าง Capacity Dimension
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    return demands[node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,     # callback สำหรับ demand
    0,                         # slack สูงสุด
    [2000, 2000],             # ความจุรถแต่ละคัน
    True,                      # start cumul to zero
    'Capacity'                 # ชื่อ dimension
)
```

**โครงสร้าง Dimension:**

```
Dimension แต่ละตัวมี:
- Cumul Variable: ค่าสะสมณ จุดแต่ละจุด
- Span Variable: ค่าที่เพิ่มขึ้นระหว่างจุด
- Slack Variable: ค่าที่เพิ่มเข้ามาได้ (เวลารอ)
```

### 3.3 Penalties และ Drop Nodes

**Penalty** ใช้เมื่ออนุญาตให้ข้าม node บางจุด

```python
# กำหนด penalty สำหรับแต่ละ node
penalties = [1000] * num_nodes
penalties[0] = 0  # depot ไม่มี penalty

# node ที่มี penalty สูงจะถูกเยี่ยมแน่นอน
# node ที่มี penalty ต่ำอาจถูกข้ามถ้าไม่คุ้ม
```

**วิธีคำนวณ:**

```
ถ้าเยี่ยม node: ไม่มีค่าใช้จ่ายเพิ่ม
ถ้าข้าม node:   เพิ่ม penalty เข้า total cost
```

---

## 4. Constraint Programming

### 4.1 หลักการ Constraint Programming

**Constraint Programming (CP)** คือการหาคำตอบโดย:

```
1. กำหนด Variables (ตัวแปร)
2. กำหนด Domains (โดเมนของค่าที่เป็นไปได้)
3. กำหนด Constraints (ข้อจำกัด)
4. ใช้ Search เพื่อหาคำตอบที่สอดคล้อง constraints
```

### 4.2 Constraints ใน VRP

**1. Next Node Constraint**

```
ทุก node (ยกเว้น depot ปลายทาง) ต้องมี node ถัดไป
ทุก node (ยกเว้น depot ต้นทาง) ต้องมี node ก่อนหน้า
```

**2. Path Constraint**

```
เส้นทางต้องเป็น path ที่ต่อกัน (ไม่มีวงวนย่อย)
```

**3. Capacity Constraint**

```
สำหรับแต่ละเส้นทาง:
    ผลรวม demand ≤ ความจุรถ
```

**4. Depot Constraint**

```
ทุกรถต้องเริ่มที่ depot
ทุกรถต้องจบที่ depot
```

### 4.3 Constraint Propagation

**Constraint Propagation** คือการลดโดเมนของตัวแปร:

```
ตัวอย่าง:
Node 3 มี demand 500
รถมีความจุ 2000
ถ้ารถบรรทุกแล้ว 1800
-> Node 3 ไม่สามารถไปต่อได้
-> ลดโดเมนของตัวแปรถัดไป
```

**ขั้นตอน:**

```
1. Solver เพิ่ม constraint
2. Propagation engine กระจายผลกระทบ
3. ลดโดเมนของตัวแปรที่เกี่ยวข้อง
4. ทำซ้ำจนไม่สามารถลดได้อีก
```

---

## 5. First Solution Strategies

### 5.1 ทำไมต้องมี First Solution?

**Problem:** VRP เป็น NP-Hard ไม่สามารถหาคำตอบที่ดีที่สุดโดยตรง

**Solution:** หาคำตอบเริ่มต้น (Initial Solution) แล้วปรับปรุง

```
Initial Solution → Local Search → Better Solution
```

### 5.2 ประเภทของ First Solution Strategies

| Strategy | คำอธิบาย | ความเร็ว | คุณภาพ |
|----------|----------|---------|--------|
| PATH_CHEAPEST_ARC | เลือกเส้นทางที่ถูกที่สุด | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| PATH_MOST_CONSTRAINED_ARC | เลือกที่มี constraints มากที่สุด | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SAVINGS | ใช้ขั้นตอนวิธี Savings | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| CHRISTOFIDES | สำหรับ TSP | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| BEST_INSERTION | แทรก node ที่ดีที่สุด | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 5.3 PATH_CHEAPEST_ARC (ที่ใช้ในโปรเจกต์)

**อัลกอริทึม:**

```
1. เริ่มที่ depot (node 0)
2. หา node ถัดไปที่มีระยะทางน้อยที่สุด
3. ตรวจสอบ capacity constraints
4. ถ้าผ่าน → ไป node นั้น
5. ถ้าไม่ผ่าน → กลับ depot เริ่มเส้นทางใหม่
6. ทำซ้ำจนครบทุก node
```

**ตัวอย่าง:**

```
Depot (0)
  ↓ เลือก node ที่ใกล้สุด = 2 (ระยะ 100m)
Node 2 (load: 500)
  ↓ เลือก node ที่ใกล้สุด = 5 (ระยะ 150m)
Node 5 (load: 1200) ← รวม 1700 < 2000 ✓
  ↓ เลือก node ที่ใกล้สุด = 8 (ระยะ 200m)
Node 8 (load: 600) ← รวม 2300 > 2000 ✗
  ↓ กลับ depot
Depot (เริ่มเส้นทางใหม่)
  ...
```

### 5.4 การเลือก Strategy

```python
first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

# หรือใช้ค่าอื่น:
# routing_enums_pb2.FirstSolutionStrategy.SAVINGS
# routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION
```

---

## 6. Local Search Metaheuristics

### 6.1 หลักการ Local Search

**Local Search** คือการปรับปรุงคำตอบโดย:

```
1. เริ่มจากคำตอบเริ่มต้น
2. สร้าง "Neighbor" (คำตอบใกล้เคียง)
3. ประเมินคำตอบ
4. ถ้าดีกว่า → ย้ายไปคำตอบนั้น
5. ทำซ้ำจนถึงเงื่อนไขหยุด
```

### 6.2 Local Search Operators

**1. Relocate**

```
ย้าย node จากตำแหน่งหนึ่งไปอีกตำแหน่ง:

ก่อน: 0 → 1 → 2 → 3 → 4 → 0
หลัง: 0 → 1 → 3 → 2 → 4 → 0
               (ย้าย 2 ไปหลัง 3)
```

**2. Exchange**

```
สลับสอง node:

ก่อน: 0 → 1 → 2 → 3 → 4 → 0
หลัง: 0 → 3 → 2 → 1 → 4 → 0
             (สลับ 1 กับ 3)
```

**3. Two Opt**

```
ตัดเส้นทางสองเส้นแล้วเชื่อมใหม่:

ก่อน: 0 → 1 → 2 → 3 → 4 → 0
       └───────────┘
              ↓
หลัง: 0 → 1 → 4 → 3 → 2 → 0
```

**4. Or Opt**

```
ย้ายลำดับ node ต่อเนื่อง:

ก่อน: 0 → 1 → 2 → 3 → 4 → 5 → 0
หลัง: 0 → 4 → 5 → 1 → 2 → 3 → 0
             (ย้าย 4-5 ไปข้างหน้า)
```

### 6.3 GUIDED_LOCAL_SEARCH (ที่ใช้ในโปรเจกต์)

**Guided Local Search (GLS)** ใช้ "Penalties" เพื่อหลีกเลี่ยง Local Optima:

```
1. เริ่มด้วย Local Search ธรรมดา
2. เมื่อติด Local Optima:
   - เพิ่ม penalty ให้ features ที่ทำให้ค่าใช้จ่ายสูง
   - เปลี่ยน landscape ของปัญหา
3. ทำ Local Search ต่อบน landscape ใหม่
```

**Feature ที่ถูก penalize:**
- เส้นทางที่มีระยะทางไกล
- การใช้รถหลายคัน
- การละเมิด constraints (soft constraints)

**ข้อดีของ GLS:**
- หลีกเลี่ยง Local Optima ได้ดี
- ไม่ต้องปรับ parameters มาก
- เหมาะกับ VRP

### 6.4 Metaheuristics อื่นๆ

| Metaheuristic | คำอธิบาย | เหมาะกับ |
|--------------|----------|----------|
| SIMULATED_ANNEALING | ยอมรับคำตอบที่แย่กว่าด้วยความน่าจะเป็น | หลีกเลี่ยง local optima |
| TABU_SEARCH | เก็บประวัติเพื่อไม่กลับไปคำตอบเดิม | หลากหลายของคำตอบ |
| GENERIC_TABU_SEARCH | Tabu search แบบปรับแต่งได้ | ปัญหาซับซ้อน |
| OBJECTIVE_TABU_SEARCH | Tabu บน objective space | หลาย objectives |

---

## 7. Search Parameters

### 7.1 การตั้งค่า Parameters หลัก

```python
search_parameters = pywrapcp.DefaultRoutingSearchParameters()

# First Solution Strategy
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

# Local Search Metaheuristic
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)

# Time Limit (ms)
search_parameters.time_limit.seconds = 30

# Solution Limit
search_parameters.solution_limit = 100

# Use LNS (Large Neighborhood Search)
search_parameters.use_full_propagation = True
```

### 7.2 Parameters สำคัญ

| Parameter | ค่าเริ่มต้น | คำอธิบาย | ผลกระทบ |
|-----------|-------------|----------|---------|
| time_limit.seconds | ไม่จำกัด | เวลาสูงสุด (วินาที) | ยิ่งนาน ยิ่งดี |
| solution_limit | ไม่จำกัด | จำนวนคำตอบสูงสุด | หยุดเมื่อครบ |
| use_cp | True | ใช้ CP solver | เร็วขึ้น |
| use_full_propagation | False | Propagation เต็ม | คุณภาพดีขึ้น |
| guiding_neighbors_solution | False | ใช้คำตอบข้างเคียง | หลากหลาย |

### 7.3 การตั้งค่า LNS (Large Neighborhood Search)

```python
search_parameters.local_search_operators.use_path_lns = True
search_parameters.local_search_operators.use_inactive_lns = True
```

**Path LNS:**
- ลบส่วนของเส้นทาง
- ใส่กลับด้วยวิธีที่ดีกว่า

**Inactive LNS:**
- ลบ node ออกชั่วคราว
- แทรกกลับด้วยตำแหน่งที่ดีกว่า

---

## 8. การทำงานภายในของ OR-Tools

### 8.1 ขั้นตอนการแก้ปัญหา

```
┌─────────────────────────────────────┐
│ 1. สร้าง Model                     │
│    - Create variables               │
│    - Add constraints                │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. First Solution                   │
│    - PATH_CHEAPEST_ARC              │
│    - สร้าง initial solution        │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. Local Search                     │
│    - GUIDED_LOCAL_SEARCH            │
│    - ปรับปรุงคำตอบ               │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. Return Best Solution             │
│    - คำตอบที่ดีที่สุดที่พบ      │
└─────────────────────────────────────┘
```

### 8.2 Decision Builder

**Decision Builder** คือกลไกหลักในการสร้างคำตอบ:

```
Decision Builder = Variables + Value Selector + Strategy
```

**ใน VRP:**
- **Variables**: Next node ของแต่ละ node
- **Value Selector**: ตัวเลือก node ถัดไป
- **Strategy**: First Solution Strategy

### 8.3 Search Monitors

**Search Monitors** ตรวจสอบและควบคุมการค้นหา:

```
Types of Monitors:
- Solution Monitor: เก็บคำตอบที่ดีที่สุด
- Time Monitor: หยุดเมื่อหมดเวลา
- Limit Monitor: หยุดเมื่อถึง limit
- Improvement Monitor: ติดตามการปรับปรุง
```

### 8.4 Callback Execution Flow

```
1. Solver ต้องการระยะทางระหว่าง node A และ node B
2. เรียก distance_callback(A, B)
3. Callback:
   - แปลง internal index → user node
   - ดึงระยะทางจาก distance_matrix
   - คืนค่าระยะทาง
4. Solver ใช้ค่านี้:
   - คำนวณ total distance
   - ตรวจสอบ constraints
   - ตัดสินใจเลือกเส้นทาง
```

---

## 9. การปรับแต่งประสิทธิภาพ

### 9.1 การเลือก Strategy ที่เหมาะสม

**สำหรับปัญหาเล็ก (< 50 nodes):**

```python
first_solution_strategy = PATH_CHEAPEST_ARC
local_search_metaheuristic = GUIDED_LOCAL_SEARCH
time_limit = 10-30 seconds
```

**สำหรับปัญหากลาง (50-100 nodes):**

```python
first_solution_strategy = SAVINGS
local_search_metaheuristic = GUIDED_LOCAL_SEARCH
time_limit = 30-60 seconds
use_full_propagation = True
```

**สำหรับปัญหาใหญ่ (> 100 nodes):**

```python
first_solution_strategy = SAVINGS
local_search_metaheuristic = GUIDED_LOCAL_SEARCH
time_limit = 60-120 seconds
use_full_propagation = True
use_path_lns = True
use_inactive_lns = True
```

### 9.2 การปรับปรุง Distance Matrix

**ใช้ Sparse Matrix:**

```python
# ถ้าระยะทางหลายค่าเป็น 0 (ไม่สามารถไปถึง)
# ใช้ sparse matrix เพื่อประหยัด memory
```

**Pre-compute Distances:**

```python
# คำนวณระยะทางล่วงหน้า
# ใช้ Euclidean, Manhattan, หรือ real road network
```

### 9.3 การใช้ Parallel Processing

```python
# OR-Tools ไม่รองรับ parallel search โดยตรง
# แต่สามารถ:
# 1. แก้ปัญหาย่อยแบบ parallel
# 2. ทดลอง parameters แบบ parallel
# 3. ใช้ multi-start แบบ parallel
```

---

## 10. ข้อจำกัดและข้อดี

### 10.1 ข้อจำกัดของ OR-Tools

**1. Open Source**
- ไม่มี commercial support
- ต้องแก้ bug เอง

**2. Single Objective**
- เน้น objective เดียว (ส่วนใหญ่คือ distance)
- หลาย objectives ต้อง encode เป็น weighted sum

**3. Static Problem**
- ไม่รองรับ dynamic VRP (ความต้องการเปลี่ยนแปลง)
- ต้องแก้ปัญหาใหม่เมื่อมีการเปลี่ยนแปลง

**4. Memory Usage**
- Large problems (> 10,000 nodes) ใช้ memory มาก
- Distance matrix ขนาด N x N

### 10.2 ข้อดีของ OR-Tools

**1. Performance**
- เร็วมากสำหรับ problems ขนาดกลาง
- มีหลาย strategies ให้เลือก

**2. Flexibility**
- รองรับ constraints หลากหลาย
- สามารถเพิ่ม custom callbacks

**3. Reliability**
- ใช้โดยบริษัทใหญ่หลายแห่ง
- Test อย่างดี

**4. Community**
- มี community ขนาดใหญ่
- เอกสารดีมาก

### 10.3 เปรียบเทียบกับวิธีอื่น

| วิธี | ความยาก | Performance | Scalability | Flexibility |
|------|---------|-------------|-------------|-------------|
| OR-Tools | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Custom GA | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Commercial | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| ACO/PSO | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## สรุป

### สิ่งสำคัญที่ต้องเข้าใจ:

1. **OR-Tools เป็น Constraint Solver** ไม่ใช่แค่ heuristic
2. **Index Manager** สำคัญมากสำหรับการจัดการ nodes
3. **Dimensions** ใช้สำหรับ track ค่าสะสมต่างๆ
4. **First Solution** + **Local Search** = คำตอบที่ดี
5. **GUIDED_LOCAL_SEARCH** เหมาะกับ VRP
6. **Callbacks** คือหัวใจของ flexibility
7. **Parameters** ต้องปรับตามขนาดปัญหา

### สำหรับ VRP Solver v2:

```
✅ ใช้ PATH_CHEAPEST_ARC → เร็ว ง่าย
✅ ใช้ GUIDED_LOCAL_SEARCH → หลีกเลี่ยง local optima
✅ ใช้ Capacity Dimension → รองรับ constraints
✅ Post-processing drop node → เพิ่มจุดทิ้ง
```

### แหล่งข้อมูลเพิ่มเติม:

- [OR-Tools Documentation](https://developers.google.com/optimization)
- [VRP Tutorial](https://developers.google.com/optimization/routing/vrp)
- [GitHub Repository](https://github.com/google/or-tools)
