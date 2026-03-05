# คู่มือการตรวจสอบผลลัพธ์ด้วย NEOS Server สำหรับ VRP Solver

## สารบัญ
1. [ปัญหาที่คุณกำลังแก้คืออะไร?](#ปัญหาที่คุณกำลังแก้คืออะไร)
2. [ทำไมต้องตรวจสอบด้วย NEOS Server?](#ทำไมต้องตรวจสอบด้วย-neos-server)
3. [การจัดประเภทปัญหา](#การจัดประเภทปัญหา)
4. [ขั้นตอนการใช้ NEOS Server ทีละขั้นตอน](#ขั้นตอนการใช้-neos-server-ทีละขั้นตอน)
5. [การสร้างแบบจำลองสำหรับ NEOS](#การสร้างแบบจำลองสำหรับ-neos)
6. [การเปรียบเทียบผลลัพธ์](#การเปรียบเทียบผลลัพธ์)
7. [ตัวอย่างแบบจำลอง AMPL](#ตัวอย่างแบบจำลอง-ampl)
8. [ตัวอย่างแบบจำลอง GAMS](#ตัวอย่างแบบจำลอง-gams)

---

## ปัญหาที่คุณกำลังแก้คืออะไร?

ปัญหา VRP ของคุณเป็น **Capacitated Vehicle Routing Problem with Heterogeneous Fleet (CVRPHF)** หรือ "ปัญหาการจัดเส้นทางเดินรถโดยมีข้อจำกัดความจุและมีหลายประเภทรถ" พร้อมข้อจำกัดเพิ่มเติม:

### ลักษณะของปัญหา

| คุณลักษณะ | คำอธิบาย |
|------------|-----------|
| **ประเภทพื้นฐาน** | Capacitated Vehicle Routing Problem (CVRP) |
| **ประเภทรถ** | Heterogeneous Fleet (หลายประเภทรถ ความจุและต้นทุนต่างกัน) |
| **ความต้องการ | 2 ประเภท (ขยะทั่วไป + ขยะรีไซเคิล) |
| **เป้าหมาย** | ลดต้นทุนรวมให้น้อยที่สุด (ต้นทุนคงที่ + ต้นทุนน้ำมัน) |
| **ข้อจำกัด** | ความจุรถ, จุดทิ้งขยะ, จุดเริ่ม/สิ้นสุดที่ Depot |

### ข้อจำกัดหลักในแบบจำลองของคุณ

1. **โครงสร้างเส้นทาง**: `Depot (Node 1) → จุดเก็บขยะ → จุดทิ้ง → Depot`
2. **ข้อจำกัดความจุคู่**:
   - ความจุสำหรับขยะทั่วไปต่อรถ
   - ความจุสำหรับขยะรีไซเคิลต่อรถ
3. **ส่วนประกอบต้นทุน**:
   - ต้นทุนคงที่ต่อคัน (บาท/คัน)
   - ต้นทุนน้ำมันต่อกิโลเมตร (บาท/กม.)
4. **จุดเช็คพอยต์บังคับ**: ทุกเส้นทางต้องผ่านจุดทิ้งขยะก่อนกลับ Depot

---

## ทำไมต้องตรวจสอบด้วย NEOS Server?

**NEOS Server** (https://neos-server.org) เป็นบริการแก้ปัญหา Optimization ฟรีผ่านอินเทอร์เน็ตที่ให้:

- เข้าถึง Solvers เชิงพาณิชย์ (CPLEX, Gurobi, XPRESS)
- Solvers ภาควิชาการ (CBC, SCIP, GLPK)
- ภาษาสร้างแบบจำลองหลายภาษา (AMPL, GAMS, LP)
- ไม่ต้องติดตั้งโปรแกรม

### ประโยชน์ของการตรวจสอบ

| ประโยชน์ | คำอธิบาย |
|-----------|----------|
| **ความเป็นอิสระของ Solver** | ตรวจสอบผลลัพธ์ OR-Tools ด้วย Solvers ที่แตกต่างกัน |
| **ช่องว่างของความเหมาะสมที่สุด | ตรวจสอบว่าคำตอบของคุณเหมาะสมที่สุดจริงหรือไม่ |
| **การเปรียบเทียบ Benchmark** | เปรียบเทียบประสิทธิภาพหลาย Solvers |
| **การตรวจสอบแบบจำลอง** | มั่นใจว่าสูตรคณิตศาสตร์ถูกต้อง |

---

## การจัดประเภทปัญหาสำหรับ NEOS

เมื่อส่งไป NEOS ปัญหาของคุณสามารถจัดประเภทได้:

### หมวดหมู่หลัก

1. **Mixed Integer Linear Programming (MILP)**
   - หมวดหมู่: `milp` (Mixed Integer Linear Program)
   - สูตรที่แม่นยำที่สุด
   - Solvers: CPLEX, Gurobi, CBC, SCIP

2. **Integer Programming (IP)**
   - หมวดหมู่: `integer` (Integer Program)
   - สูตรที่เรียบง่ายกว่า
   - แก้เร็วกว่า

### คำแนะนำ Solvers ของ NEOS

| Solver | หมวดหมู่ | เหมาะสำหรับ | เวลาสูงสุด |
|--------|----------|--------------|--------------|
| **CPLEX** | MILP | ปัญหาขนาดใหญ่, คำตอบเหมาะสมที่สุด | 1 ชั่วโมง |
| **Gurobi** | MILP | ปัญหาขนาดใหญ่, คำตอบเหมาะสมที่สุด | 1 ชั่วโมง |
| **CBC** | MILP | ปัญหาขนาดกลาง, ฟรี/โอเพนซอร์ส | 30 นาที |
| **SCIP** | MILP | ข้อจำกัดซับซ้อน | 30 นาที |
| **GLPK** | MILP | ปัญหาขนาดเล็ก, สูตรเรียบง่าย | 15 นาที |

---

## ขั้นตอนการใช้ NEOS Server ทีละขั้นตอน

### ขั้นตอนที่ 1: เลือกภาษาสร้างแบบจำลอง

NEOS รองรับหลายภาษา เลือกตามความคุ้นเคย:

| ภาษา | ระดับความยาก | แนะนำ | นามสกุลไฟล์ |
|--------|----------------|---------|---------------|
| **AMPL** | ปานกลาง | ✅ ใช่ | `.mod` + `.dat` |
| **GAMS** | ปานกลาง | ✅ ใช่ | `.gms` |
| **LP** | ง่าย | ✅ ใช่ (แบบจำลองง่าย) | `.lp` |

### ขั้นตอนที่ 2: เตรียมไฟล์ข้อมูล

#### ตัวเลือก A: AMPL (แนะนำ)

สร้างสองไฟล์:

**ไฟล์ที่ 1: ไฟล์แบบจำลอง (`vrp_model.mod`)**
```ampl
# แบบจำลอง VRP
set NODES;                   # จุดทั้งหมด
set VEHICLES;                # ประเภทรถ
param depot {NODES} binary;  # 1 ถ้าเป็นจุดรวมรถ
param checkpoint {NODES} binary; # 1 ถ้าเป็นจุดทิ้ง

# เมทริกซ์ระยะทาง
param distance {i in NODES, j in NODES};

# ความต้องการ
param general_demand {NODES};
param recycle_demand {NODES};

# ข้อมูลรถ
param general_capacity {VEHICLES};
param recycle_capacity {VEHICLES};
param fixed_cost {VEHICLES};
param fuel_cost {VEHICLES};

# ตัวแปรตัดสินใจ
var x {NODES, NODES, VEHICLES}, binary;  # 1 ถ้ารถ k เดินทางจาก i ไป j
var u {NODES} >= 0;                      # กำจัด subtour

# ลดต้นทุนรวมให้น้อยสุด
minimize TotalCost:
  sum{k in VEHICLES} (fixed_cost[k] * sum{j in NODES} x[depot_node, j, k]) +
  sum{i in NODES, j in NODES, k in VEHICLES} fuel_cost[k] * distance[i,j] * x[i,j,k] / 1000;

# เงื่อนไขข้อจำกัด:

# แต่ละจุดเยี่ยมชมเพียงครั้งเดียว
subject to VisitOnce {j in NODES}:
  sum{i in NODES, k in VEHICLES} x[i,j,k] = 1;

# การไหลของจราจร
subject to FlowConservation {i in NODES, k in VEHICLES}:
  sum{j in NODES} x[i,j,k] - sum{j in NODES} x[j,i,k] = 0;

# ข้อจำกัดความจุ
subject to GeneralCapacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} general_demand[j] * x[i,j,k] <= general_capacity[k];

subject to RecycleCapacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} recycle_demand[j] * x[i,j,k] <= recycle_capacity[k];

# กำจัด subtour (สูตร MTZ)
subject to SubtourElimination {i in NODES, j in NODES}:
  u[i] - u[j] + card(NODES) * sum{k in VEHICLES} x[i,j,k] <= card(NODES) - 1;
```

**ไฟล์ที่ 2: ไฟล์ข้อมูล (`vrp_data.dat`)**
```ampl
# ข้อมูลเฉพาะของคุณมาที่นี่
set NODES := 1 2 3 4 5;
set VEHICLES := V1 V2;

param distance: 1 2 3 4 5 :=
  1   0 1500 2000 1800 2200
  2 1500    0 1700 1600 2100
  3 2000 1700    0 1900 2000
  4 1800 1600 1900    0 1400
  5 2200 2100 2000 1400    0;

param general_demand := 1 100 2 80 3 120 4 90 5 110;
param recycle_demand := 1 20 2 15 3 25 4 18 5 22;

param general_capacity := V1 1000 V2 2000;
param recycle_capacity := V1 500 V2 1000;
param fixed_cost := V1 800 V2 1500;
param fuel_cost := V1 8 V2 12;
```

#### ตัวเลือก B: รูปแบบ LP (ง่ายกว่า)

```lp
\ VRP ในรูปแบบ LP
Minimize
  obj: 800*x1_1 + 800*x1_2 + ... + 8*1500/1000*x1_2 + ...

Subject To
  visit1: x1_1 + x2_1 + x3_1 + x4_1 + x5_1 = 1
  visit2: x1_2 + x2_2 + x3_2 + x4_2 + x5_2 = 1
  ...
  cap_general_V1: 100*x1_1 + 80*x2_1 + ... <= 1000
  cap_recycle_V1: 20*x1_1 + 15*x2_1 + ... <= 500

Binary
  x1_1 x1_2 x2_1 x2_2 ...

End
```

### ขั้นตอนที่ 3: ส่งไป NEOS Server

#### ผ่านเว็บ

1. ไปที่: https://neos-server.org/neos/solvers/

2. **เลือกหมวดหมู่ Solver**:
   - เลือก: `MILP` (Mixed Integer Linear Programming)

3. **เลือก Solver**:
   - แนะนำ: `CPLEX` หรือ `Gurobi`
   - ทางเลือก: `CBC` (solver ฟรี)

4. **เลือกรูปแบบข้อมูลนำเข้า**:
   - เลือก: `AMPL` (ถ้าใช้ .mod + .dat)
   - หรือ: `GAMS` (ถ้าใช้ .gms)
   - หรือ: `LP` (ถ้าใช้ .lp)

5. **อัปโหลดไฟล์**:
   - สำหรับ AMPL: อัปโหลดทั้ง `.mod` และ `.dat`
   - สำหรับ LP: อัปโหลด `.lp`
   - สำหรับ GAMS: อัปโหลด `.gms`

6. **ตั้งค่าอีเมล** (ไม่บังคับ):
   - ใส่อีเมลเพื่อรับผลลัพธ์เมื่องานเสร็จ

7. **คลิก "Submit Job"**

#### ผ่านอีเมล (ทางเลือก)

ส่งอีเมลไปที่: `neos@mcs.anl.gov`

รูปแบบ subject line:
```
<หมวดหมู่>::<solver>::<รูปแบบอินพุต>
```

ตัวอย่าง:
```
milp::cplex::AMPL
```

แนบไฟล์และส่ง

### ขั้นตอนที่ 4: ตรวจสอบสถานะงาน

- เว็บ: คุณจะได้รับหมายเลขงาน (เช่น `1234567`)
- ตรวจสอบที่: https://neos-server.org/neos/cgi-bin/nstatus-job.cgi?jobnumber=1234567
- อีเมล: คุณจะได้รับผลลัพธ์เมื่อเสร็จ

### ขั้นตอนที่ 5: ดาวน์โหลดผลลัพธ์

ผลลัพธ์จะประกอบด้วย:
- ค่า objective (ต้นทุนรวม)
- ค่าตัวแปร (เส้นทาง)
- เวลาแก้ปัญหา
- สถานะ solver (เหมาะสมที่สุด, เป็นไปได้, ฯลฯ)

---

## การสร้างแบบจำลองสำหรับ NEOS

### สูตร MILP แบบย่อ

สำหรับ VRP เฉพาะของคุณ ตัวแปรตัดสินใจหลักคือ:

```
x[i,j,k] = 1 ถ้ารถ k เดินทางจากจุด i ไปจุด j
         = 0 ถ้าไม่ใช่
```

โดยที่:
- `i, j` = จุด (1 ถึง N)
- `k` = ประเภทรถ (V1, V2, ...)

### ฟังก์ชันเป้าหมาย

```
ลดค่า: Σ (ต้นทุนคงที่[k] * รถที่ใช้[k]) +
         Σ (ต้นทุนน้ำมัน[k] * ระยะทาง[i,j] * x[i,j,k] / 1000)
```

### ข้อจำกัดหลัก

```
1. แต่ละลูกค้าเยี่ยมชมครั้งเดียว:
   Σ x[i,j,k] = 1  สำหรับลูกค้า j ทั้งหมด

2. การไหลของจราจร:
   Σ x[i,j,k] - Σ x[j,i,k] = 0  สำหรับ i, k ทั้งหมด

3. ความจุ (ขยะทั่วไป):
   Σ ความต้องการ_ทั่วไป[j] * x[i,j,k] ≤ ความจุ_ทั่วไป[k]

4. ความจุ (รีไซเคิล):
   Σ ความต้องการ_รีไซเคิล[j] * x[i,j,k] ≤ ความจุ_รีไซเคิล[k]

5. ข้อจำกัดจุดทิ้ง:
   ทุกเส้นทางต้องผ่านจุดทิ้งก่อนกลับ depot

6. กำจัด subtour:
   ป้องกันวงจรที่ไม่ต่อกัน
```

---

## การเปรียบเทียบผลลัพธ์

### ตารางเปรียบเทียบ

สร้างสเปรดชีตเพื่อเปรียบเทียบ:

| ตัวชี้วัด | OR-Tools (คำตอบคุณ) | NEOS (CPLEX) | NEOS (Gurobi) | ผลต่าง |
|-------------|---------------------|--------------|---------------|----------|
| ต้นทุนรวม (บาท) | 3,580.75 | ? | ? | ? |
| ระยะทาง (กม.) | 125.3 | ? | ? | ? |
| รถที่ใช้ | 2 | ? | ? | ? |
| เวลาแก้ (วินาที) | 2.3 | ? | ? | ? |
| สถานะ | เหมาะสมที่สุด | ? | ? | - |

### สิ่งที่ต้องดู

1. **ค่า Objective**: ภายใน 1-5% ถือว่าดีมาก
2. **โครงสร้างเส้นทาง**: จุดเดียวกัน (ลำดับอาจต่าง)
3. **จำนวนรถ**: เท่ากันหรือใกล้เคียงกัน
4. **ช่องว่างความเหมาะสม**: NEOS ควรรายงาน 0% สำหรับคำตอบที่เหมาะสมที่สุด

### ผลลัพธ์ที่คาดหวัง

| ขนาดปัญหา | OR-Tools เทียบกับ NEOS | เหตุผล |
|-------------|-------------------------|---------|
| ≤ 20 จุด | เหมือนกันหรือใกล้เคียงมาก | ปัญหาเล็กพอสำหรับคำตอบเหมาะสมที่สุด |
| 20-50 จุด | ภายใน 1-3% | Heuristics ของ OR-Tools ดีมาก |
| 50-100 จุด | ภายใน 3-5% | ข้อจำกัดเวลามีผลต่อคุณภาพ |
| 100+ จุด | ภายใน 5-10% | ปัญหาซับซ้อน คาดการณ์แนวโน้ม |

---

## ตัวอย่างแบบจำลอง AMPL

นี่คือแบบจำลอง AMPL สมบูรณ์สำหรับ VRP ของคุณ:

```ampl
# vrp_complete.mod

# Sets
set NODES;
set VEHICLES;
set ARCS within {NODES, NODES};

# Parameters
param depot_symbolic symbolic := "Depot";
param checkpoint_symbolic symbolic := "จุดทิ้ง";

param depot_node in NODES;
param checkpoint_node in NODES;

param distance {i in NODES, j in NODES} >= 0;
param general_demand {i in NODES} >= 0;
param recycle_demand {i in NODES} >= 0;

param general_capacity {k in VEHICLES} >= 0;
param recycle_capacity {k in VEHICLES} >= 0;
param fixed_cost {k in VEHICLES} >= 0;
param fuel_cost_per_km {k in VEHICLES} >= 0;

param num_vehicles integer >= 1;

# Derived parameters
param num_nodes := card(NODES);

# Variables
var x {i in NODES, j in NODES, k in VEHICLES}, binary;
var vehicle_used {k in VEHICLES}, binary;
var u {i in NODES} >= 0 <= num_nodes;

# Objective
minimize Total_Cost:
  sum{k in VEHICLES} fixed_cost[k] * vehicle_used[k] +
  sum{i in NODES, j in NODES, k in VEHICLES}
    (fuel_cost_per_km[k] * distance[i,j] / 1000) * x[i,j,k];

# Constraints

# Each node visited exactly once (except depot and checkpoint)
subject to Visit_Once {j in NODES: j != depot_node and j != checkpoint_node}:
  sum{i in NODES, k in VEHICLES} x[i,j,k] = 1;

# Flow conservation at depot
subject to Flow_Depot_Out:
  sum{j in NODES, k in VEHICLES} x[depot_node,j,k] = sum{k in VEHICLES} vehicle_used[k];

subject to Flow_Depot_In:
  sum{i in NODES, k in VEHICLES} x[i,depot_node,k] = sum{k in VEHICLES} vehicle_used[k];

# Flow conservation at other nodes
subject to Flow_Conservation {i in NODES:
  i != depot_node and i != checkpoint_node}:
  sum{j in NODES, k in VEHICLES} x[i,j,k] = sum{j in NODES, k in VEHICLES} x[j,i,k];

# Checkpoint visited by each vehicle
subject to Checkpoint_Visit {k in VEHICLES: vehicle_used[k] = 1}:
  sum{i in NODES} x[i,checkpoint_node,k] = sum{j in NODES} x[checkpoint_node,j,k];

# Capacity constraints
subject to General_Capacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} general_demand[j] * x[i,j,k]
  <= general_capacity[k] * vehicle_used[k];

subject to Recycle_Capacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} recycle_demand[j] * x[i,j,k]
  <= recycle_capacity[k] * vehicle_used[k];

# Subtour elimination (MTZ)
subject to MTZ_1 {i in NODES, j in NODES:
  i != depot_node and j != depot_node and i != j}:
  u[i] - u[j] + num_nodes * sum{k in VEHICLES} x[i,j,k] <= num_nodes - 1;

subject to MTZ_2 {i in NODES: i != depot_node}:
  1 <= u[i];

# Fix u for depot
subject to MTZ_Depot:
  u[depot_node] = 0;

# Vehicle used constraint
subject to Vehicle_Used_Def {k in VEHICLES, j in NODES: j != depot_node}:
  x[depot_node,j,k] <= vehicle_used[k];
```

### ตัวอย่างไฟล์ข้อมูล

```ampl
# vrp_20nodes.dat

set NODES := 1 2 3 4 5 6 7 8 9 10
             11 12 13 14 15 16 17 18 19 20;

set VEHICLES := V1 V2;

set ARCS := (1,2) (1,3) (2,1) (2,3) ...;

param depot_node := 1;
param checkpoint_node := 20;

param distance:
    1    2    3    4    5    6    7    8    9   10
   11   12   13   14   15   16   17   18   19   20 :=
1    0 1500 2000 1800 2200 2500 2800 2100 2400 3000
2 1500    0 1700 1600 2100 2300 2600 1900 2200 2800
3 2000 1700    0 1900 2000 2200 2400 1800 2100 2600
... (เมทริกซ์ระยะทางเต็ม)

param general_demand :=
  1   100
  2    80
  3   120
... (ความต้องการทั้งหมด)

param recycle_demand :=
  1    20
  2    15
  3    25
... (ความต้องการทั้งหมด)

param general_capacity :=
  V1   1000
  V2   2000;

param recycle_capacity :=
  V1    500
  V2   1000;

param fixed_cost :=
  V1    800
  V2   1500;

param fuel_cost_per_km :=
  V1      8
  V2      12;

param num_vehicles := 5;
```

---

## ตัวอย่างแบบจำลอง GAMS

```gams
$Title ปัญหา VRP การเก็บขยะ

Sets
    i       "จุด"          /1*20/
    k       "รถ"       /V1, V2/
    alias(i,j);

Parameters
    depot_node      "จุดรวมรถ"          /1/
    checkpoint_node "จุดทิ้ง"     /20/

    distance(i,j)   "เมทริกซ์ระยะทาง (เมตร)"
    general_demand(i)  "ความต้องการขยะทั่วไป (กก.)"
    recycle_demand(i)  "ความต้องการขยะรีไซเคิล (กก.)"

    gen_cap(k)      "ความจุขยะทั่วไป"
    rec_cap(k)      "ความจุขยะรีไซเคิล"
    fix_cost(k)     "ต้นทุนคงที่"
    fuel_cost(k)    "ต้นทุนน้ำมันต่อกม.";

* โหลดข้อมูลจาก Excel หรือใส่ที่นี่
$CALL GDXXRW data.xlsx par=distance rng=Sheet1!A1:T21
$GDXIN data.gdx
$LOAD distance
$GDXIN

Scalar
    num_nodes "จำนวนจุด" /20/;

Variables
    x(i,j,k)    "Binary: รถ k เดินทางจาก i ไป j"
    used(k)     "Binary: รถ k ถูกใช้"
    u(i)        "ตัวแปรกำจัด subtour"
    z           "ต้นทุนรวม";

Binary Variables x, used;

Equations
    obj         "ฟังก์ชันเป้าหมาย"
    visit_once(j)  "แต่ละจุดเยี่ยมชมครั้งเดียว"
    flow_out     "การไหลออกจาก depot"
    flow_in      "การไหลเข้าสู่ depot"
    flow_cons(i) "การไหลของจราจร"
    checkpoint(k) "เยี่ยมชมจุดทิ้ง"
    gen_cap_con(k) "ความจุขยะทั่วไป"
    rec_cap_con(k) "ความจุขยะรีไซเคิล"
    mtz1(i,j)    "กำจัด subtour ด้วย MTZ"
    mtz2(i)      "ขอบเขต MTZ"
    vehicle_def(k,j) "นิยามรถที่ใช้";

* Objective
obj..
    z =e= sum(k, fix_cost(k) * used(k))
        + sum((i,j,k), fuel_cost(k) * distance(i,j) / 1000 * x(i,j,k));

* แต่ละจุดเยี่ยมชมครั้งเดียว
visit_once(j)$(ord(j)<>depot_node and ord(j)<>checkpoint_node)..
    sum((i,k), x(i,j,k)) =e= 1;

* (เพิ่ม constraints อื่นๆ...)

Model vrp /all/;
Solve vrp using mip minimizing z;
Display x.l, z.l;
```

---

## เคล็ดลับความสำเร็จ

### 1. เริ่มจากเล็ก

- ทดสอบ NEOS กับปัญหาเล็กสุด (20 จุด) ก่อน
- ตรวจสอบว่าแบบจำลองทำงานได้ก่อนขยายขนาด

### 2. ขีดจำกัดเวลา

- ตั้งเวลาที่เหมาะสม:
  - 20-30 จุด: 5 นาที
  - 30-50 จุด: 15 นาที
  - 50-100 จุด: 30 นาที
  - 100+ จุด: 1 ชั่วโมง

### 3. ผลลัพธ์ทางอีเมล

- ให้อีเมลเสมอสำหรับงานใหญ่
- ผลลัพธ์หมดอายุภายใน 24 ชั่วโมง

### 4. การเลือก Solver

| สถานการณ์ | Solver ที่แนะนำ |
|-----------|----------------|
| ต้องการคำตอบดีที่สุด | CPLEX หรือ Gurobi |
| ทางเลือกฟรี | CBC |
| ตรวจสอบเร็วๆ | SCIP |
| การศึกษา | GLPK |

### 5. ปัญหาทั่วไป

| ปัญหา | วิธีแก้ |
|--------|---------|
| "Infeasible" | ตรวจสอบว่า constraints ไม่เคร่งเกินไป |
| "Unbounded" | ตรวจสอบว่า objective function ถูกกำหนด |
| แก้นาน | ลดขนาดปัญหาหรือเพิ่มเวลา |
| ความจำเต็ม | ใช้สูตรที่เรียบง่ายกว่า |

---

## การตีความผลลัพธ์จาก NEOS

### ตัวอย่างผลลัพธ์

```
สถานะคำตอบ: เหมาะสมที่สุด
ค่า Objective: 3580.75

ตัวแปร:
x[1,5,V1] = 1
x[5,8,V1] = 1
x[8,20,V1] = 1  (จุดทิ้ง)
x[20,1,V1] = 1  (กลับ depot)
...

การใช้ทรัพยากร:
  เวลาแก้: 3.24 วินาที
  รอบการวนซ้ำ: 15234
```

### ผลลัพธ์ที่ต้องดึงออก

1. **ค่า Objective**: เปรียบเทียบกับคำตอบของคุณ
2. **สถานะคำตอบ**: เหมาะสมที่สุด > เป็นไปได้
3. **ค่าตัวแปร**: ดึงเส้นทาง
4. **เวลาแก้**: เปรียบเทียบประสิทธิภาพ

---

## การแก้ปัญหา

### ประเด็น: คำตอบต่างจาก OR-Tools

นี่เป็นเรื่องปกติ! เหตุผล:

1. **คำตอบเหมาะสมที่สุดหลายแบบ**: ต้นทุนเท่ากัน เส้นทางต่างกัน
2. **Heuristic vs Optimal**: OR-Tools ใช้ heuristics
3. **ข้อจำกัดเวลา**: NEOS อาจหมดเวลาก่อนหาคำตอบที่ดีที่สุด
4. **ความแตกต่างของ Solver**: อัลกอริทึมต่างกัน

### เมื่อควรกังวล

| ความแตกต่าง | การดำเนินการ |
|---------------|-------------|
| ต้นทุนต่างกัน < 1% | ปกติ, ยอมรับได้ |
| ต้นทุนต่างกัน 1-5% | ตรวจสอบสูตรแบบจำลอง |
| ต้นทุนต่างกัน > 5% | ทบทวน constraints และข้อมูล |
| จำนวนรถต่างกัน | ตรวจสอบนิยามความจุ |
| จุดหายไป | ตรวจสอบ constraints การเยี่ยมชม |

---

## อ้างอิงด่วน

### เว็บไซต์ NEOS: https://neos-server.org

### ลิงก์โดยตรง

| งาน | URL |
|------|-----|
| ส่งงาน | https://neos-server.org/neos/solvers/ |
| ตรวจสอบสถานะ | https://neos-server.org/neos/cgi-bin/nstatus-job.cgi |
| รายการ Solver | https://neos-server.org/neos/solvers/ |

### การส่งทางอีเมล

```
ถึง: neos@mcs.anl.gov
หัวข้อ: milp::cplex::AMPL
แนบไฟล์: vrp_model.mod, vrp_data.dat
```

---

## สรุป Checklist

- [ ] ระบุประเภทปัญหา: **CVRP with Heterogeneous Fleet**
- [ ] เลือกภาษาสร้างแบบจำลอง: **AMPL** (แนะนำ)
- [ ] สร้างไฟล์แบบจำลอง (`.mod`)
- [ ] สร้างไฟล์ข้อมูล (`.dat`)
- [ ] เลือก solver: **CPLEX** หรือ **Gurobi**
- [ ] ส่งผ่านเว็บอินเทอร์เฟซ NEOS
- [ ] รอการแจ้งเตือนทางอีเมล
- [ ] ดาวน์โหลดและเปรียบเทียบผลลัพธ์
- [ ] บันทึกความแตกต่าง
- [ ] ตรวจสอบกับคำตอบ OR-Tools

---

## ขั้นตอนถัดไป

1. **สำหรับปัญหาเล็ก (≤50 จุด)**:
   - ใช้ CPLEX บน NEOS สำหรับคำตอบเหมาะสมที่สุด
   - เปรียบเทียบโดยตรงกับ OR-Tools

2. **สำหรับปัญหาใหญ่ (100+ จุด)**:
   - ใช้ CPLEX จำกัดเวลา 1 ชั่วโมง
   - คาดหวัง "คำตอบที่รู้จักดีที่สุด" ไม่ใช่เหมาะสมที่สุดเสมอไป
   - มุ่งเน้นเปอร์เซ็นต์ช่องว่าง

3. **สำหรับการตรวจสอบอย่างต่อเนื่อง**:
   - ส่งอัตโนมัติผ่าน NEOS API
   - สร้างสคริปต์เปรียบเทียบ

---

## แหล่งข้อมูลเพิ่มเติม

- **เอกสาร NEOS**: https://neos-server.org/neos/docs/
- **บทเรียน AMPL**: https://ampl.com/resources/tutorials/
- **บทเรียน GAMS**: https://www.gams.com/latest/docs/UG_Tutorials.html
- **วรรณกรรม VRP**: Toth & Vigo, "The Vehicle Routing Problem"

---

## คำอธิบายศัพท์เฉพาะ

| ศัพท์ | ความหมาย |
|--------|-----------|
| **Optimal** | เหมาะสมที่สุด (ไม่มีคำตอบที่ดีกว่า) |
| **Feasible** | เป็นไปได้ (เป็นคำตอบที่ถูกต้อง แต่อาจไม่ดีที่สุด) |
| **Objective Value** | ค่าเป้าหมาย (ต้นทุนรวมที่ต้องการลด) |
| **Gap** | ช่องว่างระหว่างคำตอบปัจจุบันกับคำตอบที่ดีที่สุดที่เป็นไปได้ |
| **Heuristic** | วิธีแก้ปัญหาแบบเร็ว แต่ไม่รับประกันความเหมาะสมที่สุด |
| **MILP** | Mixed Integer Linear Programming |
| **Constraint** | เงื่อนไขข้อจำกัด |
| **Depot** | จุดรวมรถ (Node 1) |
| **Checkpoint** | จุดทิ้งขยะ |

---

สร้างเมื่อ: 2025-03-05
สำหรับ: VRP Solver v2 - การวางแผนเส้นทางเก็บขยะ
