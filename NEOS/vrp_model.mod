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