# NEOS Server Verification Guide for VRP Solver

## Table of Contents
1. [What Problem Are You Solving?](#what-problem-are-you-solving)
2. [Why Verify with NEOS Server?](#why-verify-with-neos-server)
3. [Problem Classification](#problem-classification)
4. [Step-by-Step NEOS Server Usage](#step-by-step-neos-server-usage)
5. [Formulating Your Problem for NEOS](#formulating-your-problem-for-neos)
6. [Comparing Results](#comparing-results)
7. [Example AMPL Model](#example-ampl-model)
8. [Example GAMS Model](#example-gams-model)

---

## What Problem Are You Solving?

Your VRP problem is a **Capacitated Vehicle Routing Problem with Heterogeneous Fleet (CVRPHF)** with additional constraints:

### Problem Characteristics

| Feature | Description |
|---------|-------------|
| **Base Type** | Capacitated Vehicle Routing Problem (CVRP) |
| **Fleet Type** | Heterogeneous (multiple vehicle types with different capacities and costs) |
| **Demands** | Two commodity types (general waste + recyclable) |
| **Objective** | Minimize total cost (fixed costs + fuel costs) |
| **Constraints** | Capacity limits, checkpoint visits, depot start/end |

### Key Constraints in Your Model

1. **Route Structure**: `Depot (Node 1) → Collection Nodes → Checkpoint (จุดทิ้ง) → Depot`
2. **Dual Capacity Constraints**:
   - General waste capacity per vehicle
   - Recyclable waste capacity per vehicle
3. **Cost Components**:
   - Fixed cost per vehicle (บาท/คัน)
   - Fuel cost per kilometer (บาท/กม.)
4. **Mandatory Checkpoint**: Every route MUST visit the dump point before returning to depot

---

## Why Verify with NEOS Server?

**NEOS Server** (https://neos-server.org) is a free internet-based optimization service that provides:

- Access to commercial solvers (CPLEX, Gurobi, XPRESS)
- Academic solvers (CBC, SCIP, GLPK)
- Different modeling languages (AMPL, GAMS, LP)
- No installation required

### Verification Benefits

| Benefit | Description |
|---------|-------------|
| **Solver Independence** | Verify your OR-Tools solution with completely different solvers |
| **Optimality Gap** | Check if your solution is truly optimal |
| **Benchmark Comparison** | Compare performance across multiple solvers |
| **Model Validation** | Ensure your mathematical formulation is correct |

---

## Problem Classification for NEOS

When submitting to NEOS, your problem can be classified as:

### Primary Categories

1. **Mixed Integer Linear Programming (MILP)**
   - Category: `milp` (Mixed Integer Linear Program)
   - Most accurate formulation
   - Solvers: CPLEX, Gurobi, CBC, SCIP

2. **Integer Programming (IP)**
   - Category: `integer` (Integer Program)
   - Simpler formulation
   - Faster solve time

### NEOS Solver Recommendations

| Solver | Category | Best For | Time Limit |
|--------|----------|----------|------------|
| **CPLEX** | MILP | Large instances, optimal solutions | 1 hour |
| **Gurobi** | MILP | Large instances, optimal solutions | 1 hour |
| **CBC** | MILP | Medium instances, free/open-source | 30 minutes |
| **SCIP** | MILP | Complex constraints | 30 minutes |
| **GLPK** | MILP | Small instances, simple formulation | 15 minutes |

---

## Step-by-Step NEOS Server Usage

### Step 1: Choose Your Modeling Language

NEOS supports multiple languages. Choose based on your familiarity:

| Language | Difficulty | Recommended | File Extension |
|----------|------------|-------------|----------------|
| **AMPL** | Medium | ✅ Yes | `.mod` + `.dat` |
| **GAMS** | Medium | ✅ Yes | `.gms` |
| **LP** | Easy | ✅ Yes (for simple models) | `.lp` |

### Step 2: Prepare Your Input File

#### Option A: AMPL (Recommended)

Create two files:

**File 1: Model file (`vrp_model.mod`)**
```
# VRP Mathematical Model
set NODES;                   # All nodes
set VEHICLES;                # Vehicle types
param depot {NODES} binary;  # 1 if node is depot
param checkpoint {NODES} binary; # 1 if node is checkpoint

# Distance matrix
param distance {i in NODES, j in NODES};

# Demands
param general_demand {NODES};
param recycle_demand {NODES};

# Vehicle data
param general_capacity {VEHICLES};
param recycle_capacity {VEHICLES};
param fixed_cost {VEHICLES};
param fuel_cost {VEHICLES};

# Decision variables
var x {NODES, NODES, VEHICLES}, binary;  # 1 if vehicle k travels from i to j
var u {NODES} >= 0;                      # Subtour elimination

# Minimize total cost
minimize TotalCost:
  sum{k in VEHICLES} (fixed_cost[k] * sum{j in NODES} x[depot_node, j, k]) +
  sum{i in NODES, j in NODES, k in VEHICLES} fuel_cost[k] * distance[i,j] * x[i,j,k] / 1000;

# Subject to:
# Each node visited exactly once
subject to VisitOnce {j in NODES}:
  sum{i in NODES, k in VEHICLES} x[i,j,k] = 1;

# Flow conservation
subject to FlowConservation {i in NODES, k in VEHICLES}:
  sum{j in NODES} x[i,j,k] - sum{j in NODES} x[j,i,k] = 0;

# Capacity constraints
subject to GeneralCapacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} general_demand[j] * x[i,j,k] <= general_capacity[k];

subject to RecycleCapacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} recycle_demand[j] * x[i,j,k] <= recycle_capacity[k];

# Subtour elimination (MTZ formulation)
subject to SubtourElimination {i in NODES, j in NODES}:
  u[i] - u[j] + card(NODES) * sum{k in VEHICLES} x[i,j,k] <= card(NODES) - 1;
```

**File 2: Data file (`vrp_data.dat`)**
```
# Your specific data goes here
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

#### Option B: LP Format (Simpler)

```
\ VRP in LP Format
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

### Step 3: Submit to NEOS Server

#### Via Web Interface

1. Go to: https://neos-server.org/neos/solvers/

2. **Select Solver Category**:
   - Choose: `MILP` (Mixed Integer Linear Programming)

3. **Select Solver**:
   - Recommended: `CPLEX` or `Gurobi`
   - Alternative: `CBC` (free solver)

4. **Select Input Format**:
   - Choose: `AMPL` (if using .mod + .dat)
   - Or: `GAMS` (if using .gms)
   - Or: `LP` (if using .lp)

5. **Upload Files**:
   - For AMPL: Upload BOTH `.mod` and `.dat` files
   - For LP: Upload `.lp` file
   - For GAMS: Upload `.gms` file

6. **Set Email** (optional):
   - Enter your email to receive results when job completes

7. **Click "Submit Job"**

#### Via Email (Alternative)

Send email to: `neos@mcs.anl.gov`

Subject line format:
```
<category>::<solver>::<input-format>
```

Example:
```
milp::cplex::AMPL
```

Attach your files and send.

### Step 4: Monitor Job Status

- Web: You'll get a job number (e.g., `1234567`)
- Monitor at: https://neos-server.org/neos/cgi-bin/nstatus-job.cgi?jobnumber=1234567
- Email: You'll receive results when done

### Step 5: Download Results

Results will include:
- Objective value (total cost)
- Variable values (routes)
- Solve time
- Solver status (optimal, feasible, etc.)

---

## Formulating Your Problem for NEOS

### Simplified MILP Formulation

For your specific VRP, the key decision variable is:

```
x[i,j,k] = 1 if vehicle k travels from node i to node j
         = 0 otherwise
```

Where:
- `i, j` = nodes (1 to N)
- `k` = vehicle types (V1, V2, ...)

### Objective Function

```
Minimize: Σ (fixed_cost[k] * vehicle_used[k]) +
          Σ (fuel_cost[k] * distance[i,j] * x[i,j,k] / 1000)
```

### Key Constraints

```
1. Each customer visited exactly once:
   Σ x[i,j,k] = 1  for all customers j

2. Flow conservation:
   Σ x[i,j,k] - Σ x[j,i,k] = 0  for all i, k

3. Capacity (general):
   Σ demand_general[j] * x[i,j,k] ≤ capacity_general[k]

4. Capacity (recycle):
   Σ demand_recycle[j] * x[i,j,k] ≤ capacity_recycle[k]

5. Checkpoint constraint:
   Every route must pass through checkpoint before depot

6. Subtour elimination:
   Prevent disconnected cycles
```

---

## Comparing Results

### Comparison Table

Create a spreadsheet to compare:

| Metric | OR-Tools (Your Solution) | NEOS (CPLEX) | NEOS (Gurobi) | Difference |
|--------|-------------------------|--------------|---------------|------------|
| Total Cost (THB) | 3,580.75 | ? | ? | ? |
| Distance (km) | 125.3 | ? | ? | ? |
| Vehicles Used | 2 | ? | ? | ? |
| Solve Time (sec) | 2.3 | ? | ? | ? |
| Status | OPTIMAL | ? | ? | - |

### What to Look For

1. **Objective Value**: Within 1-5% is excellent
2. **Route Structure**: Same nodes visited (possibly different order)
3. **Vehicle Count**: Same or similar number of vehicles
4. **Optimality Gap**: NEOS should report 0% for optimal

### Expected Results

| Problem Size | OR-Tools vs NEOS | Reason |
|--------------|------------------|--------|
| ≤ 20 nodes | Identical or very close | Problem is small enough for optimal solutions |
| 20-50 nodes | Within 1-3% | OR-Tools heuristics are very good |
| 50-100 nodes | Within 3-5% | Time limit affects quality |
| 100+ nodes | Within 5-10% | Complex problem, approximations expected |

---

## Example AMPL Model

Here's a complete AMPL model for your VRP:

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

### Data File Example

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
... (full distance matrix)

param general_demand :=
  1   100
  2    80
  3   120
... (all demands)

param recycle_demand :=
  1    20
  2    15
  3    25
... (all demands)

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

## Example GAMS Model

```gams
$Title VRP Waste Collection

Sets
    i       "Nodes"          /1*20/
    k       "Vehicles"       /V1, V2/
    alias(i,j);

Parameters
    depot_node      "Depot node"          /1/
    checkpoint_node "Checkpoint node"     /20/

    distance(i,j)   "Distance matrix (meters)"
    general_demand(i)  "General waste demand (kg)"
    recycle_demand(i)  "Recyclable waste demand (kg)"

    gen_cap(k)      "General capacity"
    rec_cap(k)      "Recycle capacity"
    fix_cost(k)     "Fixed cost"
    fuel_cost(k)    "Fuel cost per km";

* Load data from Excel or include here
$CALL GDXXRW data.xlsx par=distance rng=Sheet1!A1:T21
$GDXIN data.gdx
$LOAD distance
$GDXIN

Scalar
    num_nodes "Number of nodes" /20/;

Variables
    x(i,j,k)    "Binary: vehicle k travels from i to j"
    used(k)     "Binary: vehicle k is used"
    u(i)        "Subtour elimination variable"
    z           "Total cost";

Binary Variables x, used;

Equations
    obj         "Objective function"
    visit_once(j)  "Each node visited once"
    flow_out     "Flow out of depot"
    flow_in      "Flow into depot"
    flow_cons(i) "Flow conservation"
    checkpoint(k) "Visit checkpoint"
    gen_cap_con(k) "General capacity"
    rec_cap_con(k) "Recycle capacity"
    mtz1(i,j)    "MTZ subtour elimination"
    mtz2(i)      "MTZ bounds"
    vehicle_def(k,j) "Vehicle used definition";

* Objective
obj..
    z =e= sum(k, fix_cost(k) * used(k))
        + sum((i,j,k), fuel_cost(k) * distance(i,j) / 1000 * x(i,j,k));

* Each node visited once
visit_once(j)$(ord(j)<>depot_node and ord(j)<>checkpoint_node)..
    sum((i,k), x(i,j,k)) =e= 1;

* (Add other constraints...)

Model vrp /all/;
Solve vrp using mip minimizing z;
Display x.l, z.l;
```

---

## Tips for Success

### 1. Start Small

- Test NEOS with your smallest problem (20 nodes) first
- Verify the model works before scaling up

### 2. Time Limits

- Set appropriate time limits:
  - 20-30 nodes: 5 minutes
  - 30-50 nodes: 15 minutes
  - 50-100 nodes: 30 minutes
  - 100+ nodes: 1 hour

### 3. Email Results

- Always provide email for large jobs
- Results expire after 24 hours

### 4. Solver Choice

| Scenario | Recommended Solver |
|----------|-------------------|
| Best solution possible | CPLEX or Gurobi |
| Free alternative | CBC |
| Quick verification | SCIP |
| Educational purpose | GLPK |

### 5. Common Issues

| Issue | Solution |
|-------|----------|
| "Infeasible" | Check constraints are not too strict |
| "Unbounded" | Check objective function is defined |
| Long solve time | Reduce problem size or increase time limit |
| Memory error | Use simpler formulation |

---

## Interpreting NEOS Results

### Sample Output

```
Solution Status: Optimal
Objective Value: 3580.75

Variables:
x[1,5,V1] = 1
x[5,8,V1] = 1
x[8,20,V1] = 1  (checkpoint)
x[20,1,V1] = 1  (return to depot)
...

Resource Usage:
  Solve time: 3.24 seconds
  iterations: 15234
```

### Key Results to Extract

1. **Objective Value**: Compare with your solution
2. **Solution Status**: Optimal > Feasible
3. **Variable Values**: Extract routes
4. **Solve Time**: Benchmark performance

---

## Troubleshooting

### Issue: Different Solution from OR-Tools

This is normal! Reasons:

1. **Multiple Optimal Solutions**: Same cost, different routes
2. **Heuristic vs Optimal**: OR-Tools uses heuristics
3. **Time Limit**: NEOS may time out before finding best
4. **Solver Differences**: Different algorithms

### When to Be Concerned

| Difference | Action |
|------------|--------|
| Cost difference < 1% | Normal, acceptable |
| Cost difference 1-5% | Check model formulation |
| Cost difference > 5% | Review constraints and data |
| Different vehicle count | Check capacity definitions |
| Missing nodes | Check visit constraints |

---

## Quick Reference

### NEOS Website: https://neos-server.org

### Direct Links

| Task | URL |
|------|-----|
| Submit Job | https://neos-server.org/neos/solvers/ |
| Check Status | https://neos-server.org/neos/cgi-bin/nstatus-job.cgi |
| Solver List | https://neos-server.org/neos/solvers/ |

### Email Submission

```
To: neos@mcs.anl.gov
Subject: milp::cplex::AMPL
Attachment: vrp_model.mod, vrp_data.dat
```

---

## Summary Checklist

- [ ] Identify problem type: **CVRP with Heterogeneous Fleet**
- [ ] Choose modeling language: **AMPL** (recommended)
- [ ] Create model file (`.mod`)
- [ ] Create data file (`.dat`)
- [ ] Select solver: **CPLEX** or **Gurobi**
- [ ] Submit to NEOS web interface
- [ ] Wait for email notification
- [ ] Download and compare results
- [ ] Document differences
- [ ] Validate against OR-Tools solution

---

## Next Steps

1. **For Small Problems (≤50 nodes)**:
   - Use CPLEX on NEOS for optimal solution
   - Compare directly with OR-Tools

2. **For Large Problems (100+ nodes)**:
   - Use CPLEX with 1-hour time limit
   - Expect "best known" solution, not necessarily optimal
   - Focus on gap percentage

3. **For Continuous Verification**:
   - Automate submission via NEOS API
   - Create comparison script

---

## Additional Resources

- **NEOS Documentation**: https://neos-server.org/neos/docs/
- **AMPL Tutorial**: https://ampl.com/resources/tutorials/
- **GAMS Tutorial**: https://www.gams.com/latest/docs/UG_Tutorials.html
- **VRP Literature**: Toth & Vigo, "The Vehicle Routing Problem"

---

Created: 2025-03-05
For: VRP Solver v2 - Waste Collection Optimization
