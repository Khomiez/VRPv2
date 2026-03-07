# VRP Model for NEOS Server - 138 Nodes
# Capacitated Vehicle Routing Problem with Heterogeneous Fleet

# Sets (Data Collections)
set NODES;                   # All nodes (locations)
set VEHICLES;                # Vehicle types (A, B, C)

# Parameters (Input Data)
param depot_node in NODES;           # Depot node (starting point)
param checkpoint_node in NODES;      # Checkpoint node (must visit before returning)

# Distance matrix in meters
param distance {i in NODES, j in NODES} >= 0;

# Waste demands in kilograms
param general_demand {i in NODES} >= 0;    # General waste
param recycle_demand {i in NODES} >= 0;    # Recyclable waste

# Vehicle data
param general_capacity {k in VEHICLES} >= 0;   # Capacity for general waste (kg)
param recycle_capacity {k in VEHICLES} >= 0;   # Capacity for recyclable waste (kg)
param fixed_cost {k in VEHICLES} >= 0;         # Fixed cost per vehicle (THB)
param fuel_cost_per_km {k in VEHICLES} >= 0;   # Fuel cost per kilometer (THB/km)

param num_vehicles integer >= 1;     # Maximum number of vehicles available

# Decision Variables
var x {i in NODES, j in NODES, k in VEHICLES}, binary;  # 1 if vehicle k travels from i to j
var vehicle_used {k in VEHICLES}, binary;              # 1 if vehicle k is used

# MTZ subtour elimination variables (per-vehicle sequence position, all non-depot nodes)
# Per-vehicle indexing prevents false infeasibility when checkpoint is visited by multiple vehicles
var u {i in NODES, k in VEHICLES: i != depot_node} >= 0, <= card(NODES);

# Objective Function: Minimize Total Cost
# Total Cost = Fixed Costs + Fuel Costs
minimize Total_Cost:
  sum{k in VEHICLES} fixed_cost[k] * vehicle_used[k] +
  sum{i in NODES, j in NODES, k in VEHICLES}
    (fuel_cost_per_km[k] * distance[i,j] / 1000.0) * x[i,j,k];

# Constraints

# 1. Each customer (non-depot, non-checkpoint) visited exactly once
subject to Visit_Once {j in NODES: j != depot_node and j != checkpoint_node}:
  sum{i in NODES, k in VEHICLES} x[i,j,k] = 1;

# 2. Flow conservation at depot
subject to Flow_Depot_Out:
  sum{j in NODES, k in VEHICLES} x[depot_node,j,k] = sum{k in VEHICLES} vehicle_used[k];

subject to Flow_Depot_In:
  sum{i in NODES, k in VEHICLES} x[i,depot_node,k] = sum{k in VEHICLES} vehicle_used[k];

# 3. Flow conservation at customer nodes (PER VEHICLE - prevents vehicle swapping)
subject to Flow_Conservation {k in VEHICLES, i in NODES: i != depot_node and i != checkpoint_node}:
  sum{j in NODES} x[i,j,k] = sum{j in NODES} x[j,i,k];

# 4. Flow conservation at checkpoint (PER VEHICLE - prevents vehicle swapping)
subject to Flow_Checkpoint {k in VEHICLES}:
  sum{i in NODES} x[i,checkpoint_node,k] = sum{j in NODES} x[checkpoint_node,j,k];

# 5. General waste capacity constraint
subject to General_Capacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} general_demand[j] * x[i,j,k]
  <= general_capacity[k] * vehicle_used[k];

# 6. Recyclable waste capacity constraint
subject to Recycle_Capacity {k in VEHICLES}:
  sum{i in NODES, j in NODES} recycle_demand[j] * x[i,j,k]
  <= recycle_capacity[k] * vehicle_used[k];

# 7. Limit total number of vehicles
subject to Max_Vehicles:
  sum{k in VEHICLES} vehicle_used[k] <= num_vehicles;

# 8. Link vehicle_used to routing variables
subject to Vehicle_Used_Link {k in VEHICLES, j in NODES: j != depot_node}:
  x[depot_node,j,k] <= vehicle_used[k];

# 9. No self-loops (vehicle cannot stay at same node)
subject to No_Self_Loop {i in NODES, k in VEHICLES}:
  x[i,i,k] = 0;

# 10. Per-vehicle depot departure/return balance
subject to Depot_Balance {k in VEHICLES}:
  sum{j in NODES} x[depot_node,j,k] = sum{i in NODES} x[i,depot_node,k];

# 11. MTZ subtour elimination constraints (per vehicle, includes checkpoint)
# Covers ALL non-depot nodes including checkpoint so subtours via checkpoint are blocked.
# Per-vehicle u[i,k] avoids false infeasibility when multiple vehicles visit checkpoint.
subject to MTZ_Subtour {i in NODES, j in NODES, k in VEHICLES:
    i != depot_node and j != depot_node and i != j}:
  u[i,k] - u[j,k] + card(NODES) * x[i,j,k] <= card(NODES) - 1;

# 12. Every active vehicle MUST visit the checkpoint exactly once
subject to Mandatory_Checkpoint_Visit {k in VEHICLES}:
  sum{i in NODES} x[i,checkpoint_node,k] = vehicle_used[k];

# 13. Checkpoint MUST be the last stop before returning to depot
# Enforces route structure: Depot -> [customers] -> Checkpoint -> Depot
subject to Checkpoint_Is_Last {k in VEHICLES}:
  x[checkpoint_node,depot_node,k] = vehicle_used[k];
