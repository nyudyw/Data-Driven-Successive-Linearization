# Config Name
config_name: 'linDistFlow_config'

# System Parameters
system:
  base_mva: 100
  base_kv: 12.66

# Model Parameters
model:
  pf_model: "LinDistFlow"/"DistFlow"
  case: "case33"
  
# Controller Parameters
controller:
  v_ref: [1.0,1.0,1.0,1.0]
  q_max: [0.005,0.005,0.005,0.005]
  v_db: [0.02,0.02,0.02,0.02] # Deadband 
  capacity: [0.01,0.01,0.01,0.01] # Maximum capacity
  node_list: [2,5,10,28] # Controllable nodes
  
# Optimization Parameters
optimization:
  solver: 'mosek'
  max_iter: 1000
  tol: 1.0e-6
  
# Test Parameters
test:
  scenarios:
    - name: 'base_case'
      v_ref: 1.0
    - name: 'high_voltage'
      v_ref: 1.05
    
# Plotting Parameters
plot:
  figsize: [10, 6]
  dpi: 300
  save_format: 'pdf'