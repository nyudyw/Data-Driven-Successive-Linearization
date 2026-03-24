from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
from configs.config_loader import ConfigLoader

def save_results(voltage_profile, sol_list, config_loader: ConfigLoader, controlled_nodes):
   """Save simulation results and plots"""
   
   # Create results directory structure
   results_dir = Path('./results')
   timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   config_name = config_loader.get_config_name()
   sim_dir = results_dir / f"{timestamp}_{config_name}"
   
   # Create directories
   sim_dir.mkdir(parents=True, exist_ok=True)
   (sim_dir / 'figures').mkdir(exist_ok=True)
   (sim_dir / 'data').mkdir(exist_ok=True)
   
   # Save voltage profiles
   np.save(sim_dir / 'data' / 'voltage_profile.npy', voltage_profile)
   
   # Save complete solution data
   sol_data = {
       'voltage': voltage_profile,
       'controlled_nodes': controlled_nodes
   }
   np.save(sim_dir / 'data' / 'complete_solutions.npy', sol_data)
   
   # Create plots
   # 1. Voltage profile over time for controlled nodes
   plt.figure(figsize=(10, 6))
   for node in controlled_nodes:
       plt.plot(voltage_profile[:, node], label=f'Bus {node}')
   plt.xlabel('Time Step')
   plt.ylabel('Voltage (p.u.)')
   plt.ylim((0.83,1.03))
   plt.title('Voltage Profiles of Controlled Buses')
   plt.legend()
   plt.grid(True)
   plt.savefig(sim_dir / 'figures' / 'voltage_profiles.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   # 2. Voltage distribution boxplot
   plt.figure(figsize=(12, 6))
   plt.boxplot([voltage_profile[:, i] for i in controlled_nodes],
               labels=[f'Bus {i}' for i in controlled_nodes])
   plt.xlabel('Bus Number')
   plt.ylabel('Voltage (p.u.)')
   plt.title('Voltage Distribution at Controlled Buses')
   plt.grid(True)
   plt.savefig(sim_dir / 'figures' / 'voltage_boxplot.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   # Save configuration info
   case_name = config_loader.get_model_config()['case']
   pf_model = config_loader.get_model_config()['pf_model']
   config_info = {
       'timestamp': timestamp,
       'config_name': config_name,
       'controlled_nodes': controlled_nodes,
       'simulation_length': len(voltage_profile),
       'case': case_name,
       'pf_model': pf_model
   }
   
   with open(sim_dir / 'simulation_info.json', 'w') as f:
       json.dump(config_info, f, indent=4)
       
   return sim_dir