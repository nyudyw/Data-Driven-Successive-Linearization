import os
import json
import numpy as np

"""
load and save pypower standard cases
"""

def load_case_from_files(load_dir,case_name):
    """
    Load case data from saved files
    
    Parameters:
    -----------
    load_dir : str
        Directory path where files are stored
    case_name : str
        Name of the case to be loaded
        
    Returns:
    --------
    dict : Dictionary in PYPOWER case format
    """
    case = {}
    load_dir = os.path.join(load_dir, case_name)
    case['bus'] = np.loadtxt(os.path.join(load_dir, 'bus.csv'), delimiter=',', skiprows=1)
    case['branch'] = np.loadtxt(os.path.join(load_dir, 'branch.csv'), delimiter=',', skiprows=1)
    case['gen'] = np.loadtxt(os.path.join(load_dir, 'gen.csv'), delimiter=',', skiprows=1)

    case['gen'] = np.atleast_2d(case['gen'])
    
    with open(os.path.join(load_dir, 'system_info.json'), 'r') as f:
        system_info = json.load(f)
    
    case.update(system_info)
    
    return case

def save_case_to_files(case_data, save_dir,case_name):
    """
    Save PYPOWER case data to multiple files in the specified directory
    
    Parameters:
    -----------
    case_data : dict
        PYPOWER case dictionary data
    save_dir : str
        Directory path to save files
    case_name : str
        Name of the case to be saved
    """
    save_dir = save_dir+'/'+case_name
    os.makedirs(save_dir, exist_ok=True)
    
    np.savetxt(os.path.join(save_dir, 'bus.csv'), 
               case_data['bus'], 
               delimiter=',',
               header='bus_i,type,Pd,Qd,Gs,Bs,area,Vm,Va,baseKV,zone,Vmax,Vmin')
    
    np.savetxt(os.path.join(save_dir, 'branch.csv'), 
               case_data['branch'], 
               delimiter=',',
               header='fbus,tbus,r,x,b,rateA,rateB,rateC,tap,shift,status,angmin,angmax')
    
    np.savetxt(os.path.join(save_dir, 'gen.csv'), 
               case_data['gen'], 
               delimiter=',',
               header='bus,Pg,Qg,Qmax,Qmin,Vg,mBase,status,Pmax,Pmin')
    
    system_data = {
        'baseMVA': case_data['baseMVA'],
        'version': case_data['version'],
        'name': case_data.get('name', 'Unknown')
    }
    
    with open(os.path.join(save_dir, 'system_info.json'), 'w') as f:
        json.dump(system_data, f, indent=4)
    
    readme_content = f"""PYPOWER Case Data
Name: {case_data.get('name', 'Unknown')}
Base MVA: {case_data['baseMVA']}
Number of buses: {len(case_data['bus'])}
Number of branches: {len(case_data['branch'])}
Number of generators: {len(case_data['gen'])}

Files:
- bus.csv: Bus data
- branch.csv: Branch data
- gen.csv: Generator data
- system_info.json: System base values and other information
"""
    
    with open(os.path.join(save_dir, 'README.txt'), 'w') as f:
        f.write(readme_content)
