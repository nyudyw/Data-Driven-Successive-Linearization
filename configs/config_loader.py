import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent / 'default_config.yaml'
        else:
            config_path = Path(__file__).parent / config_path
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_controller_config(self):
        return self.config['controller']
    
    def get_optimization_config(self):
        return self.config['optimization']
    
    def get_config_name(self):
        return self.config['config_name']
    
    def get_model_config(self):
        return self.config['model']
    
    @staticmethod
    def merge_configs(default_config, custom_config):
        """Merge defaut_config and custom_config"""
        merged = default_config.copy()
        merged.update(custom_config)
        return merged