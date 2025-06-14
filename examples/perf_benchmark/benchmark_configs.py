import os
import yaml

class BenchmarkConfigs:
    def __init__(self, config_file):
        self.load_from_config_file(config_file)
    
    def load_from_config_file(self, config_file):
        self.config_path = os.path.join(os.path.dirname(__file__), config_file)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

            self.mjcf_list = config['mjcf_list']
            self.rasterizer_list = config['rasterizer_list']
            self.batch_size_list = config['batch_size_list']
            self.resolution_list = config['resolution_list']
            self.gui = config.get('gui', False)

            # Get renderer list with defaults
            renderer_list = config.get('renderer_list', [])
            for renderer in renderer_list:
                renderer_name = renderer.get('renderer', 'madrona')
                renderer_timeout = renderer.get('timeout', 120)
                self.renderer_list.append((renderer_name, renderer_timeout))
        
            # Get raytracer config with defaults
            raytracer_config = config.get('raytracer', {})
            self.max_bounce = raytracer_config.get('max_bounce', 2)
            self.spp = raytracer_config.get('spp', 1)

            # Get simulation config with defaults
            simulation_config = config.get('simulation', {})
            self.n_steps = simulation_config.get('n_steps', 1)

            # Get camera config with defaults
            camera_config = config.get('camera', {})
            self.camera_pos = camera_config.get('position', [1.5, 0.5, 1.5])
            self.camera_lookat = camera_config.get('lookat', [0.0, 0.0, 0.5])
            self.camera_fov = camera_config.get('fov', 45.0)

            # Get display config with defaults
            display_config = config.get('display', {})
            self.gui = display_config.get('gui', False)