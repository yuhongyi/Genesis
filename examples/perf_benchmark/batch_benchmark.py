from benchmark_plotter import plot_batch_benchmark
import argparse
import subprocess
import os
from datetime import datetime
import pandas as pd
import yaml

# Create a struct to store the arguments
class BenchmarkArgs:
    def __init__(self,
                 renderer_name, rasterizer, n_envs, n_steps, resX, resY,
                 camera_posX, camera_posY, camera_posZ,
                 camera_lookatX, camera_lookatY, camera_lookatZ,
                 camera_fov, mjcf, benchmark_result_file_path,
                 max_bounce=2, spp=1, gui=False):
        self.renderer_name = renderer_name
        self.rasterizer = rasterizer
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.resX = resX
        self.resY = resY
        self.camera_posX = camera_posX
        self.camera_posY = camera_posY
        self.camera_posZ = camera_posZ
        self.camera_lookatX = camera_lookatX
        self.camera_lookatY = camera_lookatY
        self.camera_lookatZ = camera_lookatZ
        self.camera_fov = camera_fov
        self.mjcf = mjcf
        self.benchmark_result_file_path = benchmark_result_file_path
        self.max_bounce = max_bounce
        self.spp = spp
        self.gui = gui

    @staticmethod
    def parse_benchmark_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--renderer_name", type=str, default="madrona")
        parser.add_argument("-r", "--rasterizer", action="store_true", default=False)
        parser.add_argument("-n", "--n_envs", type=int, default=1024)
        parser.add_argument("-s", "--n_steps", type=int, default=1)
        parser.add_argument("-x", "--resX", type=int, default=1024)
        parser.add_argument("-y", "--resY", type=int, default=1024)
        parser.add_argument("-i", "--camera_posX", type=float, default=1.5)
        parser.add_argument("-j", "--camera_posY", type=float, default=0.5)
        parser.add_argument("-k", "--camera_posZ", type=float, default=1.5)
        parser.add_argument("-l", "--camera_lookatX", type=float, default=0.0)
        parser.add_argument("-m", "--camera_lookatY", type=float, default=0.0)
        parser.add_argument("-o", "--camera_lookatZ", type=float, default=0.5)
        parser.add_argument("-v", "--camera_fov", type=float, default=45)
        parser.add_argument("-f", "--mjcf", type=str, default="xml/franka_emika_panda/panda.xml")
        parser.add_argument("-g", "--benchmark_result_file_path", type=str, default="benchmark.csv")
        parser.add_argument("-b", "--max_bounce", type=int, default=2)
        parser.add_argument("-p", "--spp", type=int, default=64)
        parser.add_argument("-t", "--gui", action="store_true", default=False)
        args = parser.parse_args()
        benchmark_args = BenchmarkArgs(
            renderer_name=args.renderer_name,
            rasterizer=args.rasterizer,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            resX=args.resX,
            resY=args.resY,
            camera_posX=args.camera_posX,
            camera_posY=args.camera_posY,
            camera_posZ=args.camera_posZ,
            camera_lookatX=args.camera_lookatX,
            camera_lookatY=args.camera_lookatY,
            camera_lookatZ=args.camera_lookatZ,
            camera_fov=args.camera_fov,
            mjcf=args.mjcf,
            benchmark_result_file_path=args.benchmark_result_file_path,
            max_bounce=args.max_bounce,
            spp=args.spp,
            gui=args.gui,
        )
        print(f"Benchmark with args:")
        print(f"  renderer_name: {benchmark_args.renderer_name}")
        print(f"  rasterizer: {benchmark_args.rasterizer}")
        print(f"  n_envs: {benchmark_args.n_envs}")
        print(f"  n_steps: {benchmark_args.n_steps}")
        print(f"  resolution: {benchmark_args.resX}x{benchmark_args.resY}")
        print(f"  camera_pos: ({benchmark_args.camera_posX}, {benchmark_args.camera_posY}, {benchmark_args.camera_posZ})")
        print(f"  camera_lookat: ({benchmark_args.camera_lookatX}, {benchmark_args.camera_lookatY}, {benchmark_args.camera_lookatZ})")
        print(f"  camera_fov: {benchmark_args.camera_fov}")
        print(f"  mjcf: {benchmark_args.mjcf}")
        print(f"  benchmark_result_file_path: {benchmark_args.benchmark_result_file_path}")
        print(f"  max_bounce: {benchmark_args.max_bounce}")
        print(f"  spp: {benchmark_args.spp}")
        print(f"  gui: {benchmark_args.gui}")
        return benchmark_args
    
class BatchBenchmarkArgs:
    def __init__(self, config_file, continue_from):
        self.config_file = config_file
        self.continue_from = continue_from

    def parse_batch_benchmark_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--config_file", type=str, default="benchmark_config_minimal.yml")
        parser.add_argument("-c", "--continue_from", type=str, default=None)
        args = parser.parse_args()
        return BatchBenchmarkArgs(
            config_file=args.config_file,
            continue_from=args.continue_from
            )

def load_benchmark_config(config_file):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get rendering config with defaults
    rendering_config = config.get('rendering', {})
    max_bounce = rendering_config.get('max_bounce', 2)
    spp = rendering_config.get('spp', 1)

    # Get simulation config with defaults
    simulation_config = config.get('simulation', {})
    n_steps = simulation_config.get('n_steps', 1)

    # Get camera config with defaults
    camera_config = config.get('camera', {})
    camera_pos = camera_config.get('position', [1.5, 0.5, 1.5])
    camera_lookat = camera_config.get('lookat', [0.0, 0.0, 0.5])
    camera_fov = camera_config.get('fov', 45.0)

    # Get display config with defaults
    display_config = config.get('display', {})
    gui = display_config.get('gui', False)
    
    return {
        'mjcf_list': config['mjcf_list'],
        'renderer_list': config['renderer_list'],
        'rasterizer_list': config['rasterizer_list'],
        'batch_size_list': config['batch_size_list'],
        'resolution_list': config['resolution_list'],
        'timeout': config.get('timeout', 120),  # Default to 120 if not specified
        'max_bounce': max_bounce,
        'spp': spp,
        'n_steps': n_steps,
        'camera_pos': camera_pos,
        'camera_lookat': camera_lookat,
        'camera_fov': camera_fov,
        'gui': gui
    }

def create_batch_args(benchmark_result_file_path, config_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(benchmark_result_file_path), exist_ok=True)
    
    # Load configuration
    config = load_benchmark_config(config_file)
    mjcf_list = config['mjcf_list']
    renderer_list = config['renderer_list']
    rasterizer_list = config['rasterizer_list']
    batch_size_list = config['batch_size_list']
    resolution_list = config['resolution_list']
    n_steps = config['n_steps']
    camera_pos = config['camera_pos']
    camera_lookat = config['camera_lookat']
    camera_fov = config['camera_fov']
    max_bounce = config['max_bounce']
    spp = config['spp']
    gui = config['gui']

    # Batch data for resolution and batch size needs to be sorted in ascending order of resX x resY
    # so that if one resolution fails, all the resolutions, which are larger, will be skipped.
    resolution_list.sort(key=lambda x: x[0] * x[1])

    # Create a hierarchical dictionary to store all combinations
    batch_args_dict = {}

    # Build hierarchical structure
    for renderer in renderer_list:
        batch_args_dict[renderer] = {}
        for rasterizer in rasterizer_list:
                batch_args_dict[renderer][rasterizer] = {}
                for mjcf in mjcf_list:
                    batch_args_dict[renderer][rasterizer][mjcf] = {}
                    for batch_size in batch_size_list:
                        batch_args_dict[renderer][rasterizer][mjcf][batch_size] = {}
                        for resolution in resolution_list:
                            resX, resY = resolution
                            # Create benchmark args for this combination
                            args = BenchmarkArgs(
                                renderer_name=renderer,
                                rasterizer=rasterizer,
                                n_envs=batch_size,
                                n_steps=n_steps,
                                resX=resX,
                                resY=resY,
                                camera_posX=camera_pos[0],
                                camera_posY=camera_pos[1],
                                camera_posZ=camera_pos[2],
                                camera_lookatX=camera_lookat[0],
                                camera_lookatY=camera_lookat[1],
                                camera_lookatZ=camera_lookat[2],
                                camera_fov=camera_fov,
                                mjcf=mjcf,
                                benchmark_result_file_path=benchmark_result_file_path,
                                max_bounce=max_bounce,
                                spp=spp,
                                gui=gui
                            )
                            batch_args_dict[renderer][rasterizer][mjcf][batch_size][(resX,resY)] = args

    return batch_args_dict

def create_benchmark_result_file(continue_from_file_path):
    if continue_from_file_path is not None:
        if not os.path.exists(continue_from_file_path):
            raise FileNotFoundError(f"Continue from file not found: {continue_from_file_path}")
        print(f"Continuing from file: {continue_from_file_path}")
        return continue_from_file_path
    else:
        # Create benchmark result data file with header
        benchmark_data_directory = "logs/benchmark"
        if not os.path.exists(benchmark_data_directory):
            os.makedirs(benchmark_data_directory)
        benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_result_file_path = f"{benchmark_data_directory}/batch_benchmark_{benchmark_timestamp}.csv"
        with open(benchmark_result_file_path, "w") as f:
            f.write("result,mjcf,renderer,rasterizer,n_envs,n_steps,resX,resY,camera_posX,camera_posY,camera_posZ,camera_lookatX,camera_lookatY,camera_lookatZ,camera_fov,time_taken,time_taken_per_env,fps,fps_per_env\n")
        print(f"Created new benchmark result file: {benchmark_result_file_path}")
        return benchmark_result_file_path

def get_previous_runs(continue_from_file_path):
    if continue_from_file_path is None:
        return []
    
    # Read the existing benchmark data file
    df = pd.read_csv(continue_from_file_path)
    
    # Create a list of tuples containing run info and status
    previous_runs = []
    
    for _, row in df.iterrows():
        run_info = (
            row['mjcf'],
            row['renderer'],
            row['n_envs'],
            (row['resX'], row['resY']),
            row['result']  # 'succeeded' or 'failed'
        )
        previous_runs.append(run_info)
    
    return previous_runs

def get_benchmark_script_path(renderer_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))    
    if renderer_name == "madrona":
        return f"{current_dir}/benchmark_madrona.py"
    elif renderer_name == "pyrender":
        return f"{current_dir}/benchmark_pyrender.py"
    elif renderer_name == "omniverse":
        return f"{current_dir}/benchmark_omni.py"
    else:
        raise ValueError(f"Invalid renderer name: {renderer_name}")

def run_batch_benchmark(batch_args_dict, previous_runs=None):
    if previous_runs is None:
        previous_runs = []
    
    for renderer in batch_args_dict:
        benchmark_script_path = get_benchmark_script_path(renderer)    
        if not os.path.exists(benchmark_script_path):
            raise FileNotFoundError(f"Benchmark script not found: {benchmark_script_path}")
        print(f"Running benchmark for {renderer}")
        
        for rasterizer in batch_args_dict[renderer]:
            for mjcf in batch_args_dict[renderer][rasterizer]:
                for batch_size in batch_args_dict[renderer][rasterizer][mjcf]:
                    last_resolution_failed = False
                    for resolution in batch_args_dict[renderer][rasterizer][mjcf][batch_size]:
                        if last_resolution_failed:
                            break
                        
                        # Check if this run was in a previous execution
                        run_info = (mjcf, rasterizer, batch_size, resolution)
                        skip_this_run = False
                        
                        for prev_run in previous_runs:
                            if run_info == prev_run[:4]:  # Compare only the run parameters, not the status
                                skip_this_run = True
                                if prev_run[4] == 'failed':
                                    # Skip this and subsequent resolutions if it failed before
                                    last_resolution_failed = True
                                break
                        
                        if skip_this_run:
                            continue
                            
                        # Run the benchmark
                        batch_args = batch_args_dict[renderer][rasterizer][mjcf][batch_size][resolution]
                        
                        # launch a process to run the benchmark
                        cmd = ["python3", benchmark_script_path]
                        if batch_args.rasterizer:
                            cmd.append("--rasterizer")
                        if batch_args.gui:
                            cmd.append("--gui")
                        cmd.extend([
                            "--renderer_name", batch_args.renderer_name,
                            "--n_envs", str(batch_args.n_envs),
                            "--n_steps", str(batch_args.n_steps),
                            "--resX", str(batch_args.resX), 
                            "--resY", str(batch_args.resY),
                            "--camera_posX", str(batch_args.camera_posX),
                            "--camera_posY", str(batch_args.camera_posY),
                            "--camera_posZ", str(batch_args.camera_posZ), 
                            "--camera_lookatX", str(batch_args.camera_lookatX),
                            "--camera_lookatY", str(batch_args.camera_lookatY),
                            "--camera_lookatZ", str(batch_args.camera_lookatZ),
                            "--camera_fov", str(batch_args.camera_fov),
                            "--mjcf", batch_args.mjcf,
                            "--benchmark_result_file_path", batch_args.benchmark_result_file_path,
                            "--max_bounce", str(batch_args.max_bounce),
                            "--spp", str(batch_args.spp),
                        ])
                        try:
                            # Read timeout from config
                            process = subprocess.Popen(cmd)
                            try:
                                # Hack to avoid omniverse runs to take forever.
                                timeout = 30 if batch_args.renderer_name == "omniverse" else None
                                return_code = process.wait(timeout=timeout)
                                if return_code != 0:
                                    raise subprocess.CalledProcessError(return_code, cmd)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()  # Wait for the process to be killed
                                raise TimeoutError(f"Process did not complete within {timeout} seconds")
                        except Exception as e:
                            print(f"Error running benchmark: {str(e)}")
                            last_resolution_failed = True
                            # Write failed result without timing data
                            with open(batch_args.benchmark_result_file_path, 'a') as f:
                                f.write(f'failed,{batch_args.mjcf},{batch_args.renderer_name},{batch_args.rasterizer},{batch_args.n_envs},{batch_args.n_steps},{batch_args.resX},{batch_args.resY},{batch_args.camera_posX},{batch_args.camera_posY},{batch_args.camera_posZ},{batch_args.camera_lookatX},{batch_args.camera_lookatY},{batch_args.camera_lookatZ},{batch_args.camera_fov},,,,\n')
                            break

def sort_and_dedupe_benchmark_result_file(benchmark_result_file_path):
    # Sort by mjcf asc, renderer asc, rasterizer desc, n_envs asc, resX asc, resY asc, n_envs asc
    df = pd.read_csv(benchmark_result_file_path)
    df = df.sort_values(
        by=['mjcf', 'renderer', 'rasterizer', 'resX', 'resY', 'n_envs', 'result'],
        ascending=[True, True, False, True, True, True, False]
    )

    # Deduplicate by keeping the first occurrence of each unique combination of mjcf, renderer, rasterizer, resX, resY, n_envs
    # Keep succeeded runs if there are multiple runs for the same combination.
    df = df.drop_duplicates(
        subset=['mjcf', 'renderer', 'rasterizer', 'resX', 'resY', 'n_envs'],
        keep='first'
    )
    df.to_csv(benchmark_result_file_path, index=False)

def main():
    batch_benchmark_args = BatchBenchmarkArgs.parse_batch_benchmark_args()
    benchmark_result_file_path = create_benchmark_result_file(batch_benchmark_args.continue_from)
    
    # Get list of previous runs if continuing from a previous run
    previous_runs = get_previous_runs(batch_benchmark_args.continue_from)

    # Run benchmark in batch        
    batch_args_dict = create_batch_args(benchmark_result_file_path, config_file=batch_benchmark_args.config_file)
    run_batch_benchmark(batch_args_dict, previous_runs)

    # Sort benchmark result file
    sort_and_dedupe_benchmark_result_file(benchmark_result_file_path)
    
    # Generate plots
    plot_batch_benchmark(benchmark_result_file_path, config_file=batch_benchmark_args.config_file)

if __name__ == "__main__":
    main()
