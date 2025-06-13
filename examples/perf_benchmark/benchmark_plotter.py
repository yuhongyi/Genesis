import glob
import os
import html
import argparse
import yaml

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def generatePlotHtml(plots_dir):
    #Generate an html page to display all the plots

    # Get all plot files
    plot_files = glob.glob(f"{plots_dir}/*.png")
    if len(plot_files) == 0:
        print(f"No plot files found in {plots_dir}")
        return
    
    # Separate regular plots from comparison charts
    regular_plot_files = [p for p in plot_files if p.endswith('_plot.png') and not p.endswith('_comparison_plot.png')]
    
    # Group regular plots by MJCF file
    plot_groups = {}
    for plot_file in regular_plot_files:
        basename = os.path.basename(plot_file)
        mjcf_name = basename.split('_')[0]
        if mjcf_name not in plot_groups:
            plot_groups[mjcf_name] = []
        plot_groups[mjcf_name].append(plot_file)

    # Sort plot groups by mjcf name and plot file name
    plot_groups = sorted(plot_groups.items(), key=lambda x: (x[0], x[1][0]))
    
    # Group comparison plots by resolution
    comparison_plot_files = {}
    for plot_file in plot_files:
        if plot_file.endswith('_comparison_plot.png'):
            # Extract resolution from filename (e.g., "128x128" from "..._128x128_comparison_plot.png")
            resolution = plot_file.split('_')[-3]  # Get the resolution part
            if resolution not in comparison_plot_files:
                comparison_plot_files[resolution] = []
            comparison_plot_files[resolution].append(plot_file)
    
    # Sort resolutions by their dimensions
    def get_resolution_dims(res):
        width, height = map(int, res.split('x'))
        return width * height  # Sort by total pixels
    
    sorted_resolutions = sorted(comparison_plot_files.keys(), key=get_resolution_dims)

    # Create HTML file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .plot-container { margin-bottom: 40px; }
            .section { margin-bottom: 60px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            h1 { color: #333; }
            h2 { color: #333; }
            h3 { color: #666; margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Benchmark Results</h1>
    """

    # Add comparison plots sections by resolution
    if comparison_plot_files:
        html_content += "<div class='section'>\n"
        html_content += "<h2>Performance Comparison Plots</h2>\n"
        for resolution in sorted_resolutions:
            html_content += f"<h3>Resolution: {resolution}</h3>\n"
            html_content += "<div class='plot-container'>\n"
            for plot in comparison_plot_files[resolution]:
                html_content += f"<img src='{html.escape(os.path.basename(plot))}' alt='{html.escape(os.path.basename(plot))}'/><br/>\n"
            html_content += "</div>\n"
        html_content += "</div>\n"

    # Add regular plots section
    html_content += "<div class='section'>\n"
    html_content += "<h2>Performance Plots</h2>\n"
    for mjcf_name, plots in plot_groups:
        html_content += f"<div class='plot-container'>\n"
        for plot in plots:
            html_content += f"<h3>{html.escape(mjcf_name)} - {os.path.basename(plot)}</h3>\n"
            html_content += f"<img src='{html.escape(os.path.basename(plot))}' alt='{html.escape(os.path.basename(plot))}'/><br/>\n"
        html_content += "</div>\n"
    html_content += "</div>\n"

    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(f"{plots_dir}/index.html", 'w') as f:
        f.write(html_content)

def load_benchmark_config(config_file):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_comparison_data_set(config_file):
    config = load_benchmark_config(config_file)
    comparison_sets = []
    
    if(config['comparison_sets'] is None):
        return []
    
    for comparison_set in config['comparison_sets']:
        set_tuples = []
        for config_item in comparison_set:
            set_tuples.append((config_item['renderer'], config_item['rasterizer']))
        comparison_sets.append(tuple(set_tuples))
    
    return comparison_sets

def plot_batch_benchmark(data_file_path, config_file, width=20, height=15):
    # Load the log file as csv
    # For each mjcf, rasterizer (rasterizer or not(=raytracer)), generate a plot image and save it to a directory.
    # The plot image has batch size on the x-axis and fps on the y-axis.
    # Each resolution has a different color.
    # The plot image has a legend for the resolution.
    # The plot image has a title for the mjcf.
    # The plot image has a x-axis label for the batch size.
    # The plot image has a y-axis label for the fps.

    # Read CSV file
    df = pd.read_csv(data_file_path)

    # Create plots directory if it doesn't exist
    plots_dir = f"logs/benchmark/plots/{os.path.basename(data_file_path)}"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    # Generate individual plots for each mjcf/rasterizer combination
    generate_individual_plots(df, plots_dir, width, height)

    # Generate difference plots for specific aspect ratios
    for aspect_ratio in ["1:1", "4:3", "16:9"]:
        for renderer_info_array in get_comparison_data_set(config_file):
            generate_comparison_plots(df, plots_dir, width, height, renderer_info_array, aspect_ratio)

    # Generate an html page to display all the plots
    generatePlotHtml(plots_dir)

def generate_individual_plots(df, plots_dir, width, height):
    # Get unique combinations of mjcf and rasterizer
    for mjcf in df['mjcf'].unique():
        for renderer in df[df['mjcf'] == mjcf]['renderer'].unique():
            for rasterizer in df[(df['mjcf'] == mjcf) & (df['renderer'] == renderer)]['rasterizer'].unique():
                # Filter data for this mjcf and rasterizer
                data = df[(df['mjcf'] == mjcf) & (df['renderer'] == renderer) & (df['rasterizer'] == rasterizer)]

                # continue if there is no data
                if len(data) == 0:
                    print(f"No data found for {mjcf} and {renderer} and rasterizer:{rasterizer}")
                    continue
                
                # Create new figure
                plt.figure(figsize=(width, height))
                
                # Group data by resolution
                resolutions = sorted(data.groupby(['resX', 'resY']), key=lambda x: (x[0][0], x[0][1]))
                
                # Get all batch sizes
                all_batch_sizes = sorted(data['n_envs'].unique())
                
                # Create bar chart
                x = np.arange(len(all_batch_sizes))
                bar_width = min(0.1, 0.8 / len(resolutions))
                
                # Plot bars for each resolution
                for i, (resolution, res_data) in enumerate(resolutions):
                    # Create mapping from batch size to index
                    batch_to_idx = {batch: idx for idx, batch in enumerate(all_batch_sizes)}
                    
                    # Create array of FPS for all batch sizes
                    fps_array = np.zeros(len(all_batch_sizes))
                    for batch, fps in zip(res_data['n_envs'], res_data['fps']):
                        fps_array[batch_to_idx[batch]] = fps
                    
                    # Plot bars
                    bars = plt.bar(x + i * bar_width, fps_array, bar_width, 
                           label=f'{resolution[0]}x{resolution[1]}')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        bar_height = bar.get_height()
                        if bar_height > 0:  # Only add label if there's a value
                            plt.annotate(f'{bar_height:.1f}',
                                      xy=(bar.get_x() + bar.get_width() / 2, bar_height),
                                      xytext=(0, 3),  # 3 points vertical offset
                                      textcoords="offset points",
                                      ha='center', va='bottom', fontsize=8)
                
                # Customize plot
                plt.title(f'Performance for {os.path.basename(mjcf)}\n{renderer} {"Rasterizer" if rasterizer else "Raytracer"}')
                plt.xlabel('Batch Size')
                plt.ylabel('FPS')
                plt.xticks(x + bar_width * (len(resolutions) - 1) / 2, all_batch_sizes)
                plt.legend(title='Resolution')
                plt.grid(True, axis='y')

                # Save plot
                plot_filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_{renderer}_{'rasterizer' if rasterizer else 'raytracer'}_plot.png"
                plt.savefig(plot_filename)
                plt.close()

def generate_comparison_plots(df, plots_dir, width, height, renderer_info_array, aspect_ratio=None):
    renderer_name_array = [renderer_info[0] for renderer_info in renderer_info_array]
    renderer_is_rasterizer_array = [renderer_info[1] for renderer_info in renderer_info_array]
    rasterizer_str_array = ['rasterizer' if renderer_is_rasterizer else 'raytracer' for renderer_is_rasterizer in renderer_is_rasterizer_array]

    # Filter by aspect ratio if specified
    if aspect_ratio:
        if aspect_ratio == "1:1":
            df = df[df['resX'] == df['resY']]
        elif aspect_ratio == "4:3":
            df = df[df['resX'] * 3 == df['resY'] * 4]
        elif aspect_ratio == "16:9":
            df = df[df['resX'] * 9 == df['resY'] * 16]
        else:
            raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    plt.clf()
    plt.cla()

    # Generate plots showing fps comparison between renderer_1 and renderer_2
    for mjcf in df['mjcf'].unique():
        mjcf_data = df[df['mjcf'] == mjcf]
        
        # Get resolutions available for both renderer_1 and renderer_2
        renderer_resolutions = [set(zip(mjcf_data[(mjcf_data['renderer'] == renderer_name) & (mjcf_data['rasterizer'] == renderer_is_rasterizer)]['resX'], 
                                mjcf_data[(mjcf_data['renderer'] == renderer_name) & (mjcf_data['rasterizer'] == renderer_is_rasterizer)]['resY'])) for renderer_name, renderer_is_rasterizer in renderer_info_array]
        common_res = set.intersection(*renderer_resolutions)
        
        # continue if there is no data
        if len(common_res) == 0:
            print(f"No data found for {mjcf}")
            continue

        # Plot comparison for each resolution
        for resX, resY in sorted(common_res, key=lambda x: x[0] * x[1]):
            plt.figure(figsize=(width, height))
            renderer_data_array = []
            for renderer_name, renderer_is_rasterizer in renderer_info_array:
                renderer_data = mjcf_data[(mjcf_data['renderer'] == renderer_name) & 
                                (mjcf_data['rasterizer'] == renderer_is_rasterizer) &
                                (mjcf_data['resX'] == resX) & 
                                (mjcf_data['resY'] == resY)]
                renderer_data_array.append(renderer_data)
            
            # Match batch sizes and calculate difference
            common_batch = set.intersection(*[set(renderer_data['n_envs']) for renderer_data in renderer_data_array])
            batch_sizes = sorted(list(common_batch))
            
            fps_array = []
            for renderer_data in renderer_data_array:
                fps_array.append(renderer_data[renderer_data['n_envs'].isin(batch_sizes)]['fps'].values)

            # Create bar chart
            x = np.arange(len(batch_sizes))
            bar_width = min(0.1, 0.8 / len(renderer_info_array))

            # Plot bars
            bar_groups = [plt.bar(x + i * bar_width, fps, bar_width, label=f'{renderer_name} {rasterizer_str}') for i, (fps, renderer_name, rasterizer_str) in enumerate(zip(fps_array, renderer_name_array, rasterizer_str_array))]

            # Add value labels on top of bars
            def add_labels(bars):
                for bar in bars:
                    bar_height = bar.get_height()
                    plt.annotate(f'{bar_height:.1f}',
                              xy=(bar.get_x() + bar.get_width() / 2, bar_height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)

            for bar_group in bar_groups:
                add_labels(bar_group)

            # Customize plot
            renderer_str_array = [f'{renderer_name} {rasterizer_str}' for renderer_name, rasterizer_str in zip(renderer_name_array, rasterizer_str_array)]
            renderer_str_array_str = ', '.join(renderer_str_array)
            plt.title(f'FPS Comparison: {renderer_str_array_str}\n{os.path.basename(mjcf)} - Resolution: {resX}x{resY}')
            plt.xlabel('Batch Size')
            plt.ylabel('FPS')
            plt.xticks(x, batch_sizes)
            plt.legend()
            plt.grid(True, axis='y')

            # Save plot
            plot_filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_{renderer_str_array_str}_{resX}x{resY}_comparison_plot.png"
            plt.savefig(plot_filename, dpi=100)  # Added dpi parameter for better quality
            plt.close()

def main():
    import sys
    import os
    print("Script arguments:", sys.argv)  # Debug print
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file_path", type=str, default="logs/benchmark/batch_benchmark_20250613_131628.csv",
                       help="Path to the benchmark data CSV file")
    parser.add_argument("-c", "--config_file", type=str, default="benchmark_config_minimal.yml",
                       help="Path to the benchmark config file")
    parser.add_argument("-w", "--width", type=int, default=20,
                       help="Width of the plot in inches")
    parser.add_argument("-y", "--height", type=int, default=8,
                       help="Height of the plot in inches")
    
    # If no arguments provided, try to get from environment variables
    if len(sys.argv) == 1:
        data_file = os.environ.get('BENCHMARK_DATA_FILE')
        if data_file:
            sys.argv.extend(['-d', data_file])
    
    args = parser.parse_args()
    print("Parsed arguments:", args)  # Debug print
    plot_batch_benchmark(args.data_file_path, args.config_file, args.width, args.height)

if __name__ == "__main__":
    main()
