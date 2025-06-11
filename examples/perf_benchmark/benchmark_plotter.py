import glob
import os
import html
import argparse

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

def get_comparison_data_set():
    return [
        (("batch_renderer", True), ("pyrender", True)),
        (("batch_renderer", True), ("batch_renderer", False)),
    ]

def plot_batch_benchmark(data_file_path, width=20, height=15):
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
        for renderer_info_1, renderer_info_2 in get_comparison_data_set():
            generate_comparison_plots(df, plots_dir, width, height, renderer_info_1, renderer_info_2, aspect_ratio=aspect_ratio)

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
                bar_width = 0.8 / len(resolutions)  # Adjust bar width based on number of resolutions
                
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

def generate_comparison_plots(df, plots_dir, width, height, renderer_info_1, renderer_info_2, aspect_ratio=None):
    renderer_1_name, renderer_1_is_rasterizer = renderer_info_1
    renderer_2_name, renderer_2_is_rasterizer = renderer_info_2
    rasterizer_1_str = 'rasterizer' if renderer_1_is_rasterizer else 'raytracer'
    rasterizer_2_str = 'rasterizer' if renderer_2_is_rasterizer else 'raytracer'

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
        renderer_1_res = set(zip(mjcf_data[(mjcf_data['renderer'] == renderer_1_name) & (mjcf_data['rasterizer'] == renderer_1_is_rasterizer)]['resX'], 
                                mjcf_data[(mjcf_data['renderer'] == renderer_1_name) & (mjcf_data['rasterizer'] == renderer_1_is_rasterizer)]['resY']))
        renderer_2_res = set(zip(mjcf_data[(mjcf_data['renderer'] == renderer_2_name) & (mjcf_data['rasterizer'] == renderer_2_is_rasterizer)]['resX'],
                               mjcf_data[(mjcf_data['renderer'] == renderer_2_name) & (mjcf_data['rasterizer'] == renderer_2_is_rasterizer)]['resY']))
        common_res = renderer_1_res.intersection(renderer_2_res)
        
        # continue if there is no data
        if len(common_res) == 0:
            print(f"No data found for {mjcf}")
            continue

        # Plot comparison for each resolution
        for resX, resY in sorted(common_res, key=lambda x: x[0] * x[1]):
            plt.figure(figsize=(width, height))
            
            renderer_1_data = mjcf_data[(mjcf_data['renderer'] == renderer_1_name) & 
                                 (mjcf_data['rasterizer'] == renderer_1_is_rasterizer) &
                                 (mjcf_data['resX'] == resX) & 
                                 (mjcf_data['resY'] == resY)]
            renderer_2_data = mjcf_data[(mjcf_data['renderer'] == renderer_2_name) &
                                (mjcf_data['rasterizer'] == renderer_2_is_rasterizer) &
                                (mjcf_data['resX'] == resX) &
                                (mjcf_data['resY'] == resY)]
            
            # Match batch sizes and calculate difference
            common_batch = set(renderer_1_data['n_envs']).intersection(set(renderer_2_data['n_envs']))
            batch_sizes = sorted(list(common_batch))
            
            renderer_1_fps = renderer_1_data[renderer_1_data['n_envs'].isin(batch_sizes)]['fps'].values
            renderer_2_fps = renderer_2_data[renderer_2_data['n_envs'].isin(batch_sizes)]['fps'].values
            diff_fps = renderer_1_fps / renderer_2_fps

            # Create bar chart
            x = np.arange(len(batch_sizes))
            bar_width = 0.35

            # Plot bars
            bars1 = plt.bar(x - bar_width/2, renderer_1_fps, bar_width, label=f'{renderer_1_name} {rasterizer_1_str}')
            bars2 = plt.bar(x + bar_width/2, renderer_2_fps, bar_width, label=f'{renderer_2_name} {rasterizer_2_str}')

            # Add value labels on top of bars
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.1f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)

            add_labels(bars1)
            add_labels(bars2)

            # Customize plot
            plt.title(f'FPS Comparison: {renderer_1_name} {rasterizer_1_str} vs {renderer_2_name} {rasterizer_2_str}\n{os.path.basename(mjcf)} - Resolution: {resX}x{resY}')
            plt.xlabel('Batch Size')
            plt.ylabel('FPS')
            plt.xticks(x, batch_sizes)
            plt.legend()
            plt.grid(True, axis='y')

            # Save plot
            plot_filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_{renderer_1_name}_{rasterizer_1_str}_{renderer_2_name}_{rasterizer_2_str}_{resX}x{resY}_comparison_plot.png"
            plt.savefig(plot_filename, dpi=100)  # Added dpi parameter for better quality
            plt.close()

def main():
    import sys
    import os
    print("Script arguments:", sys.argv)  # Debug print
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file_path", type=str, default="logs/benchmark/batch_benchmark_20250610_160138_combined.csv",
                       help="Path to the benchmark data CSV file")
    parser.add_argument("-w", "--width", type=int, default=20,
                       help="Width of the plot in inches")
    parser.add_argument("-y", "--height", type=int, default=15,
                       help="Height of the plot in inches")
    
    # If no arguments provided, try to get from environment variables
    if len(sys.argv) == 1:
        data_file = os.environ.get('BENCHMARK_DATA_FILE')
        if data_file:
            sys.argv.extend(['-d', data_file])
    
    args = parser.parse_args()
    print("Parsed arguments:", args)  # Debug print
    plot_batch_benchmark(args.data_file_path, args.width, args.height)

if __name__ == "__main__":
    main()
