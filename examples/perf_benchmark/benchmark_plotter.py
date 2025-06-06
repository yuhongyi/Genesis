import glob
import os
import html
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def generatePlotHtml(plots_dir):
    #Generate an html page to display all the plots

    # Get all plot files
    plot_files = glob.glob(f"{plots_dir}/*.png")
    if len(plot_files) == 0:
        print(f"No plot files found in {plots_dir}")
        return
    
    # Separate regular plots from difference plots
    regular_plots = [p for p in plot_files if not p.endswith('_difference.png') and not any(p.endswith(f'_difference_{ar.replace(":", "x")}.png') for ar in ["1:1", "4:3", "16:9"])]
    aspect_ratio_plots = {
        "1:1": [p for p in plot_files if p.endswith('_difference_1x1.png')],
        "4:3": [p for p in plot_files if p.endswith('_difference_4x3.png')],
        "16:9": [p for p in plot_files if p.endswith('_difference_16x9.png')]
    }
    
    # Group regular plots by MJCF file
    plot_groups = {}
    for plot_file in regular_plots:
        basename = os.path.basename(plot_file)
        mjcf_name = basename.split('_')[0]
        if mjcf_name not in plot_groups:
            plot_groups[mjcf_name] = []
        plot_groups[mjcf_name].append(plot_file)

    # Sort plot groups by mjcf name and plot file name
    plot_groups = sorted(plot_groups.items(), key=lambda x: (x[0], x[1][0]))

    # Group aspect ratio plots by MJCF file
    aspect_ratio_groups = {}
    for aspect_ratio, plots in aspect_ratio_plots.items():
        aspect_ratio_groups[aspect_ratio] = {}
        for plot_file in plots:
            basename = os.path.basename(plot_file)
            mjcf_name = basename.split('_')[0]
            if mjcf_name not in aspect_ratio_groups[aspect_ratio]:
                aspect_ratio_groups[aspect_ratio][mjcf_name] = []
            aspect_ratio_groups[aspect_ratio][mjcf_name].append(plot_file)
        
        # Sort each aspect ratio's plot groups
        aspect_ratio_groups[aspect_ratio] = sorted(aspect_ratio_groups[aspect_ratio].items(), key=lambda x: (x[0], x[1][0]))

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

    # Add regular plots section
    html_content += "<div class='section'>\n"
    html_content += "<h2>Performance Plots</h2>\n"
    for mjcf_name, plots in plot_groups:
        html_content += f"<div class='plot-container'>\n"
        for plot in plots:
            html_content += f"<h3>{html.escape(mjcf_name)} - {'Rasterizer' if 'rasterizer' in os.path.basename(plot) else 'Raytracer'}</h3>\n"
            html_content += f"<img src='{html.escape(os.path.basename(plot))}' alt='{html.escape(os.path.basename(plot))}'/><br/>\n"
        html_content += "</div>\n"
    html_content += "</div>\n"

    # Add aspect ratio difference plots sections
    for aspect_ratio, mjcf_groups in aspect_ratio_groups.items():
        if mjcf_groups:
            html_content += "<div class='section'>\n"
            html_content += f"<h2>Performance Difference Plots (Rasterizer / Raytracer) ({aspect_ratio} Resolutions Only)</h2>\n"
            for mjcf_name, plots in mjcf_groups:
                html_content += f"<div class='plot-container'>\n"
                html_content += f"<h3>{html.escape(mjcf_name)}</h3>\n"
                html_content += f"<img src='{html.escape(os.path.basename(plots[0]))}' alt='{html.escape(os.path.basename(plots[0]))}'/><br/>\n"
                html_content += "</div>\n"
            html_content += "</div>\n"

    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(f"{plots_dir}/index.html", 'w') as f:
        f.write(html_content)

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
        generate_ratio_plots(df, plots_dir, width, height, aspect_ratio=aspect_ratio)

    # Generate an html page to display all the plots
    generatePlotHtml(plots_dir)

def generate_individual_plots(df, plots_dir, width, height):
    # Get unique combinations of mjcf and rasterizer
    for mjcf in df['mjcf'].unique():
        for rasterizer in df[df['mjcf'] == mjcf]['rasterizer'].unique():
            # Filter data for this mjcf and rasterizer
            data = df[(df['mjcf'] == mjcf) & (df['rasterizer'] == rasterizer)]

            # continue if there is no data
            if len(data) == 0:
                print(f"No data found for {mjcf} and {rasterizer}")
                continue
            
            # Create new figure
            plt.figure(figsize=(width, height))
            
            # Plot each resolution with different color
            # Sort by resX and resY before plotting
            for res in sorted(data.groupby(['resX', 'resY']), key=lambda x: (x[0][0], x[0][1])):
                resolution = res[0]
                res_data = res[1]
                plt.plot(res_data['n_envs'], res_data['fps'], 
                        marker='o', label=f'{resolution[0]}x{resolution[1]}')
                
                # Add x/y value annotations near each point
                for x, y in zip(res_data['n_envs'], res_data['fps']):
                    plt.annotate(f'({x:.0f}, {y:.1f})', 
                               (x, y),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=8)
            
            # Customize plot
            plt.title(f'Performance for {os.path.basename(mjcf)}\n{"Rasterizer" if rasterizer else "Raytracer"}')
            plt.xlabel('Batch Size')
            plt.ylabel('FPS')
            plt.grid(True)
            plt.legend(title='Resolution')
            plt.xscale('log')
            plt.yscale('log')
            
            # Force both axes to use ScalarFormatter
            # Get current axes
            ax = plt.gca()
            
            # Set major formatter for both axes
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

            # Disable scientific notation
            ax.ticklabel_format(axis='both', style='plain')
            
            # Add more tick marks
            ax.xaxis.set_major_locator(plt.LogLocator(base=2))
            ax.yaxis.set_major_locator(plt.LogLocator(base=10))
            
            # Add minor tick marks
            ax.xaxis.set_minor_locator(plt.LogLocator(base=2, subs=(.5,)))
            ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=(.5,)))
            
            # Save plot
            rasterizer_str = "rasterizer" if rasterizer else "raytracer"
            filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_{rasterizer_str}.png"
            plt.savefig(filename)
            plt.close()

def generate_difference_plots(df, plots_dir, width, height, aspect_ratio=None):
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

    # Generate plots showing fps difference between rasterizer and raytracer
    for mjcf in df['mjcf'].unique():
        mjcf_data = df[df['mjcf'] == mjcf]
        
        # Get resolutions available for both rasterizer and raytracer
        rasterizer_res = set(zip(mjcf_data[mjcf_data['rasterizer']]['resX'], 
                                mjcf_data[mjcf_data['rasterizer']]['resY']))
        raytracer_res = set(zip(mjcf_data[~mjcf_data['rasterizer']]['resX'],
                               mjcf_data[~mjcf_data['rasterizer']]['resY']))
        common_res = rasterizer_res.intersection(raytracer_res)

        # continue if there is no data
        if len(common_res) == 0:
            print(f"No data found for {mjcf}")
            continue

        plt.figure(figsize=(width, height))
        
        # Plot difference for each common resolution
        max_abs_diff = 0  # Track maximum absolute difference for y-axis limits
        min_abs_diff = 1e10
        for resX, resY in sorted(common_res, key=lambda x: x[0] * x[1]):
            rast_data = mjcf_data[(mjcf_data['rasterizer']) & 
                                 (mjcf_data['resX'] == resX) & 
                                 (mjcf_data['resY'] == resY)]
            ray_data = mjcf_data[(~mjcf_data['rasterizer']) &
                                (mjcf_data['resX'] == resX) &
                                (mjcf_data['resY'] == resY)]
            
            # Match batch sizes and calculate difference
            common_batch = set(rast_data['n_envs']).intersection(set(ray_data['n_envs']))
            batch_sizes = sorted(list(common_batch))
            
            rast_fps = rast_data[rast_data['n_envs'].isin(batch_sizes)]['fps'].values
            ray_fps = ray_data[ray_data['n_envs'].isin(batch_sizes)]['fps'].values
            diff_fps = rast_fps - ray_fps
            
            # Update max absolute difference
            max_abs_diff = max(max_abs_diff, diff_fps.max())
            min_abs_diff = min(min_abs_diff, diff_fps.min())

            plt.plot(batch_sizes, diff_fps, marker='o', label=f'{resX}x{resY}')
            
            # Add annotations
            for x, y in zip(batch_sizes, diff_fps):
                plt.annotate(f'({x:.0f}, {y:.1f})',
                           (x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8)
        
        # Set title based on aspect ratio
        subtitle1 = "FPS Difference (Rasterizer - Raytracer)"
        if aspect_ratio:
            subtitle2 = f"({aspect_ratio} Resolutions Only)"
        else:
            subtitle2 = ""
        plt.title(f'{subtitle1}\n{os.path.basename(mjcf)} {subtitle2}')
        plt.xlabel('Batch Size')
        plt.ylabel(subtitle1)
        plt.grid(True)
        plt.legend(title='Resolution')
        plt.xscale('log')
        plt.yscale('symlog', linthresh=1.0)  # Use symlog with linear threshold of 1.0
        
        # Format axes
        ax = plt.gca()
            
        # Set major formatter for both axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # Disable scientific notation
        ax.ticklabel_format(axis='both', style='plain')
        
        # Add more tick marks
        ax.xaxis.set_major_locator(plt.LogLocator(base=2))
        ax.yaxis.set_major_locator(plt.LogLocator(base=10))
        
        # Add minor tick marks
        ax.xaxis.set_minor_locator(plt.LogLocator(base=2, subs=(.5,)))
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=(.5,)))
            
        # Set y-axis limits symmetrically
        ax.set_ylim(0, max_abs_diff*1.1)  # Add 10% padding
        
        # Add horizontal line at y=0 to show crossover point
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        # Save plot with aspect ratio in filename if specified
        if aspect_ratio:
            filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_difference_{aspect_ratio.replace(':', 'x')}.png"
        else:
            filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_difference.png"
        plt.savefig(filename)
        plt.close()

def generate_ratio_plots(df, plots_dir, width, height, aspect_ratio=None):
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

    # Generate plots showing fps difference between rasterizer and raytracer
    for mjcf in df['mjcf'].unique():
        mjcf_data = df[df['mjcf'] == mjcf]
        
        # Get resolutions available for both rasterizer and raytracer
        rasterizer_res = set(zip(mjcf_data[mjcf_data['rasterizer']]['resX'], 
                                mjcf_data[mjcf_data['rasterizer']]['resY']))
        raytracer_res = set(zip(mjcf_data[~mjcf_data['rasterizer']]['resX'],
                               mjcf_data[~mjcf_data['rasterizer']]['resY']))
        common_res = rasterizer_res.intersection(raytracer_res)
        
        # continue if there is no data
        if len(common_res) == 0:
            print(f"No data found for {mjcf}")
            continue

        plt.figure(figsize=(width, height))
        
        # Plot difference for each common resolution
        max_abs_diff = 0  # Track maximum absolute difference for y-axis limits
        min_abs_diff = 1e10
        for resX, resY in sorted(common_res, key=lambda x: x[0] * x[1]):
            rast_data = mjcf_data[(mjcf_data['rasterizer']) & 
                                 (mjcf_data['resX'] == resX) & 
                                 (mjcf_data['resY'] == resY)]
            ray_data = mjcf_data[(~mjcf_data['rasterizer']) &
                                (mjcf_data['resX'] == resX) &
                                (mjcf_data['resY'] == resY)]
            
            # Match batch sizes and calculate difference
            common_batch = set(rast_data['n_envs']).intersection(set(ray_data['n_envs']))
            batch_sizes = sorted(list(common_batch))
            
            rast_fps = rast_data[rast_data['n_envs'].isin(batch_sizes)]['fps'].values
            ray_fps = ray_data[ray_data['n_envs'].isin(batch_sizes)]['fps'].values
            diff_fps = rast_fps / ray_fps
            
            # Update max absolute difference
            max_abs_diff = max(max_abs_diff, diff_fps.max())
            min_abs_diff = min(min_abs_diff, diff_fps.min())

            plt.plot(batch_sizes, diff_fps, marker='o', label=f'{resX}x{resY}')
            
            # Add annotations
            for x, y in zip(batch_sizes, diff_fps):
                plt.annotate(f'({x:.0f}, {y:.1f})',
                           (x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8)
        
        # Set title based on aspect ratio
        subtitle1 = "FPS Difference (Rasterizer / Raytracer)"
        if aspect_ratio:
            subtitle2 = f"({aspect_ratio} Resolutions Only)"
        else:
            subtitle2 = ""
        plt.title(f'{subtitle1}\n{os.path.basename(mjcf)} {subtitle2}')
        plt.xlabel('Batch Size')
        plt.ylabel(subtitle1)
        plt.grid(True)
        plt.legend(title='Resolution')
        plt.xscale('log')
        plt.yscale('log')  # Use log scale with smaller base for better resolution near 1.0
        
        # Format axes
        ax = plt.gca()
            
        # Set major formatter for both axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # Disable scientific notation
        ax.ticklabel_format(axis='both', style='plain')
        
        # Add more tick marks
        ax.xaxis.set_major_locator(plt.LogLocator(base=2))
        ax.yaxis.set_major_locator(plt.LogLocator(base=10))
        
        # Add minor tick marks
        ax.xaxis.set_minor_locator(plt.LogLocator(base=2, subs=(.5,)))
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=(.5,)))

        # Set y-axis limits to ensure all points are visible
        #ax.set_ylim(min_abs_diff * 0.9, max_abs_diff * 1.1)  # Add 10% padding
        ax.autoscale(enable=True, axis='y')
        
        # Add horizontal line at y=1 to show crossover point
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        # Save plot with aspect ratio in filename if specified
        if aspect_ratio:
            filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_difference_{aspect_ratio.replace(':', 'x')}.png"
        else:
            filename = f"{plots_dir}/{os.path.splitext(os.path.basename(mjcf))[0]}_difference.png"
        plt.savefig(filename)
        plt.close()

def main():
    import sys
    import os
    print("Script arguments:", sys.argv)  # Debug print
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file_path", type=str, default="logs/benchmark/batch_benchmark_20250524_180908.csv",
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
