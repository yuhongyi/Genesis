#!/usr/bin/env python3
"""
Script to convert URDF and XML files to JSON format using Genesis Apollo renderer.
Outputs .jurdf files for URDF inputs and .jxml files for XML inputs.
"""

import argparse
import os
import sys
from pathlib import Path

import genesis as gs
import numpy as np
from genesis.utils.geom import trans_to_T
from genesis.utils.scene_exporter import SceneDescriptionExporter
from genesis.utils.scene_exporter import (
    _make_tensor,
    _build_mesh_transform_idx,
    _pos_to_y_up,
    _quat_to_y_up,
    _camera_quat_to_y_up,
)

def convert_file_to_json(input_file: str, output_dir: str = None):
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine file type and output extension
    file_ext = input_path.suffix.lower()
    if file_ext == '.urdf':
        output_ext = '.jurdf'
        morph_class = gs.morphs.URDF
    elif file_ext == '.xml':
        output_ext = '.jxml'
        morph_class = gs.morphs.MJCF
    else:
        raise ValueError(f"Unsupported file type: {file_ext} ")
    
    # Determine output path
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file = output_dir_path / (input_path.stem + output_ext)
    else:
        output_file = input_path.parent / (input_path.stem + output_ext)
    
    print(f"Converting {input_file} to {output_file}")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene with Apollo renderer for JSON export
    scene = gs.Scene(
        renderer=gs.options.renderers.Rasterizer(),
    )

    # Add the entity from the input file
    entity = scene.add_entity(
        morph_class(
            file=str(input_file),
            pos=(0.0, 0.0, 0.0),
        ),
    )
    
    # Build the scene (this triggers the JSON export)
    scene.build(n_envs=0)

    # Export the scene description and load it into the renderer
    scene_exporter = SceneDescriptionExporter(scene)
    scene_exporter.export_to_file(output_file)
    
    print(f"Successfully exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert URDF and XML files to JSON format using Genesis Apollo renderer"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input URDF or XML file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="Output directory (defaults to same directory as input file)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_file_to_json(args.input_file, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()