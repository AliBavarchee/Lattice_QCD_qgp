import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

# Parse the fermion data
def parse_fermion_field_file(filename):
    data = []
    current_coords = {}

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('x='):
            # Handle coordinate line
            try:
                match = re.match(r'x=\s*(\d+)\s*y=\s*(\d+)\s*z=\s*(\d+)\s*t=\s*(\d+)', line)
                if match:
                    x, y, z, t = map(int, match.groups())
                    current_coords = {'x': x, 'y': y, 'z': z, 't': t}
                else:
                    print(f"Skipping line due to format issue: {line}")
            except ValueError as e:
                print(f"Error parsing coordinates line: {line} - {e}")
        
        elif line.startswith('psi('):
            # Handle psi line
            try:
                parts = re.split(r'\s*=\s*', line)
                if len(parts) == 2:
                    psi_values = parts[1].split()
                    if len(psi_values) == 2:
                        index = int(re.findall(r'\d+', parts[0])[0])
                        real_part = float(psi_values[0])
                        imag_part = float(psi_values[1])
                        data.append({
                            'x': current_coords.get('x', None),
                            'y': current_coords.get('y', None),
                            'z': current_coords.get('z', None),
                            't': current_coords.get('t', None),
                            'i': index,
                            'psi': complex(real_part, imag_part)
                        })
                    else:
                        print(f"Skipping psi line due to format issue: {line}")
                else:
                    print(f"Skipping psi line due to format issue: {line}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing psi line: {line} - {e}")

    df = pd.DataFrame(data)
    df.fillna(value={'x': -1, 'y': -1, 'z': -1, 't': -1}, inplace=True)
    return df

# Parse the gauge data
def parse_gauge_field_file(filename):
    gauge_data = []
    current_coordinates = None
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('x='):
                try:
                    coords = re.findall(r'x=\s*(\d+)\s*y=\s*(\d+)\s*z=\s*(\d+)\s*t=\s*(\d+)', line)[0]
                    current_coordinates = tuple(map(int, coords))
                except IndexError:
                    print(f"Error parsing coordinates line: {line}")
                    current_coordinates = None
                continue
            if line.startswith('U('):
                if current_coordinates is None:
                    continue
                try:
                    matrix_line = re.findall(r'U\(\s*(\d+),\s*(\d+)\)\s*=\s*(.*)', line)
                    if matrix_line:
                        i, j, values = matrix_line[0]
                        i, j = int(i), int(j)
                        values = values.split()
                        if len(values) == 2:
                            real, imag = values
                            real = float(real) if real != '**********' else np.nan
                            imag = float(imag) if imag != '**********' else np.nan
                            gauge_data.append((*current_coordinates, i, j, real, imag))
                except Exception as e:
                    print(f"Error parsing gauge line: {line} - {e}")
                continue
    
    df = pd.DataFrame(gauge_data, columns=['x', 'y', 'z', 't', 'i', 'j', 'real', 'imag'])
    return df

# Load data
fermion_df = parse_fermion_field_file('fermion_fields.txt')
gauge_df = parse_gauge_field_file('gauge_fields.txt')

# Prepare data for Plotly
fermion_df['real_psi'] = fermion_df['psi'].apply(lambda x: x.real)
fermion_df['imag_psi'] = fermion_df['psi'].apply(lambda x: x.imag)

# 3D Visualization of Fermion Field
fig_fermion = go.Figure(data=go.Scatter3d(
    x=fermion_df['x'],
    y=fermion_df['y'],
    z=fermion_df['z'],
    mode='markers',
    marker=dict(
        size=5,
        color=fermion_df['real_psi'],  # Color by real part of psi
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Real Part of psi')
    )
))

fig_fermion.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z'
), title='3D Visualization of Fermion Fields (Real part of psi)')

fig_fermion.show()

# 3D Visualization of Gauge Field
fig_gauge = go.Figure(data=go.Scatter3d(
    x=gauge_df['x'],
    y=gauge_df['y'],
    z=gauge_df['z'],
    mode='markers',
    marker=dict(
        size=5,
        color=gauge_df['real'],  # Color by real part of gauge field
        colorscale='Plasma',
        opacity=0.8,
        colorbar=dict(title='Real Part of U')
    )
))

fig_gauge.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z'
), title='3D Visualization of Gauge Fields (Real Part)')

fig_gauge.show()
