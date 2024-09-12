import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# parse fermion data
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
                # Extract coordinates from the line
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
                # Extract the index and values
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

    # Convert list to DataFrame
    df = pd.DataFrame(data)

    # Debug information
    print(f"Parsed {len(data)} rows")

    # Fill NaNs for coordinates if necessary (optional)
    df.fillna(value={'x': -1, 'y': -1, 'z': -1, 't': -1}, inplace=True)
    
    return df

# Parse guage data
def parse_gauge_field_file(filename):
    gauge_data = []
    current_coordinates = None
    current_mu = None
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('x='):
                # Coordinates line
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
                    # Parse matrix values
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
            if line.startswith('mu='):
                # Skip the mu line or handle it if needed
                continue
    
    df = pd.DataFrame(gauge_data, columns=['x', 'y', 'z', 't', 'i', 'j', 'real', 'imag'])
    return df




# fermion data frame
filename = 'fermion_fields.txt'
fermion_df = parse_fermion_field_file(filename)

# gause data frame
gauge_df = parse_gauge_field_file('gauge_fields.txt')
# print out dfs
print("=-=-=-=-=fermion_df-=-=-=-=-=")
print(fermion_df.head())
print("=-=-=-=-=gauge_df-=-=-=-=-=")
print(gauge_df.head())



# Visulalization

#Visualization of Fermion Fields

# Prepare the data for fermion field visualization
fermion_df['real_psi'] = fermion_df['psi'].apply(lambda x: x.real)

# Select a specific time slice (e.g., t = 1) to visualize fermion fields
t_slice = 1
fermion_df_t1 = fermion_df[fermion_df['t'] == t_slice]

# Create a pivot table for heatmap of fermion fields
fermion_heatmap_data = fermion_df_t1.pivot_table(index='x', columns='y', values='real_psi', aggfunc='mean').fillna(0)

# Plot the heatmap for Fermion Fields
fig_fermion = px.imshow(fermion_heatmap_data, color_continuous_scale='viridis',
                        labels={'color': 'Real Part of psi'},
                        title=f'Heatmap of Fermion Fields (Real Part of psi) at t={t_slice}')
fig_fermion.update_layout(xaxis_title='Y Coordinate', yaxis_title='X Coordinate')
fig_fermion.show()

# Prepare the data for gauge field visualization
gauge_df['real_U'] = gauge_df['real']








# Filter data for a specific time slice if needed (e.g., t = 1)
t_slice = 1
gauge_df_t1 = gauge_df[gauge_df['t'] == t_slice]

# Create a 3D scatter plot using Plotly
fig = go.Figure(data=go.Scatter3d(
    x=gauge_df_t1['x'],
    y=gauge_df_t1['y'],
    z=gauge_df_t1['z'],
    mode='markers',
    marker=dict(
        size=5,
        color=gauge_df_t1['real'],  # Set color to the real part of the gauge field
        colorscale='Plasma',
        opacity=0.8,
        colorbar=dict(title='Real Part of U')
    )
))

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title=f'3D Visualization of Gauge Fields (Real Part) at t={t_slice}',
)

# Show plot
fig.show()

