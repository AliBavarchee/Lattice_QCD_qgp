import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
# Prepare the data
fermion_df['real_psi'] = fermion_df['psi'].apply(lambda x: x.real)
fermion_df['imag_psi'] = fermion_df['psi'].apply(lambda x: x.imag)

# 3D Scatter Plot of Fermion Fields
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(fermion_df['x'], fermion_df['y'], fermion_df['z'], c=fermion_df['real_psi'], cmap='viridis', marker='o', s=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Visualization of Fermion Fields (Real part of psi)')
plt.colorbar(sc, label='Real Part of psi')
plt.show()


# Visualization of Gauge Fields
# Select a specific time slice (e.g., t = 1) to visualize
t_slice = 1
gauge_df_t1 = gauge_df[gauge_df['t'] == t_slice]

# Prepare pivot table for heatmap
heatmap_data = gauge_df_t1.pivot_table(index='x', columns='y', values='real', aggfunc='mean').fillna(0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f")
plt.title(f'Heatmap of Gauge Field Real Part at t={t_slice}')
plt.xlabel('Y Coordinate')
plt.ylabel('X Coordinate')
plt.show()


# 3D Visualization of Gauge Field
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot real part of the gauge field
ax.scatter(gauge_df['x'], gauge_df['y'], gauge_df['z'], c=gauge_df['real'], cmap='plasma', marker='^', s=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Visualization of Gauge Fields (Real Part)')
plt.colorbar(sc, label='Real Part of U')
plt.show()
