# Naïve Lattice Full QCD Simulation

This repository contains Fortran code for simulating Quantum Chromodynamics (QCD) on a discrete lattice. The code provides tools to initialize and evolve gauge fields, calculate observables, and analyze the behavior of QCD in a non-perturbative regime.

## Fortran Codes

- **`lattice_qcd.f90`**: Defines the core module for the lattice QCD simulation. It contains parameters, data structures, and subroutines necessary for initializing and manipulating the lattice fields.
- **`main.f90`**: The main program file that drives the simulation. It sets up the initial conditions, performs the main simulation loop, and handles data output.


### `lattice_qcd.f90`

This file defines a module named `lattice_qcd` that includes:

- **Parameters and Lattice Setup**:
  - `dim`: Number of space-time dimensions (typically 4).
  - `nx, ny, nz, nt`: Lattice sizes in the x, y, z, and t dimensions.
  - `Nc`: Number of colors in SU(3) gauge theory (set to 3 for QCD).
  - `beta`: Gauge coupling parameter, a key input for simulations.

- **Data Structures**:
  - `U`: A multidimensional complex array representing the gauge fields on the lattice. These fields correspond to the SU(3) matrices associated with each link.
  - `psi`: A complex array representing the fermion (quark) fields on the lattice.

- **Subroutines**:
  - `initialize_gauge_fields()`: Initializes the gauge fields (`U`) to identity matrices as the starting point for the simulation.
  - Additional subroutines for updating gauge fields, applying boundary conditions, and calculating observables.

### `main.f90`

This is the main entry point for running the lattice QCD simulation. It includes:

- **Initialization**:
  - Calls to subroutines in `lattice_qcd.f90` to set up the initial state of the lattice.
  
- **Simulation Loop**:
  - The main loop that iteratively updates the lattice fields and computes relevant observables.

- **Data Output**:
  - Code for saving or displaying the results of the simulation, such as gauge field configurations and computed quantities like Wilson loops.

## EXECUTION

### Compiling the Code

To compile the code, run the following command in your terminal:

```bash
gfortran -o lattice_qcd_simulation main.f90 lattice_qcd.f90
```

This will generate an executable named `lattice_qcd_simulation`.

### Running the Simulation

To run the simulation, execute the following command:

```bash
./lattice_qcd_simulation
```

The program will initialize the lattice, run the simulation loop, and output results to the terminal or specified files.

## Outputs

`gauge_fields.txt` as the gauge data and `fermion_fields.txt` as the fermion data are saved on the same directory after performing the simulation.

