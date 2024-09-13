# Naïve Lattice Full QCD Simulation

The Fortran90 code is provided for a simulation for Lattice QCD, using the Hamiltonian Monte Carlo (HMC) algorithm.

### Key Components:

1. **Parameters and Variables:**
   - The code defines parameters such as lattice dimensions (`nx`, `ny`, `nz`, `nt`), the number of colors (`Nc`), gauge coupling (`beta`), step size for integration (`epsilon`), and the number of leapfrog steps (`Nsteps`).
   - `U` represents gauge fields, `psi` represents fermion fields, and `P` represents conjugate momenta.

2. **Initialization:**
   - **Gauge Fields:** Initialized to identity matrices.
   - **Fermion Fields:** Initialized with random complex values.
   - **Momenta:** Initialized with random complex values.

3. **Hamiltonian Calculation:**
   - **Kinetic Energy:** Computed from conjugate momenta.
   - **Potential Energy:** Computed using Wilson gauge action.

4. **Leapfrog Integration:**
   - **Momentum Update:** A half-step update for momenta.
   - **Gauge Field Update:** A full-step update for gauge fields.
   - **Momentum Update:** Another half-step update for momenta.

5. **Hamiltonian Monte Carlo (HMC):**
   - **Acceptance Step:** Uses the Metropolis criterion to decide whether to accept or reject the new configuration based on the change in Hamiltonian.

### Some Improvements and Corrections should be addressed:

1. **Random Number Generation for Acceptance Probability:**
   The code for Metropolis acceptance criteria seems incorrect! :white_check_mark:

2. **Restoring Old Configuration:**
   The code is ment to restore the old configuration if the new configuration is rejected but does not implement it. It has to save the old configuration before the leapfrog integration and restore it if needed! :x:

3. **Hamiltonian Calculation - Trace Function:**
   The `trace` function is declared as external but not implemented. It should be provided the implementation or use a standard Fortran trace function! ===> Mission completed by `matmul` combined with `trace`. :white_check_mark:

4. **Potential Energy Calculation:**
   We need to ensure the calculation of potential energy aligns with the lattice gauge action used in your specific setup. Double-check the formula for the Wilson gauge action to make sure it’s implemented correctly. :white_check_mark:

5. **Comments and Documentation:**

6. **Error Handling and Debugging:**
   It is nice to add error handling or debug print statements to track issues during execution. :x:


-----------
Reference:
[Introduction to Lattice QCD by Hideo Matsufuru](https://research.kek.jp/people/matufuru/Research/Docs/Lattice/Introduction/note_lattice.pdf)

