# Lennard-Jones Monte Carlo Practice

This folder contains a small **practice project** for learning the basics of **Monte Carlo simulation** in statistical physics.

The current code implements a simple **Metropolis Monte Carlo simulation** for a **Lennard-Jones particle system** in a periodic simulation box.  
This project is mainly intended for **study, practice, and code organization on GitHub**, rather than for production-level scientific simulation.

## What this project does

This practice code includes:

- initialization of particles on a simple cubic lattice
- Lennard-Jones potential energy calculation
- periodic boundary conditions with the minimum image convention
- single-particle Metropolis Monte Carlo moves
- energy sampling during the simulation
- radial distribution function `g(r)` from the final configuration
- basic visualization of simulation results

## Files

```text
.
├── lj_monte_carlo_practice.py
└── README.md
