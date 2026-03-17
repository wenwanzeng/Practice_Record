"""
Practice Project: Lennard-Jones Monte Carlo Simulation

This code is written as a personal practice project to learn:
1. The Metropolis Monte Carlo algorithm
2. Lennard-Jones potential energy calculation
3. Periodic boundary conditions
4. Basic structural analysis such as radial distribution function g(r)

This is not intended to be a production-grade simulation package.
It is mainly for learning, testing, and GitHub practice.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Use a relative output directory so the project is portable on GitHub
OUTPUT_DIR = r"C:\Users\15877\Desktop\UCAS\20260317kMC"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class LennardJonesMC:
    """Monte Carlo simulation for a Lennard-Jones particle system."""

    def __init__(self, n_atoms, box_length, temperature,
                 epsilon=1.0, sigma=1.0, cutoff=3.0):
        """
        Initialize the Monte Carlo simulation.

        Parameters
        ----------
        n_atoms : int
            Number of atoms.
        box_length : float
            Simulation box length.
        temperature : float
            Reduced temperature.
        epsilon : float
            Lennard-Jones epsilon parameter.
        sigma : float
            Lennard-Jones sigma parameter.
        cutoff : float
            Interaction cutoff radius.
        """
        self.N = n_atoms
        self.L = box_length
        self.T = temperature
        self.epsilon = epsilon
        self.sigma = sigma
        self.r_cut = cutoff

        # In reduced units, beta = 1 / T
        self.beta = 1.0 / temperature

        # Initialize atomic positions on a simple cubic lattice
        self.positions = self._init_lattice()

        # Compute the initial total energy of the system
        self.energy = self._compute_total_energy()

        # Statistics for Monte Carlo sampling
        self.n_attempts = 0
        self.n_accepted = 0
        self.energy_history = []

    def _init_lattice(self):
        """
        Initialize atom positions on a simple cubic lattice.

        This gives a clean, non-overlapping starting configuration.
        """
        n = int(np.ceil(self.N ** (1 / 3)))
        spacing = self.L / n

        positions = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(positions) < self.N:
                        pos = np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing
                        positions.append(pos)

        return np.array(positions)

    def _lj_potential(self, r):
        """
        Compute the Lennard-Jones potential with a shifted cutoff.

        The shift makes U(r_cut) = 0 so that the potential is continuous
        at the cutoff distance.
        """
        if r > self.r_cut:
            return 0.0

        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 ** 2

        # Energy shift at the cutoff
        u_cut = 4 * self.epsilon * (
            (self.sigma / self.r_cut) ** 12 -
            (self.sigma / self.r_cut) ** 6
        )

        return 4 * self.epsilon * (sr12 - sr6) - u_cut

    def _compute_pair_energy(self, i, positions=None):
        """
        Compute the interaction energy contribution associated with atom i.

        This is used in local Monte Carlo moves so that we do not need
        to recompute the full system energy every time.
        """
        if positions is None:
            positions = self.positions

        energy = 0.0
        for j in range(self.N):
            if i != j:
                # Minimum image convention for periodic boundary conditions
                r_ij = positions[i] - positions[j]
                r_ij -= self.L * np.round(r_ij / self.L)
                r = np.linalg.norm(r_ij)

                if r < self.r_cut:
                    energy += self._lj_potential(r)

        # Divide by 2 to remain consistent with the total-energy bookkeeping
        return energy / 2

    def _compute_total_energy(self):
        """
        Compute the total potential energy of the system.

        Each pair is counted only once by looping over i < j.
        """
        energy = 0.0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                r_ij = self.positions[i] - self.positions[j]
                r_ij -= self.L * np.round(r_ij / self.L)
                r = np.linalg.norm(r_ij)

                if r < self.r_cut:
                    energy += self._lj_potential(r)

        return energy

    def mc_move(self, max_displacement=0.1):
        """
        Perform one Metropolis Monte Carlo displacement move.

        Parameters
        ----------
        max_displacement : float
            Maximum displacement amplitude in each Cartesian direction.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        # Randomly choose one atom
        i = np.random.randint(0, self.N)

        # Save its old position in case the move is rejected
        old_pos = self.positions[i].copy()

        # Compute the interaction energy before the trial move
        old_energy = self._compute_pair_energy(i)

        # Propose a random displacement in 3D
        displacement = (2 * np.random.random(3) - 1) * max_displacement
        self.positions[i] += displacement

        # Apply periodic boundary conditions to keep the atom inside the box
        self.positions[i] -= self.L * np.floor(self.positions[i] / self.L)

        # Compute the interaction energy after the trial move
        new_energy = self._compute_pair_energy(i)

        # Energy difference for the trial move
        delta_e = new_energy - old_energy

        # Count this trial
        self.n_attempts += 1

        # Metropolis acceptance rule:
        # 1) always accept if energy decreases
        # 2) otherwise accept with probability exp(-beta * delta_e)
        if delta_e < 0 or np.random.random() < np.exp(-self.beta * delta_e):
            self.energy += delta_e
            self.n_accepted += 1
            return True
        else:
            # Reject: restore the old position
            self.positions[i] = old_pos
            return False

    def volume_move(self, max_dV=0.1):
        """
        Attempt a volume move for a simplified NPT-like Monte Carlo scheme.

        Note:
        This is included only as a learning example and is simplified.
        The pressure term is omitted here (effectively assuming P = 0).
        """
        old_L = self.L
        old_V = self.L ** 3
        old_energy = self.energy

        # Trial move in log-volume space
        ln_V_new = np.log(old_V) + (2 * np.random.random() - 1) * max_dV
        V_new = np.exp(ln_V_new)
        L_new = V_new ** (1 / 3)

        # Scale positions according to the new box size
        scale = L_new / old_L
        self.positions *= scale
        self.L = L_new

        # Recompute total energy after scaling
        new_energy = self._compute_total_energy()

        delta_e = new_energy - old_energy

        # Simplified acceptance weight for a volume move
        weight = self.N * np.log(V_new / old_V) - self.beta * delta_e

        if weight > 0 or np.random.random() < np.exp(weight):
            self.energy = new_energy
            self.n_accepted += 1
            return True
        else:
            # Reject: restore old box and positions
            self.positions /= scale
            self.L = old_L
            self.energy = old_energy
            return False

    def run(self, n_steps, n_equil=1000, sample_interval=10):
        """
        Run the Monte Carlo simulation.

        Parameters
        ----------
        n_steps : int
            Number of production MC steps.
        n_equil : int
            Number of equilibration steps before production.
        sample_interval : int
            Interval for storing energy samples.
        """
        print(f"Starting MC simulation: {n_steps} production steps")
        print(f"Temperature: {self.T}, Number of atoms: {self.N}")

        # Equilibration stage
        print("Equilibrating...")
        for _ in range(n_equil):
            self.mc_move()

        # Reset statistics before the production run
        self.n_attempts = 0
        self.n_accepted = 0
        self.energy_history = []

        # Production stage
        print("Running production...")
        for step in range(n_steps):
            self.mc_move()

            # Store energy every few steps
            if step % sample_interval == 0:
                self.energy_history.append(self.energy)

            # Print progress occasionally
            if (step + 1) % 10000 == 0:
                acceptance = self.n_accepted / self.n_attempts
                print(f"  Step: {step + 1}, Energy: {self.energy:.2f}, Acceptance: {acceptance:.3f}")

        print("\nSimulation finished.")
        print(f"Average energy: {np.mean(self.energy_history):.2f}")
        print(f"Final acceptance rate: {self.n_accepted / self.n_attempts:.3f}")

    def compute_radial_distribution(self, n_bins=100):
        """
        Compute the radial distribution function g(r) from the final configuration.

        Parameters
        ----------
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        tuple
            (r_values, g_r)
        """
        r_max = self.L / 2
        dr = r_max / n_bins
        hist = np.zeros(n_bins)

        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                r_ij = self.positions[i] - self.positions[j]
                r_ij -= self.L * np.round(r_ij / self.L)
                r = np.linalg.norm(r_ij)

                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        # Count both i->j and j->i contributions
                        hist[bin_idx] += 2

        # Bin centers
        r_bins = np.linspace(0, r_max, n_bins + 1)[:-1] + dr / 2

        # Spherical shell volume
        shell_volume = 4 * np.pi * r_bins ** 2 * dr

        # Number density
        rho = self.N / (self.L ** 3)

        # Normalize the histogram
        g_r = hist / (self.N * rho * shell_volume)

        return r_bins, g_r

    def visualize(self):
        """
        Visualize simulation results and save the figure to the output folder.
        """
        fig = plt.figure(figsize=(15, 10))

        # 1. Energy evolution
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(self.energy_history, 'b-', alpha=0.7)
        ax1.axhline(
            np.mean(self.energy_history),
            color='r',
            linestyle='--',
            label=f'Mean: {np.mean(self.energy_history):.1f}'
        )
        ax1.set_xlabel('Sample Step')
        ax1.set_ylabel('Potential Energy')
        ax1.set_title('Energy Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Energy histogram
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(self.energy_history, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(self.energy_history), color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Potential Energy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Energy Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. Radial distribution function
        r, g_r = self.compute_radial_distribution()
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(r, g_r, 'b-', linewidth=2)
        ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('r (reduced units)')
        ax3.set_ylabel('g(r)')
        ax3.set_title('Radial Distribution Function')
        ax3.set_xlim(0, self.L / 2)
        ax3.grid(True, alpha=0.3)

        # 4. Final 3D configuration
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
            c='blue',
            s=50,
            alpha=0.6
        )
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('Final Configuration')
        ax4.set_xlim(0, self.L)
        ax4.set_ylim(0, self.L)
        ax4.set_zlim(0, self.L)

        # 5. Acceptance rate
        ax5 = fig.add_subplot(2, 3, 5)
        acceptance_rate = self.n_accepted / self.n_attempts
        ax5.bar(['MC Move'], [acceptance_rate], color='purple', alpha=0.7)
        ax5.axhline(0.5, color='red', linestyle='--', label='Target: 50%')
        ax5.set_ylabel('Acceptance Rate')
        ax5.set_title('Monte Carlo Acceptance Rate')
        ax5.set_ylim(0, 1)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Text summary panel
        ax6 = fig.add_subplot(2, 3, 6)
        energy_std = np.std(self.energy_history)

        ax6.text(0.5, 0.7, f'Temperature: {self.T:.2f}', ha='center',
                 transform=ax6.transAxes, fontsize=12)
        ax6.text(0.5, 0.5, f'Mean Energy: {np.mean(self.energy_history):.2f}',
                 ha='center', transform=ax6.transAxes, fontsize=12)
        ax6.text(0.5, 0.3, f'Energy Std: {energy_std:.2f}', ha='center',
                 transform=ax6.transAxes, fontsize=12)
        ax6.text(0.5, 0.1, f'Acceptance: {acceptance_rate:.3f}', ha='center',
                 transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')
        ax6.set_title('Simulation Statistics')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "monte_carlo_simulation.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    # Example run for this practice project
    mc = LennardJonesMC(
        n_atoms=64,
        box_length=8.0,
        temperature=1.0,
        epsilon=1.0,
        sigma=1.0,
        cutoff=3.0
    )

    mc.run(n_steps=50000, n_equil=5000, sample_interval=10)
    mc.visualize()

    print("\nPractice project simulation finished!")