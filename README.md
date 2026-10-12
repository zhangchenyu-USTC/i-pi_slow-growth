# i-PI-3.3.0-slow-growth

This project implements strictly constrained slow-growth molecular dynamics on top of i-PI 3.3.0.

Many enhanced-sampling methods are available in i-PI through the PLUMED Python interface. However, unlike strictly constrained slow-growth, PLUMED's
[`MOVINGRESTRAINT`](https://www.plumed.org/doc-v2.10/user-doc/html/_m_o_v_i_n_g_r_e_s_t_r_a_i_n_t.html)
pulls a collective variable (CV) using a moving harmonic restraint. The actual CV can deviate from its target. The spring constant, timestep, and pulling rate can therefore affect finite-sampling results.

Here, SHAKE constrains the CV to its target within a specified numerical tolerance after every completed MD step, without requiring a spring constant. A RATTLE mode additionally projects velocities to satisfy the moving constraint.
The constraints are enforced by i-PI. Energies and physical forces are supplied by standard i-PI clients, such as VASP, CP2K, or MACE.

## Installation

Python ≥ 3.10 and NumPy < 3 are required. A separate virtual environment is recommended to avoid overwriting other i-PI installations.

Install from the repository root:

```bash
python -m pip install -e .
```

## Input example

Replace the `<motion>` block inside `<system>` in a standard i-PI `input.xml` with the following block.
Initialization, force clients, output, and total steps use the standard i-PI settings. Set the NVT temperature in `<ensemble>`.

Complete VASP client inputs and the initial structure are provided in the [Volmer example](examples/features/slow_growth/volmer/README.md).
The example adapts the K⁺-coordinated water Volmer pathway from Qin, Vegge, and Hansen's
[*Cation-Coordinated Inner-Sphere CO2 Electroreduction at Au–Water Interfaces*](https://doi.org/10.1021/jacs.2c11643)
and their [open dataset](https://doi.org/10.5281/zenodo.7474433).
The CV is

$$
\xi=d(\mathrm{O},\mathrm{H})-d(\mathrm{H},\mathrm{Au}),
$$

with atom indices and coefficients specified below. This example uses Γ-point sampling, i-PI SHAKE, and canonical stochastic velocity rescaling (CSVR/SVR), with τ=200 fs and a CV tolerance of 1e−6 Å.

```xml
    <motion mode="slow_growth">
      <fixcom>false</fixcom>
      <slow_growth mode="nvt" algorithm="shake">
        <timestep units="femtosecond">1.0</timestep>
        <thermostat mode="svr">
          <tau units="femtosecond">200.0</tau>
        </thermostat>
        <distance>
          <atoms>[119, 68, 68, 134]</atoms>
          <coefficients>[1.0, -1.0]</coefficients>
          <start units="angstrom">-1.2413615592403282</start>
          <increment units="angstrom">0.001</increment>
        </distance>
        <shake>
          <tolerance units="angstrom">1.0e-6</tolerance>
          <max_iterations>1000</max_iterations>
        </shake>
      </slow_growth>
    </motion>
```

Atom indices are **0-based**. The initial CV must match `start` within the specified tolerance; otherwise, the simulation stops with an error.
The first step uses `start`; `increment` is added only after that step is completed. A negative increment reverses the scan. A zero increment fixes the CV for Blue-Moon sampling.
Changing `algorithm="shake"` to `"rattle"` also explicitly constrains the rate of change of the CV.

Supported scalar, length-valued CVs include bond distances and their linear combinations, including distance differences.
Distances from an atom to the geometric midpoint of two atoms and linear combinations of Cartesian components are also supported.
The latter include differences between two atoms' Z coordinates.
The current implementation supports one bead, a fixed cell, NVE or NVT (CSVR/SVR), and a single force/MTS level.

## Free-energy analysis

### 1. Which properties should be output?

Add the following properties to `<output>` inside `<simulation>`. Adjust the filename and stride as needed.

```xml
<properties stride="1" filename="out">
  [ step, time{femtosecond}, temperature{kelvin},
    slow_growth_cv{angstrom}, slow_growth_target{angstrom},
    slow_growth_error{angstrom}, slow_growth_gradient{ev/ang},
    slow_growth_metric, slow_growth_gkt{ev/ang},
    slow_growth_work{electronvolt} ]
</properties>
```

| Property | Meaning | Role in free-energy analysis |
| --- | --- | --- |
| `slow_growth_cv` | Actual CV, denoted ξ | Required as the reaction coordinate for binning and integration |
| `slow_growth_gradient` | Raw instantaneous gradient, $`g_{\mathrm{raw}}=-\lambda_{\mathrm{internal}}`$ | Required for mean-gradient estimation; do not take its absolute value |
| `slow_growth_metric` | Blue-Moon weight, $`w=Z^{-1/2}`$ | Required for weighted averaging with general CVs; a constant weight cancels |
| `slow_growth_gkt` | Geometric/mass-metric correction, $`Gk_BT`$ | Required for general CVs; the correction can be omitted when known to be zero |
| `slow_growth_target`, `slow_growth_error` | Target used for this step, and actual CV minus target | Check whether the constraint is satisfied |
| `slow_growth_work` | Cumulative integral of the raw gradient | For monitoring only; not required for general free-energy post-processing |

Additional outputs include total, position, and velocity constraint multipliers; Z; the weighted gradient; SHAKE iteration counts; and velocity residuals.
See the [usage guide](docs/slow-growth-usage.md) for details.
`slow_growth_lambda` has the opposite sign to the raw gradient. Only one of these two columns is needed for integration.

### 2. How is the free-energy curve obtained?

For NVT data, first inspect the trajectory and temperature. Concatenate restart segments in their actual continuation order and remove duplicate frames.
Bin a continuous slow-growth trajectory by CV, for example using an initial bin width of 0.05 Å.
For fixed-CV windows, discard equilibration before averaging the production samples in each window.

The Blue-Moon mean gradient in bin or window j is

$$
\bar g_j=\frac{\sum_{n\in j}w_n\left[g_{\mathrm{raw},n}+\left(Gk_BT\right)_n\right]}{\sum_{n\in j}w_n}.
$$

Then integrate using the trapezoidal rule, with the bin-mean CV or window target as the coordinate:

$$
A_{j+1}=A_j+\frac{\bar g_j+\bar g_{j+1}}{2}\left(\xi_{j+1}-\xi_j\right).
$$

Set the reactant reference point to zero to obtain the relative free-energy curve along the selected CV.
With the gradient and GkT output in eV/Å and the CV in Å, the integral is in eV.
Once the reactant, transition region, and product basin have been identified, the activation free energy and reaction free energy along this CV are

$$
\Delta A^\ddagger=A_{\mathrm{TS}}-A_{\mathrm{R}},\qquad \Delta A=A_{\mathrm{P}}-A_{\mathrm{R}}.
$$

Mean-gradient integration provides a reliable equilibrium free-energy estimate only for sufficiently slow scans or adequately equilibrated fixed windows.

### 3. Why does slow_growth_work not replace this analysis?

`slow_growth_work` accumulates a stepwise trapezoidal integral of the raw gradient along the actual CV. It neither adds GkT nor performs Blue-Moon weighted averaging.

For a single unit-coefficient bond distance, the current midpoint-distance CV, and Cartesian linear combinations, Z is constant and G=0.
The Blue-Moon correction then reduces to ordinary averaging of the raw gradient followed by integration.
For general bond-distance combinations, Z can depend on the configuration. Use the weighted formula above rather than treating `slow_growth_work` as a corrected free energy.
With `increment=0` in a fixed window, this cumulative quantity remains essentially unchanged and does not represent free-energy differences between windows.

### 4. Test convergence with respect to the growth rate

One slow-growth trajectory provides **one free-energy estimate at the chosen growth rate**. It does not by itself establish convergence to the equilibrium free energy.
The growth rate is `increment / timestep`. Reducing the increment at a fixed timestep increases the sampling time spent traversing the same CV interval.
Keep the scanned CV range unchanged and increase the total number of steps accordingly.

Compare mean gradients, barriers, and reaction free energies at different growth rates. Also check independent replicate trajectories and forward/reverse scans.
Even with a strictly constrained CV, other degrees of freedom, such as water-network rearrangements and ion coordination, may not equilibrate sufficiently quickly.
For sampling-convergence considerations, see
[*Constant-Potential Machine Learning Force Field for the Electrochemical Interface*](https://doi.org/10.1021/acs.jctc.5c00784),
J. Chem. Theory Comput. (2025). A suitable growth rate must be established for each system.

### 5. Recommended workflow

An alternative is to scan the full CV range at a faster growth rate and save configurations at different CV values.
Create separate fixed windows, set each window's `start` to its initial structure's actual CV, and set `increment` to `0.0` for Blue-Moon sampling.

For fixed windows, `algorithm="rattle"` is recommended to satisfy both the position constraint and the condition of zero CV velocity.
Equilibrate each window before collecting sufficiently long production samples. Check block-averaged gradients, different initial configurations, and the stability of uncertainties with increasing sampling time.
Finally, integrate the window-averaged gradients using the equations in Section 2.

Initialize each window from a new input rather than only changing `increment` in an old checkpoint.
The checkpoint also stores the next target and the original scan's integration history.

This workflow allows slower-converging windows to be extended individually and may improve convergence. Other degrees of freedom must still be checked for adequate sampling.

## License and citation

This project is based on i-PI 3.3.0 and retains its [license](licenses/LICENSE.md) and original copyright notices.
When using i-PI, please cite the [i-PI 3.0 paper](https://doi.org/10.1063/5.0215869).
