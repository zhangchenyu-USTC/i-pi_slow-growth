# Volmer: K⁺-coordinated water dissociation

## Source and adaptation

This example is adapted from the work of Xueping Qin, Tejs Vegge, and Heine Anton Hansen:
[*Cation-Coordinated Inner-Sphere CO2 Electroreduction at Au–Water Interfaces*](https://doi.org/10.1021/jacs.2c11643),
J. Am. Chem. Soc. **145**, 1897–1905 (2023).
The structure and reaction path come from the authors' [open Zenodo dataset](https://doi.org/10.5281/zenodo.7474433),
specifically `SG_HER/Volmer/K_coordinated_H2O` in `SG_HER.zip`.

The structure data retain the original [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
and are attributed to the authors listed above. The repository's code license does not replace the original data license.
**POTCAR, VASP executables, simulation results, and scripts specific to the original server are not distributed with this example.**

## Reaction coordinate and settings

The collective variable (CV) is the reaction coordinate, defined as the difference between the O–H and H–Au distances:

$$
\xi=d(\mathrm{O},\mathrm{H})-d(\mathrm{H},\mathrm{Au})
$$

The i-PI 0-based atom indices are O=119, H=68, and Au=134, giving:

```xml
<distance>
  <atoms>[119, 68, 68, 134]</atoms>
  <coefficients>[1.0, -1.0]</coefficients>
  <start units="angstrom">-1.2413615592403282</start>
  <increment units="angstrom">0.001</increment>
</distance>
```

| Parameter | Setting in this example |
| --- | --- |
| Temperature and ensemble | 298 K, NVT |
| Algorithm and timestep | SHAKE, 1 fs |
| SVR time constant | 200 fs |
| CV tolerance | 1e−6 Å |
| Scan increment and total steps | +0.001 Å/step, 2000 steps |
| Final-frame target | 0.7576384407596719 Å |
| Structure and fixed atoms | 173 atoms; 24 fixed Au atoms, with indices listed in input.xml |
| VASP force settings | ENCUT=400 eV, IVDW=11, ALGO=V, Γ-only 1×1×1 |

## Running the example

See `sbatch.slurm` for the submission script used to run this example.
