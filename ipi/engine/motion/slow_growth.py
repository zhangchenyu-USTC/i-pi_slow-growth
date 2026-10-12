"""Length-coordinate slow-growth molecular dynamics.

This module implements a Liu/CP-MACE-compatible SHAKE propagation and an
optional time-dependent RATTLE velocity projection. The external force
provider is not involved in the constraint iterations. A coordinate can be
one minimum-image distance, a linear combination of such distances, the
distance from an atom to the geometric midpoint of two atoms, or a linear
combination of raw Cartesian coordinate components.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.engine.barostats import Barostat
from ipi.engine.motion.dynamics import Dynamics, DummyIntegrator
from ipi.engine.thermostats import Thermostat, ThermoSVR
from ipi.utils.depend import dstrip
from ipi.utils.units import Constants, unit_to_internal


_BLUE_MOON_FD_STEP = unit_to_internal("length", "angstrom", 1.0e-4)


class SlowGrowthDistanceConstraint:
    """A supported scalar, length-valued slow-growth coordinate."""

    def __init__(
        self,
        atoms,
        start,
        increment,
        tolerance,
        max_iterations,
        coefficients=None,
        coordinate_type="linear",
        midpoint_reference=None,
        has_midpoint_reference=False,
        components=None,
    ):
        atoms = np.asarray(atoms, dtype=int).reshape(-1)
        if np.any(atoms < 0):
            raise ValueError("Slow-growth atom indices must be non-negative")
        coordinate_type = str(coordinate_type).lower()
        if coordinate_type not in ("linear", "midpoint", "cartesian"):
            raise ValueError(
                "Slow-growth coordinate_type must be linear, midpoint, or cartesian"
            )
        self.coordinate_type = coordinate_type
        self.input_atoms = atoms.copy()
        components = np.asarray(
            [] if components is None else components, dtype=int
        ).reshape(-1)

        if self.coordinate_type == "linear":
            if len(components) != 0:
                raise ValueError(
                    "Slow-growth distance coordinates do not accept components"
                )
            if len(atoms) < 2 or len(atoms) % 2 != 0:
                raise ValueError(
                    "Slow growth requires one or more pairs of atom indices"
                )
            atom_pairs = atoms.reshape((-1, 2))
            if np.any(atom_pairs[:, 0] == atom_pairs[:, 1]):
                raise ValueError(
                    "Atoms in each slow-growth distance must be different"
                )
            if coefficients is None:
                coefficients = np.ones(len(atom_pairs), dtype=float)
            coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
            if len(coefficients) == 0:
                coefficients = np.ones(len(atom_pairs), dtype=float)
            if len(coefficients) != len(atom_pairs):
                raise ValueError(
                    "Slow-growth distance coefficients must match the number of atom pairs"
                )
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("Slow-growth distance coefficients must be finite")
            if not np.any(coefficients != 0.0):
                raise ValueError(
                    "At least one slow-growth distance coefficient must be non-zero"
                )
            self.atom_pairs = atom_pairs.copy()
            self.coefficients = coefficients.copy()
            self.midpoint_atoms = np.zeros(0, dtype=int)
            self.components = np.zeros(0, dtype=int)
        elif self.coordinate_type == "midpoint":
            if len(components) != 0:
                raise ValueError(
                    "Slow-growth midpoint coordinates do not accept components"
                )
            if len(atoms) != 3 or len(set(atoms.tolist())) != 3:
                raise ValueError(
                    "Slow-growth midpoint distance requires three different atom indices"
                )
            coefficients = np.asarray(
                [] if coefficients is None else coefficients, dtype=float
            ).reshape(-1)
            if len(coefficients) != 0:
                raise ValueError(
                    "Slow-growth midpoint distance does not accept coefficients"
                )
            self.atom_pairs = np.zeros((0, 2), dtype=int)
            self.coefficients = np.zeros(0, dtype=float)
            self.midpoint_atoms = atoms.copy()
            self.components = np.zeros(0, dtype=int)
        else:
            if len(atoms) == 0:
                raise ValueError(
                    "Slow-growth Cartesian coordinates require at least one atom"
                )
            if len(components) != len(atoms):
                raise ValueError(
                    "Slow-growth Cartesian components must match the number of atoms"
                )
            if np.any((components < 0) | (components > 2)):
                raise ValueError(
                    "Slow-growth Cartesian components must be 0 (x), 1 (y), or 2 (z)"
                )
            if coefficients is None:
                coefficients = np.ones(len(atoms), dtype=float)
            coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
            if len(coefficients) == 0:
                coefficients = np.ones(len(atoms), dtype=float)
            if len(coefficients) != len(atoms):
                raise ValueError(
                    "Slow-growth Cartesian coefficients must match the number of atoms"
                )
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("Slow-growth Cartesian coefficients must be finite")
            self.atom_pairs = np.zeros((0, 2), dtype=int)
            self.coefficients = coefficients.copy()
            self.midpoint_atoms = np.zeros(0, dtype=int)
            self.components = components.copy()

        midpoint_reference = np.asarray(
            [] if midpoint_reference is None else midpoint_reference, dtype=float
        ).reshape(-1)
        self.has_midpoint_reference = bool(has_midpoint_reference)
        if self.coordinate_type == "midpoint":
            if self.has_midpoint_reference:
                if len(midpoint_reference) != 3:
                    raise ValueError(
                        "Slow-growth midpoint restart reference must have three components"
                    )
                if not np.all(np.isfinite(midpoint_reference)):
                    raise ValueError(
                        "Slow-growth midpoint restart reference must be finite"
                    )
                self.midpoint_reference = midpoint_reference.copy()
            else:
                if len(midpoint_reference) != 0:
                    raise ValueError(
                        "Slow-growth midpoint reference requires "
                        "has_midpoint_reference=true"
                    )
                self.midpoint_reference = np.zeros(3, dtype=float)
        else:
            if self.has_midpoint_reference or len(midpoint_reference) != 0:
                raise ValueError(
                    "Slow-growth midpoint reference is only valid for midpoint coordinates"
                )
            self.midpoint_reference = np.zeros(3, dtype=float)

        self.atoms = np.asarray(list(dict.fromkeys(atoms.tolist())), dtype=int)
        self._atom_to_local = {
            int(atom): local for local, atom in enumerate(self.atoms)
        }
        self._cartesian_gradient = np.zeros((len(self.atoms), 3), dtype=float)
        if self.coordinate_type == "cartesian":
            for atom, component, coefficient in zip(
                self.input_atoms, self.components, self.coefficients
            ):
                self._cartesian_gradient[
                    self._atom_to_local[int(atom)], int(component)
                ] += coefficient
            if not np.any(self._cartesian_gradient != 0.0):
                raise ValueError(
                    "Slow-growth Cartesian terms cancel to a zero-gradient coordinate"
                )
        self.is_single_distance = bool(
            self.coordinate_type == "linear"
            and len(self.atom_pairs) == 1
            and self.coefficients[0] == 1.0
        )

        self.validate_target(start, "start coordinate")
        if not np.isfinite(increment):
            raise ValueError("Slow-growth coordinate increment must be finite")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Slow-growth SHAKE tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("Slow-growth maximum iteration count must be positive")

        self.start = float(start)
        self.increment = float(increment)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.beads = None
        self.cell = None

    def validate_target(self, target, label="target"):
        """Checks a scheduled value without forbidding signed combinations."""

        if not np.isfinite(target):
            raise ValueError("Slow-growth {} must be finite".format(label))
        if (
            self.is_single_distance or self.coordinate_type == "midpoint"
        ) and target <= 0.0:
            raise ValueError("Slow-growth distance {} must be positive".format(label))

    def bind(self, beads, cell):
        if beads.nbeads != 1:
            raise ValueError("Slow growth is only implemented for nbeads=1")
        if np.any(self.atoms >= beads.natoms):
            raise ValueError(
                "Slow-growth atom index is out of bounds for {:d} atoms".format(
                    beads.natoms
                )
            )

        if self.coordinate_type != "cartesian":
            h = np.asarray(dstrip(cell.h), dtype=float)
            if h.shape != (3, 3) or not np.all(np.isfinite(h)):
                raise ValueError("Slow growth requires a finite 3x3 simulation cell")
            if abs(np.linalg.det(h)) < 1.0e-14:
                raise ValueError(
                    "Slow growth with minimum-image distance requires a non-singular cell"
                )

        self.beads = beads
        self.cell = cell

    def describe(self):
        """Returns a compact description for diagnostics."""

        if self.coordinate_type == "midpoint":
            return "midpoint_atoms={}".format(self.midpoint_atoms.tolist())
        if self.coordinate_type == "cartesian":
            return "atoms={}, components={}, coefficients={}".format(
                self.input_atoms.tolist(),
                self.components.tolist(),
                self.coefficients.tolist(),
            )
        return "pairs={}, coefficients={}".format(
            self.atom_pairs.tolist(), self.coefficients.tolist()
        )

    def _positions(self, q, pair_index=0):
        q = np.asarray(q, dtype=float).reshape(-1)
        i, j = self.atom_pairs[pair_index]
        return q[3 * i : 3 * i + 3], q[3 * j : 3 * j + 3]

    def displacement(self, q, pair_index=0):
        """Returns r_i-r_j in the minimum-image convention."""

        if self.cell is None:
            raise RuntimeError("Slow-growth distance constraint has not been bound")
        qi, qj = self._positions(q, pair_index)
        h = np.asarray(dstrip(self.cell.h), dtype=float)
        ih = np.asarray(dstrip(self.cell.ih), dtype=float)
        scaled = np.dot(ih, qi - qj)
        scaled -= np.rint(scaled)
        return np.dot(h, scaled)

    def _midpoint_base_fractional(self, q):
        """Returns the pairwise-MIC midpoint vector in fractional coordinates."""

        if self.cell is None:
            raise RuntimeError("Slow-growth distance constraint has not been bound")
        q = np.asarray(q, dtype=float).reshape(-1)
        anchor, first, second = self.midpoint_atoms
        qa = q[3 * anchor : 3 * anchor + 3]
        q1 = q[3 * first : 3 * first + 3]
        q2 = q[3 * second : 3 * second + 3]
        ih = np.asarray(dstrip(self.cell.ih), dtype=float)

        anchor_to_first = np.dot(ih, q1 - qa)
        anchor_to_first -= np.rint(anchor_to_first)
        first_to_second = np.dot(ih, q2 - q1)
        first_to_second -= np.rint(first_to_second)
        return anchor_to_first + 0.5 * first_to_second

    def midpoint_fractional(self, q):
        """Returns the midpoint vector on the continuous stored image branch.

        Pairwise minimum images define the molecular midpoint. The remaining
        whole-cell translation is selected to be closest to the last accepted
        midpoint vector. Trial SHAKE evaluations never change that reference.
        """

        midpoint = self._midpoint_base_fractional(q)
        if self.has_midpoint_reference:
            midpoint += np.rint(self.midpoint_reference - midpoint)
        else:
            midpoint -= np.rint(midpoint)
        return midpoint

    def initialize_midpoint_reference(self, q):
        """Initializes or refreshes the image from an accepted configuration."""

        if self.coordinate_type != "midpoint":
            return
        midpoint = self.midpoint_fractional(q)
        self.midpoint_reference = midpoint.copy()
        self.has_midpoint_reference = True

    def commit_midpoint_reference(self, q):
        """Commits the continuous image after a successful SHAKE projection."""

        self.initialize_midpoint_reference(q)

    def midpoint_displacement(self, q):
        """Returns the continuous anchor-to-midpoint Cartesian vector."""

        midpoint = self.midpoint_fractional(q)
        h = np.asarray(dstrip(self.cell.h), dtype=float)
        return np.dot(h, midpoint)

    def value(self, q):
        if self.coordinate_type == "midpoint":
            return float(np.linalg.norm(self.midpoint_displacement(q)))
        if self.coordinate_type == "cartesian":
            q = np.asarray(q, dtype=float).reshape((-1, 3))
            value = 0.0
            for atom, component, coefficient in zip(
                self.input_atoms, self.components, self.coefficients
            ):
                value += coefficient * q[int(atom), int(component)]
            return float(value)
        value = 0.0
        for pair_index, coefficient in enumerate(self.coefficients):
            value += coefficient * np.linalg.norm(self.displacement(q, pair_index))
        return float(value)

    def residual(self, q, target):
        return self.value(q) - float(target)

    def gradient(self, q):
        """Returns the accumulated derivative for every involved atom."""

        if self.coordinate_type == "midpoint":
            vector = self.midpoint_displacement(q)
            distance = np.linalg.norm(vector)
            if distance < 1.0e-14:
                raise ValueError(
                    "Slow-growth midpoint distance is zero; its gradient is undefined"
                )
            unit = vector / distance
            anchor, first, second = self.midpoint_atoms
            gradient = np.zeros((len(self.atoms), 3), dtype=float)
            gradient[self._atom_to_local[int(anchor)]] -= unit
            gradient[self._atom_to_local[int(first)]] += 0.5 * unit
            gradient[self._atom_to_local[int(second)]] += 0.5 * unit
            return gradient

        if self.coordinate_type == "cartesian":
            return self._cartesian_gradient.copy()

        gradient = np.zeros((len(self.atoms), 3), dtype=float)
        for pair_index, ((atom_i, atom_j), coefficient) in enumerate(
            zip(self.atom_pairs, self.coefficients)
        ):
            rij = self.displacement(q, pair_index)
            distance = np.linalg.norm(rij)
            if distance < 1.0e-14:
                raise ValueError(
                    "Slow-growth distance pair {} is zero; its gradient is undefined".format(
                        [int(atom_i), int(atom_j)]
                    )
                )
            unit = rij / distance
            gradient[self._atom_to_local[int(atom_i)]] += coefficient * unit
            gradient[self._atom_to_local[int(atom_j)]] -= coefficient * unit
        return gradient

    def metric_tensor(self, q, m3):
        """Returns Z = grad(xi) M^-1 grad(xi)^T."""

        gradient = self.gradient(q)
        m3 = np.asarray(m3, dtype=float).reshape(-1)
        value = 0.0
        for local, atom in enumerate(self.atoms):
            mass = m3[3 * atom]
            if mass <= 0.0 or not np.isfinite(mass):
                raise ValueError("Slow-growth constrained atoms need finite masses")
            value += np.dot(gradient[local], gradient[local]) / mass
        return float(value)

    def metric_gradient_correction(self, q, m3):
        """Returns VASP's one-constraint Blue-Moon geometric factor G.

        VASP evaluates the Cartesian derivative of ``det(Z)`` by centered
        finite differences and contracts it with ``grad(xi) M^-1``. For one
        constraint this is ``G=(grad(xi) M^-1 grad(Z))/(2 Z^2)``.
        """

        if self.is_single_distance or self.coordinate_type in (
            "midpoint",
            "cartesian",
        ):
            return 0.0

        q = np.asarray(q, dtype=float).reshape(-1)
        m3 = np.asarray(m3, dtype=float).reshape(-1)
        gradient = self.gradient(q)
        metric = self.metric_tensor(q, m3)
        if not np.isfinite(metric) or metric <= 0.0:
            raise RuntimeError("Slow-growth constraint has a singular mass metric")

        metric_derivative = np.zeros_like(gradient)
        for local, atom in enumerate(self.atoms):
            for component in range(3):
                index = 3 * atom + component
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[index] += _BLUE_MOON_FD_STEP
                q_minus[index] -= _BLUE_MOON_FD_STEP
                metric_derivative[local, component] = (
                    self.metric_tensor(q_plus, m3) - self.metric_tensor(q_minus, m3)
                ) / (2.0 * _BLUE_MOON_FD_STEP)

        directional_derivative = 0.0
        for local, atom in enumerate(self.atoms):
            mass = m3[3 * atom]
            directional_derivative += np.dot(
                gradient[local] / mass, metric_derivative[local]
            )
        return float(directional_derivative / (2.0 * metric * metric))


class SlowGrowthSHAKESolver:
    """SHAKE position and affine RATTLE velocity projections."""

    def __init__(self, constraint):
        self.constraint = constraint

    def project_position(self, q_old, p_trial, m3, dt, target):
        """Projects a full drift onto the requested distance.

        The iteration follows the public CP-MACE slow-growth implementation:
        it uses the distance residual directly, updates the momentum, and
        reconstructs the trial coordinates from the beginning of the drift.
        """

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Slow-growth timestep must be positive")
        self.constraint.validate_target(target)

        q_old = np.asarray(q_old, dtype=float).reshape(-1).copy()
        p_new = np.asarray(p_trial, dtype=float).reshape(-1).copy()
        m3 = np.asarray(m3, dtype=float).reshape(-1)
        self.constraint.initialize_midpoint_reference(q_old)
        q_new = q_old + dt * p_new / m3
        gradient_old = self.constraint.gradient(q_old)
        lambda_sum = 0.0

        for iteration in range(self.constraint.max_iterations + 1):
            error = self.constraint.residual(q_new, target)
            if abs(error) <= self.constraint.tolerance:
                return q_new, p_new, lambda_sum, iteration, error
            if iteration == self.constraint.max_iterations:
                break

            gradient_new = self.constraint.gradient(q_new)
            denominator = 0.0
            for local, atom in enumerate(self.constraint.atoms):
                mass = m3[3 * atom]
                denominator += np.dot(gradient_old[local], gradient_new[local]) / mass
            denominator *= dt * dt

            if not np.isfinite(denominator) or abs(denominator) < 1.0e-30:
                raise RuntimeError(
                    "Slow-growth SHAKE encountered a singular constraint Jacobian"
                )

            delta_lambda = error / denominator
            lambda_sum += delta_lambda
            for local, atom in enumerate(self.constraint.atoms):
                sl = slice(3 * atom, 3 * atom + 3)
                p_new[sl] -= dt * delta_lambda * gradient_new[local]
            q_new = q_old + dt * p_new / m3

        actual = self.constraint.value(q_new)
        raise RuntimeError(
            "Slow-growth SHAKE did not converge after {:d} iterations: "
            "coordinate={}, target={:.16e}, actual={:.16e}, "
            "error={:.16e}".format(
                self.constraint.max_iterations,
                self.constraint.describe(),
                target,
                actual,
                actual - target,
            )
        )

    def project_velocity(self, q, p, m3, target_velocity):
        """Applies the affine RATTLE condition grad(xi).v = target_velocity."""

        p_new = np.asarray(p, dtype=float).reshape(-1).copy()
        m3 = np.asarray(m3, dtype=float).reshape(-1)
        self.constraint.initialize_midpoint_reference(q)
        gradient = self.constraint.gradient(q)
        projected_velocity = 0.0
        for local, atom in enumerate(self.constraint.atoms):
            sl = slice(3 * atom, 3 * atom + 3)
            projected_velocity += np.dot(gradient[local], p_new[sl] / m3[sl])

        metric = self.constraint.metric_tensor(q, m3)
        impulse = (projected_velocity - target_velocity) / metric
        for local, atom in enumerate(self.constraint.atoms):
            sl = slice(3 * atom, 3 * atom + 3)
            p_new[sl] -= impulse * gradient[local]

        residual = 0.0
        for local, atom in enumerate(self.constraint.atoms):
            sl = slice(3 * atom, 3 * atom + 3)
            residual += np.dot(gradient[local], p_new[sl] / m3[sl])
        residual -= target_velocity
        return p_new, float(impulse), float(residual)


class SlowGrowthIntegrator(DummyIntegrator):
    """Velocity-Verlet/CSVR integrator with one full SHAKE drift per step."""

    def get_qdt(self):
        return self.dt

    def bind(self, motion):
        super().bind(motion)
        self.motion = motion

    def _kinetic_energy(self):
        p = np.asarray(dstrip(self.beads.p[0]), dtype=float)
        m3 = np.asarray(dstrip(self.beads.m3[0]), dtype=float)
        return 0.5 * float(np.dot(p, p / m3))

    def _force_half_step(self):
        force = np.asarray(dstrip(self.forces.mts_forces[0].f[0]), dtype=float).copy()
        if self.ensemble.has_bias:
            force += np.asarray(dstrip(self.bias.f[0]), dtype=float)
        if len(getattr(self, "fixatoms_dof", ())) > 0:
            self.beads.p[0, self.activeatoms_mask] += force[self.activeatoms_mask] * (
                0.5 * self.dt
            )
        else:
            self.beads.p[0] += force * (0.5 * self.dt)
        self.pconstraints()

    def _rattle_velocity_projection(self, target_velocity, thermostat=False):
        kinetic_before = self._kinetic_energy() if thermostat else 0.0
        p_new, impulse, residual = self.motion.solver.project_velocity(
            dstrip(self.beads.q[0]),
            dstrip(self.beads.p[0]),
            dstrip(self.beads.m3[0]),
            target_velocity,
        )
        self.beads.p[0] = p_new
        if thermostat:
            self.ensemble.eens += kinetic_before - self._kinetic_energy()
        self.motion._step_velocity_impulse += impulse
        self.motion.last_velocity_error = residual

    def _thermostat_half_step(self, target_velocity):
        self.thermostat.step()
        self.pconstraints()
        if self.motion.algorithm == "rattle":
            self._rattle_velocity_projection(target_velocity, thermostat=True)

    def step(self, step=None):
        target, current_velocity, next_velocity = self.motion.begin_step(step)
        self.motion._step_velocity_impulse = 0.0

        if self.enstype == "nvt":
            self._thermostat_half_step(current_velocity)

        self._force_half_step()

        q_new, p_new, shake_lambda, iterations, error = (
            self.motion.solver.project_position(
                dstrip(self.beads.q[0]),
                dstrip(self.beads.p[0]),
                dstrip(self.beads.m3[0]),
                self.dt,
                target,
            )
        )
        self.motion.constraint.commit_midpoint_reference(q_new)
        self.beads.q[0] = q_new
        self.beads.p[0] = p_new

        self._force_half_step()

        if self.motion.algorithm == "rattle":
            self._rattle_velocity_projection(next_velocity)

        if self.enstype == "nvt":
            self._thermostat_half_step(next_velocity)

        self.motion.finish_step(shake_lambda, iterations, error)


class SlowGrowthDynamics(Dynamics):
    """Classical length-coordinate slow-growth molecular dynamics."""

    def __init__(
        self,
        timestep,
        mode="nve",
        algorithm="shake",
        thermostat=None,
        fixcom=False,
        fixatoms_dof=None,
        atoms=None,
        coefficients=None,
        coordinate_type="linear",
        midpoint_reference=None,
        has_midpoint_reference=False,
        components=None,
        start=-1.0,
        increment=0.0,
        tolerance=1.0e-8,
        max_iterations=1000,
        current_target=-1.0,
        last_target=-1.0,
        completed_steps=0,
        cumulative_work=0.0,
        previous_cv=-1.0,
        previous_gradient=0.0,
        has_previous=False,
        last_lambda=0.0,
        last_shake_lambda=0.0,
        last_velocity_lambda=0.0,
        last_gradient=0.0,
        last_error=0.0,
        last_iterations=0,
        last_velocity_error=0.0,
    ):
        if mode not in ("nve", "nvt"):
            raise ValueError("Slow growth only supports nve and nvt")
        if algorithm not in ("shake", "rattle"):
            raise ValueError("Slow-growth algorithm must be shake or rattle")
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("Slow-growth timestep must be positive")
        if thermostat is None:
            thermostat = ThermoSVR() if mode == "nvt" else Thermostat()

        super().__init__(
            timestep=timestep,
            mode=mode,
            splitting="obabo",
            thermostat=thermostat,
            barostat=Barostat(),
            fixcom=fixcom,
            fixatoms_dof=fixatoms_dof,
            nmts=np.asarray([1], dtype=int),
        )

        self.algorithm = algorithm
        self.constraint = SlowGrowthDistanceConstraint(
            atoms=atoms,
            start=start,
            increment=increment,
            tolerance=tolerance,
            max_iterations=max_iterations,
            coefficients=coefficients,
            components=components,
            coordinate_type=coordinate_type,
            midpoint_reference=midpoint_reference,
            has_midpoint_reference=has_midpoint_reference,
        )
        self.solver = SlowGrowthSHAKESolver(self.constraint)
        self.integrator = SlowGrowthIntegrator()

        self.start = float(start)
        self.increment = float(increment)
        self.current_target = (
            self.start
            if float(current_target) == -1.0 and int(completed_steps) == 0
            else float(current_target)
        )
        self.last_target = (
            self.start
            if float(last_target) == -1.0 and int(completed_steps) == 0
            else float(last_target)
        )
        self.completed_steps = int(completed_steps)
        self.cumulative_work = float(cumulative_work)
        self.previous_cv = float(previous_cv)
        self.previous_gradient = float(previous_gradient)
        self.has_previous = bool(has_previous)

        self.last_cv = self.last_target
        self.last_lambda = float(last_lambda)
        self.last_shake_lambda = float(last_shake_lambda)
        self.last_velocity_lambda = float(last_velocity_lambda)
        self.last_gradient = float(last_gradient)
        self.last_error = float(last_error)
        self.last_iterations = int(last_iterations)
        self.last_velocity_error = float(last_velocity_error)
        self.metric_z = 0.0
        self.metric_weight = 0.0
        self.gkt = 0.0
        self.weighted_gradient = 0.0
        self._step_velocity_impulse = 0.0

        if self.completed_steps < 0:
            raise ValueError("Slow-growth completed_steps cannot be negative")
        if not np.isfinite(self.current_target):
            raise ValueError("Slow-growth current_target must be finite")
        self.constraint.validate_target(self.current_target, "current target")
        self.constraint.validate_target(self.last_target, "last target")
        if self.last_iterations < 0:
            raise ValueError("Slow-growth last_iterations cannot be negative")
        restart_scalars = (
            self.cumulative_work,
            self.previous_cv,
            self.previous_gradient,
            self.last_lambda,
            self.last_shake_lambda,
            self.last_velocity_lambda,
            self.last_gradient,
            self.last_error,
            self.last_velocity_error,
        )
        if not np.all(np.isfinite(restart_scalars)):
            raise ValueError("Slow-growth restart state must contain finite values")
        if self.has_previous and not np.isfinite(self.previous_cv):
            raise ValueError("Slow-growth previous_cv must be finite")

    def get_fixdof(self):
        return super().get_fixdof() + 1

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        if beads.nbeads != 1:
            raise ValueError("Slow growth is only implemented for nbeads=1")
        self.constraint.validate_target(self.current_target, "scheduled target")
        if self.enstype == "nvt" and not isinstance(self.thermostat, ThermoSVR):
            raise ValueError(
                "Slow-growth NVT currently requires the CSVR/SVR thermostat"
            )
        if bforce.nmtslevels != 1:
            raise ValueError("Slow growth currently supports one force/MTS level only")

        self.constraint.bind(beads, cell)
        self.constraint.initialize_midpoint_reference(dstrip(beads.q[0]))
        if self.constraint.coordinate_type == "cartesian":
            constrained_dofs = np.asarray(
                [
                    3 * atom + component
                    for local, atom in enumerate(self.constraint.atoms)
                    for component in range(3)
                    if self.constraint._cartesian_gradient[local, component] != 0.0
                ],
                dtype=int,
            )
        else:
            constrained_dofs = np.concatenate(
                [
                    np.arange(3 * atom, 3 * atom + 3, dtype=int)
                    for atom in self.constraint.atoms
                ]
            )
        overlap = np.intersect1d(self.fixatoms_dof, constrained_dofs)
        if len(overlap) > 0:
            raise ValueError(
                "Slow-growth constraint gradients overlap fixed Cartesian DOFs: {}".format(
                    overlap.tolist()
                )
            )
        actual = self.constraint.value(dstrip(beads.q[0]))
        if abs(actual - self.last_target) > self.constraint.tolerance:
            raise ValueError(
                "Initial slow-growth coordinate does not match the expected target: "
                "coordinate={}, expected={:.16e}, actual={:.16e}, "
                "error={:.16e}, "
                "tolerance={:.16e}".format(
                    self.constraint.describe(),
                    self.last_target,
                    actual,
                    actual - self.last_target,
                    self.constraint.tolerance,
                )
            )

        super().bind(ens, beads, nm, cell, bforce, prng, omaker)

        self.last_cv = actual
        self.last_error = actual - self.last_target
        metric = self.constraint.metric_tensor(
            dstrip(self.beads.q[0]), dstrip(self.beads.m3[0])
        )
        self.metric_z = metric
        self.metric_weight = metric ** (-0.5)
        geometric_correction = self.constraint.metric_gradient_correction(
            dstrip(self.beads.q[0]), dstrip(self.beads.m3[0])
        )
        self.gkt = geometric_correction * Constants.kb * self.ensemble.temp
        self.weighted_gradient = self.metric_weight * (self.last_gradient + self.gkt)

        if self.algorithm == "rattle":
            initial_velocity = (self.current_target - self.last_target) / self.dt
            p_new, _, residual = self.solver.project_velocity(
                dstrip(self.beads.q[0]),
                dstrip(self.beads.p[0]),
                dstrip(self.beads.m3[0]),
                initial_velocity,
            )
            self.beads.p[0] = p_new
            self.last_velocity_error = residual

    def begin_step(self, step=None):
        self.constraint.validate_target(self.current_target, "scheduled target")
        current_velocity = (self.current_target - self.last_target) / self.dt
        next_velocity = self.increment / self.dt
        return self.current_target, current_velocity, next_velocity

    def finish_step(self, shake_lambda, iterations, error):
        self.last_shake_lambda = float(shake_lambda)
        self.last_velocity_lambda = (
            self._step_velocity_impulse / self.dt if self.algorithm == "rattle" else 0.0
        )
        self.last_lambda = self.last_shake_lambda + self.last_velocity_lambda
        self.last_gradient = -self.last_lambda
        self.last_iterations = int(iterations)

        self.last_target = self.current_target
        self.last_cv = self.constraint.value(dstrip(self.beads.q[0]))
        self.last_error = self.last_cv - self.last_target
        if abs(self.last_error) > self.constraint.tolerance:
            raise RuntimeError(
                "Slow-growth constraint was lost after integration: error={:.16e}".format(
                    self.last_error
                )
            )
        if abs(error) > self.constraint.tolerance:
            raise RuntimeError(
                "Slow-growth SHAKE returned an unconverged residual: {:.16e}".format(
                    error
                )
            )

        if self.has_previous:
            self.cumulative_work += (
                0.5
                * (self.previous_gradient + self.last_gradient)
                * (self.last_cv - self.previous_cv)
            )
        self.previous_cv = self.last_cv
        self.previous_gradient = self.last_gradient
        self.has_previous = True

        metric = self.constraint.metric_tensor(
            dstrip(self.beads.q[0]), dstrip(self.beads.m3[0])
        )
        self.metric_z = metric
        self.metric_weight = metric ** (-0.5)
        geometric_correction = self.constraint.metric_gradient_correction(
            dstrip(self.beads.q[0]), dstrip(self.beads.m3[0])
        )
        temperature = self.ensemble.temp if geometric_correction != 0.0 else 0.0
        self.gkt = geometric_correction * Constants.kb * temperature
        self.weighted_gradient = self.metric_weight * (self.last_gradient + self.gkt)

        self.completed_steps += 1
        self.current_target = self.last_target + self.increment
