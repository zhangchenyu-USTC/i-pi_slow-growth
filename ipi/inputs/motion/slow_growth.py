"""Input handling for length-coordinate slow-growth dynamics."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

import ipi.engine.thermostats
from ipi.inputs.thermostats import InputThermo
from ipi.utils.inputvalue import (
    InputArray,
    InputAttribute,
    InputDictionary,
    InputValue,
    input_default,
)

__all__ = ["InputSlowGrowth"]


class InputSlowGrowthDistance(InputDictionary):
    attribs = {
        "coordinate_type": (
            InputAttribute,
            {
                "dtype": str,
                "default": "linear",
                "options": ["linear", "midpoint", "cartesian"],
                "help": (
                    "Linear combination of pair distances, an atom-to-"
                    "geometric-midpoint distance compatible with VASP M, or "
                    "a linear combination of raw Cartesian components."
                ),
            },
        ),
    }
    fields = {
        "atoms": (
            InputArray,
            {
                "dtype": int,
                "default": np.zeros(0, dtype=int),
                "help": (
                    "Zero-based atoms defining the coordinate. Linear distance "
                    "coordinates interpret them as flattened atom pairs."
                ),
            },
        ),
        "coefficients": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros(0, dtype=float),
                "help": (
                    "Linear coefficient for each distance or Cartesian term; "
                    "defaults to one."
                ),
            },
        ),
        "components": (
            InputArray,
            {
                "dtype": int,
                "default": np.zeros(0, dtype=int),
                "help": (
                    "For coordinate_type=cartesian, one component per atom: "
                    "0=x, 1=y, and 2=z."
                ),
            },
        ),
        "midpoint_reference": (
            InputArray,
            {
                "dtype": float,
                "default": np.zeros(0, dtype=float),
                "help": (
                    "Restart state: last accepted continuous midpoint vector "
                    "in fractional coordinates."
                ),
            },
        ),
        "has_midpoint_reference": (
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Restart state: whether midpoint_reference is initialized.",
            },
        ),
        "start": (
            InputValue,
            {
                "dtype": float,
                "default": -1.0,
                "dimension": "length",
                "help": "Length-coordinate target used in the first slow-growth MD step.",
            },
        ),
        "increment": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "length",
                "help": "Signed coordinate change after each complete MD step.",
            },
        ),
        "current_target": (
            InputValue,
            {
                "dtype": float,
                "default": -1.0,
                "dimension": "length",
                "help": "Restart state: target to use in the next MD step.",
            },
        ),
        "last_target": (
            InputValue,
            {
                "dtype": float,
                "default": -1.0,
                "dimension": "length",
                "help": "Restart state: target used in the last completed MD step.",
            },
        ),
    }
    dynamic = {}
    default_help = (
        "Defines a supported scalar length coordinate and its schedule."
    )
    default_label = "SLOWGROWTHDISTANCE"

    def store(self, motion):
        self.coordinate_type.store(motion.constraint.coordinate_type)
        self.atoms.store(motion.constraint.input_atoms)
        self.coefficients.store(motion.constraint.coefficients)
        self.components.store(motion.constraint.components)
        if motion.constraint.has_midpoint_reference:
            self.midpoint_reference.store(motion.constraint.midpoint_reference)
        else:
            self.midpoint_reference.store(np.zeros(0, dtype=float))
        self.has_midpoint_reference.store(
            motion.constraint.has_midpoint_reference
        )
        self.start.store(motion.start)
        self.increment.store(motion.increment)
        self.current_target.store(motion.current_target)
        self.last_target.store(motion.last_target)

    def fetch(self):
        values = super().fetch()
        coordinate_type = self.coordinate_type.fetch()
        atoms = self.atoms.fetch().reshape(-1)
        coefficients = self.coefficients.fetch().reshape(-1)
        components = self.components.fetch().reshape(-1)
        midpoint_reference = self.midpoint_reference.fetch().reshape(-1)
        has_midpoint_reference = self.has_midpoint_reference.fetch()
        if coordinate_type == "linear":
            if len(components) != 0:
                raise ValueError(
                    "Slow-growth distance coordinates do not accept components"
                )
            if len(atoms) < 2 or len(atoms) % 2 != 0:
                raise ValueError(
                    "Slow-growth distance requires one or more atom pairs"
                )
            if len(coefficients) == 0:
                coefficients = np.ones(len(atoms) // 2, dtype=float)
            if len(coefficients) != len(atoms) // 2:
                raise ValueError(
                    "Slow-growth distance coefficients must match the number of atom pairs"
                )
            if has_midpoint_reference or len(midpoint_reference) != 0:
                raise ValueError(
                    "Slow-growth midpoint reference is only valid for midpoint coordinates"
                )
        elif coordinate_type == "midpoint":
            if len(components) != 0:
                raise ValueError(
                    "Slow-growth midpoint coordinates do not accept components"
                )
            if len(atoms) != 3 or len(set(atoms.tolist())) != 3:
                raise ValueError(
                    "Slow-growth midpoint distance requires three different atom indices"
                )
            if len(coefficients) != 0:
                raise ValueError(
                    "Slow-growth midpoint distance does not accept coefficients"
                )
            if has_midpoint_reference:
                if len(midpoint_reference) != 3:
                    raise ValueError(
                        "Slow-growth midpoint restart reference must have three components"
                    )
                if not np.all(np.isfinite(midpoint_reference)):
                    raise ValueError(
                        "Slow-growth midpoint restart reference must be finite"
                    )
            elif len(midpoint_reference) != 0:
                raise ValueError(
                    "Slow-growth midpoint reference requires "
                    "has_midpoint_reference=true"
                )
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
            if len(coefficients) == 0:
                coefficients = np.ones(len(atoms), dtype=float)
            if len(coefficients) != len(atoms):
                raise ValueError(
                    "Slow-growth Cartesian coefficients must match the number of atoms"
                )
            if has_midpoint_reference or len(midpoint_reference) != 0:
                raise ValueError(
                    "Slow-growth midpoint reference is only valid for midpoint coordinates"
                )
        values["coordinate_type"] = coordinate_type
        values["atoms"] = atoms
        values["coefficients"] = coefficients
        values["components"] = components
        values["midpoint_reference"] = midpoint_reference
        values["has_midpoint_reference"] = has_midpoint_reference
        return values


class InputSlowGrowthSHAKE(InputDictionary):
    fields = {
        "tolerance": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0e-8,
                "dimension": "length",
                "help": "Maximum absolute length-coordinate residual accepted by SHAKE.",
            },
        ),
        "max_iterations": (
            InputValue,
            {
                "dtype": int,
                "default": 1000,
                "help": "Maximum number of SHAKE geometry iterations per MD step.",
            },
        ),
    }
    dynamic = {}
    default_help = "Numerical settings for the slow-growth SHAKE solver."
    default_label = "SLOWGROWTHSHAKE"

    def store(self, motion):
        self.tolerance.store(motion.constraint.tolerance)
        self.max_iterations.store(motion.constraint.max_iterations)

    def fetch(self):
        return super().fetch()


class InputSlowGrowth(InputDictionary):
    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "nve",
                "options": ["nve", "nvt"],
                "help": "The classical ensemble used during slow growth.",
            },
        ),
        "algorithm": (
            InputAttribute,
            {
                "dtype": str,
                "default": "shake",
                "options": ["shake", "rattle"],
                "help": "Position-only Liu-compatible SHAKE or affine RATTLE.",
            },
        ),
    }

    fields = {
        "timestep": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0,
                "dimension": "time",
                "help": "The complete slow-growth MD timestep.",
            },
        ),
        "thermostat": (
            InputThermo,
            {
                "default": input_default(factory=ipi.engine.thermostats.ThermoSVR),
                "help": "CSVR/SVR thermostat used by slow-growth NVT.",
            },
        ),
        "distance": (
            InputSlowGrowthDistance,
            {"help": InputSlowGrowthDistance.default_help},
        ),
        "shake": (
            InputSlowGrowthSHAKE,
            {"help": InputSlowGrowthSHAKE.default_help},
        ),
        "completed_steps": (
            InputValue,
            {
                "dtype": int,
                "default": 0,
                "help": "Restart state: number of completed slow-growth steps.",
            },
        ),
        "cumulative_work": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "energy",
                "help": "Restart state: trapezoid-integrated free-energy curve.",
            },
        ),
        "previous_cv": (
            InputValue,
            {
                "dtype": float,
                "default": -1.0,
                "dimension": "length",
                "help": "Restart state: previous CV used by trapezoid integration.",
            },
        ),
        "previous_gradient": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "force",
                "help": "Restart state: previous free-energy gradient.",
            },
        ),
        "has_previous": (
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Restart state: whether a previous integration point exists.",
            },
        ),
        "last_lambda": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "force",
                "help": "Restart state: last total constraint multiplier.",
            },
        ),
        "last_shake_lambda": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "force",
                "help": "Restart state: last SHAKE position multiplier.",
            },
        ),
        "last_velocity_lambda": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "force",
                "help": "Restart state: last RATTLE velocity multiplier.",
            },
        ),
        "last_gradient": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "force",
                "help": "Restart state: last free-energy gradient.",
            },
        ),
        "last_error": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "length",
                "help": "Restart state: last position-constraint residual.",
            },
        ),
        "last_iterations": (
            InputValue,
            {
                "dtype": int,
                "default": 0,
                "help": "Restart state: last SHAKE iteration count.",
            },
        ),
        "last_velocity_error": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "velocity",
                "help": "Restart state: last affine RATTLE velocity residual.",
            },
        ),
    }
    dynamic = {}
    default_help = "Length-coordinate classical slow-growth molecular dynamics."
    default_label = "SLOWGROWTH"

    def store(self, motion):
        if motion == {}:
            return
        self.mode.store(motion.enstype)
        self.algorithm.store(motion.algorithm)
        self.timestep.store(motion.dt)
        self.thermostat.store(motion.thermostat)
        self.distance.store(motion)
        self.shake.store(motion)
        self.completed_steps.store(motion.completed_steps)
        self.cumulative_work.store(motion.cumulative_work)
        self.previous_cv.store(motion.previous_cv)
        self.previous_gradient.store(motion.previous_gradient)
        self.has_previous.store(motion.has_previous)
        self.last_lambda.store(motion.last_lambda)
        self.last_shake_lambda.store(motion.last_shake_lambda)
        self.last_velocity_lambda.store(motion.last_velocity_lambda)
        self.last_gradient.store(motion.last_gradient)
        self.last_error.store(motion.last_error)
        self.last_iterations.store(motion.last_iterations)
        self.last_velocity_error.store(motion.last_velocity_error)

    def fetch(self):
        values = super().fetch()
        values["mode"] = self.mode.fetch()
        values["algorithm"] = self.algorithm.fetch()
        values.pop("distance", None)
        values.pop("shake", None)
        values.update(self.distance.fetch())
        values.update(self.shake.fetch())
        return values
