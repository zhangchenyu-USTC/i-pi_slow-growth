"""Unit tests for length-coordinate slow-growth dynamics."""

from types import SimpleNamespace

import numpy as np
import pytest

from ipi.engine.beads import Beads
from ipi.engine.cell import Cell
from ipi.engine.motion.slow_growth import (
    SlowGrowthDistanceConstraint,
    SlowGrowthDynamics,
    SlowGrowthIntegrator,
    SlowGrowthSHAKESolver,
)
from ipi.inputs.motion.motion import InputMotion
from ipi.utils.depend import dstrip
from ipi.utils.io.inputs.io_xml import xml_parse_string


class _TwoAtomBeads:
    nbeads = 1
    natoms = 2


def _constraint(
    start=0.4,
    increment=0.1,
    tolerance=1.0e-12,
    atoms=(0, 1),
    coefficients=None,
    components=None,
    coordinate_type="linear",
):
    constraint = SlowGrowthDistanceConstraint(
        atoms=atoms,
        start=start,
        increment=increment,
        tolerance=tolerance,
        max_iterations=100,
        coefficients=coefficients,
        components=components,
        coordinate_type=coordinate_type,
    )
    constraint.bind(
        SimpleNamespace(nbeads=1, natoms=max(atoms) + 1),
        Cell(np.eye(3) * 10.0),
    )
    return constraint


def test_minimum_image_distance_and_gradient():
    constraint = _constraint()
    q = np.array([9.8, 0.0, 0.0, 0.2, 0.0, 0.0])

    np.testing.assert_allclose(constraint.displacement(q), [-0.4, 0.0, 0.0])
    assert constraint.value(q) == pytest.approx(0.4)
    np.testing.assert_allclose(
        constraint.gradient(q), [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    assert constraint.metric_tensor(q, np.ones(6)) == pytest.approx(2.0)


def test_cartesian_component_difference_uses_raw_coordinates():
    constraint = _constraint(
        start=8.0,
        atoms=(0, 1),
        components=(2, 2),
        coefficients=(1.0, -1.0),
        coordinate_type="cartesian",
    )
    q = np.array([0.0, 0.0, 9.0, 0.0, 0.0, 1.0])
    m3 = np.repeat([2.0, 4.0], 3)

    # Cartesian components are continuous raw i-PI coordinates, not MIC values.
    assert constraint.value(q) == pytest.approx(8.0)
    np.testing.assert_allclose(
        constraint.gradient(q), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    assert constraint.metric_tensor(q, m3) == pytest.approx(0.75)
    assert constraint.metric_gradient_correction(q, m3) == 0.0


def test_cartesian_component_combination_accumulates_repeated_atoms():
    constraint = _constraint(
        start=5.0,
        atoms=(0, 0, 1),
        components=(0, 2, 2),
        coefficients=(2.0, -1.0, 1.0),
        coordinate_type="cartesian",
    )
    q = np.array([2.0, 0.0, 1.0, 0.0, 0.0, 2.0])

    assert constraint.value(q) == pytest.approx(5.0)
    np.testing.assert_allclose(
        constraint.gradient(q), [[2.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
    )


def test_cartesian_component_shake_and_rattle_projection():
    constraint = _constraint(
        start=1.0,
        atoms=(0, 1),
        components=(2, 2),
        coefficients=(1.0, -1.0),
        coordinate_type="cartesian",
    )
    solver = SlowGrowthSHAKESolver(constraint)
    q = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    m3 = np.ones(6)

    q_new, p_new, _, iterations, error = solver.project_position(
        q_old=q,
        p_trial=np.zeros(6),
        m3=m3,
        dt=1.0,
        target=1.2,
    )
    assert constraint.value(q_new) == pytest.approx(1.2, abs=1.0e-12)
    assert iterations == 1
    assert error == pytest.approx(0.0, abs=1.0e-12)

    p_projected, _, velocity_error = solver.project_velocity(
        q=q_new,
        p=p_new,
        m3=m3,
        target_velocity=0.03,
    )
    gradient = constraint.gradient(q_new)
    velocity = sum(
        np.dot(gradient[local], p_projected[3 * atom : 3 * atom + 3])
        for local, atom in enumerate(constraint.atoms)
    )
    assert velocity == pytest.approx(0.03, abs=1.0e-14)
    assert velocity_error == pytest.approx(0.0, abs=1.0e-14)


def test_midpoint_distance_value_gradient_metric_and_projection():
    constraint = _constraint(
        start=0.6,
        atoms=(0, 1, 2),
        coordinate_type="midpoint",
    )
    q = np.array([9.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.6, 0.0, 0.0])

    np.testing.assert_allclose(
        constraint.midpoint_displacement(q), [0.6, 0.0, 0.0]
    )
    assert constraint.value(q) == pytest.approx(0.6)
    np.testing.assert_allclose(
        constraint.gradient(q),
        [[-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )
    assert constraint.metric_tensor(q, np.ones(9)) == pytest.approx(1.5)
    assert constraint.metric_gradient_correction(q, np.ones(9)) == 0.0

    q_new, _, _, _, error = SlowGrowthSHAKESolver(constraint).project_position(
        q_old=q,
        p_trial=np.zeros(9),
        m3=np.ones(9),
        dt=1.0,
        target=0.7,
    )
    assert constraint.value(q_new) == pytest.approx(0.7, abs=1.0e-12)
    assert error == pytest.approx(0.0, abs=1.0e-12)


def test_midpoint_image_is_continuous_across_half_cell_and_back():
    constraint = _constraint(
        start=4.9,
        atoms=(0, 1, 2),
        coordinate_type="midpoint",
    )
    q_before = np.array([0.0, 0.0, 0.0, 0.0, 4.8, 0.0, 0.0, 5.0, 0.0])
    q_after = np.array([0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, -4.8, 0.0])

    constraint.initialize_midpoint_reference(q_before)
    np.testing.assert_allclose(constraint.midpoint_reference, [0.0, 0.49, 0.0])
    assert constraint.value(q_before) == pytest.approx(4.9)
    assert constraint.value(q_after) == pytest.approx(5.1)
    np.testing.assert_allclose(
        constraint.gradient(q_after),
        [[0.0, -1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]],
    )

    # Trial evaluations do not alter the image used by the SHAKE iteration.
    np.testing.assert_allclose(constraint.midpoint_reference, [0.0, 0.49, 0.0])
    constraint.commit_midpoint_reference(q_after)
    np.testing.assert_allclose(constraint.midpoint_reference, [0.0, 0.51, 0.0])

    # Re-crossing the same boundary must remain on the positive branch.
    assert constraint.value(q_before) == pytest.approx(4.9)
    constraint.commit_midpoint_reference(q_before)
    np.testing.assert_allclose(constraint.midpoint_reference, [0.0, 0.49, 0.0])


def test_midpoint_shake_crosses_half_cell_in_both_directions():
    constraint = _constraint(
        start=4.99,
        atoms=(0, 1, 2),
        coordinate_type="midpoint",
    )
    solver = SlowGrowthSHAKESolver(constraint)
    q_start = np.array(
        [0.0, 0.0, 0.0, 0.0, 4.89, 0.0, 0.0, 5.09, 0.0]
    )

    q_forward, _, _, _, forward_error = solver.project_position(
        q_old=q_start,
        p_trial=np.zeros(9),
        m3=np.ones(9),
        dt=1.0,
        target=5.01,
    )
    assert forward_error == pytest.approx(0.0, abs=1.0e-12)
    assert constraint.value(q_forward) == pytest.approx(5.01, abs=1.0e-12)
    assert np.max(np.abs(q_forward - q_start)) < 0.02
    constraint.commit_midpoint_reference(q_forward)
    np.testing.assert_allclose(
        constraint.midpoint_reference, [0.0, 0.501, 0.0], atol=1.0e-14
    )

    q_backward, _, _, _, backward_error = solver.project_position(
        q_old=q_forward,
        p_trial=np.zeros(9),
        m3=np.ones(9),
        dt=1.0,
        target=4.99,
    )
    assert backward_error == pytest.approx(0.0, abs=1.0e-12)
    assert constraint.value(q_backward) == pytest.approx(4.99, abs=1.0e-12)
    assert np.max(np.abs(q_backward - q_forward)) < 0.02
    constraint.commit_midpoint_reference(q_backward)
    np.testing.assert_allclose(
        constraint.midpoint_reference, [0.0, 0.499, 0.0], atol=1.0e-14
    )


def test_shake_projects_distance_without_force_calls():
    constraint = _constraint()
    solver = SlowGrowthSHAKESolver(constraint)
    q = np.array([9.8, 0.0, 0.0, 0.2, 0.0, 0.0])

    q_new, p_new, multiplier, iterations, error = solver.project_position(
        q_old=q,
        p_trial=np.zeros(6),
        m3=np.ones(6),
        dt=1.0,
        target=0.5,
    )

    assert constraint.value(q_new) == pytest.approx(0.5, abs=1.0e-12)
    assert error == pytest.approx(0.0, abs=1.0e-12)
    assert iterations == 1
    assert multiplier == pytest.approx(-0.05)
    np.testing.assert_allclose(p_new, [-0.05, 0.0, 0.0, 0.05, 0.0, 0.0])


def test_distance_difference_value_gradient_metric_and_projection():
    constraint = _constraint(
        start=-1.0,
        atoms=(0, 1, 1, 2),
        coefficients=(1.0, -1.0),
    )
    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0])

    assert constraint.value(q) == pytest.approx(-1.0)
    np.testing.assert_allclose(
        constraint.gradient(q),
        [[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    assert constraint.metric_tensor(q, np.ones(9)) == pytest.approx(6.0)

    q_new, _, _, _, error = SlowGrowthSHAKESolver(constraint).project_position(
        q_old=q,
        p_trial=np.zeros(9),
        m3=np.ones(9),
        dt=1.0,
        target=-0.9,
    )
    assert constraint.value(q_new) == pytest.approx(-0.9, abs=1.0e-12)
    assert error == pytest.approx(0.0, abs=1.0e-12)


def test_rattle_enforces_affine_velocity():
    constraint = _constraint(start=0.5)
    solver = SlowGrowthSHAKESolver(constraint)
    q = np.array([9.75, 0.0, 0.0, 0.25, 0.0, 0.0])
    p = np.array([0.4, 0.2, 0.0, -0.1, 0.2, 0.0])

    p_new, _, error = solver.project_velocity(
        q=q, p=p, m3=np.ones(6), target_velocity=-0.02
    )
    gradient = constraint.gradient(q)
    actual_velocity = np.dot(gradient[0], p_new[:3]) + np.dot(gradient[1], p_new[3:])

    assert error == pytest.approx(0.0, abs=1.0e-14)
    assert actual_velocity == pytest.approx(-0.02, abs=1.0e-14)


class _CountingPairForce:
    def __init__(self, beads, force_constant, equilibrium):
        self.beads = beads
        self.force_constant = force_constant
        self.equilibrium = equilibrium
        self.calls = 0
        self.mts_forces = [self]

    @property
    def f(self):
        self.calls += 1
        q = np.asarray(dstrip(self.beads.q[0])).reshape(2, 3)
        rij = q[0] - q[1]
        distance = np.linalg.norm(rij)
        unit = rij / distance
        derivative = self.force_constant * (distance - self.equilibrium)
        force = np.zeros((1, 6))
        force[0, :3] = -derivative * unit
        force[0, 3:] = derivative * unit
        return force


def test_rattle_harmonic_gradient_and_fake_force_count():
    beads = Beads(natoms=2, nbeads=1)
    beads.m[:] = [1.0, 1.0]
    beads.q[0] = [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0]
    beads.p[0] = 0.0
    cell = Cell(np.eye(3) * 10.0)

    motion = SlowGrowthDynamics(
        timestep=0.1,
        mode="nve",
        algorithm="rattle",
        fixcom=False,
        atoms=[0, 1],
        start=1.0,
        increment=0.0,
        tolerance=1.0e-12,
        max_iterations=100,
    )
    motion.constraint.bind(beads, cell)
    motion.beads = beads

    force = _CountingPairForce(beads, force_constant=2.0, equilibrium=0.5)
    integrator = SlowGrowthIntegrator()
    integrator.motion = motion
    integrator.beads = beads
    integrator.forces = force
    integrator.ensemble = SimpleNamespace(has_bias=False)
    integrator.enstype = "nve"
    integrator._dt = motion._dt
    integrator.pconstraints = lambda: None

    integrator.step(step=0)

    assert force.calls == 2
    assert motion.last_cv == pytest.approx(1.0, abs=1.0e-12)
    assert motion.last_gradient == pytest.approx(1.0, abs=2.0e-12)
    assert motion.last_velocity_error == pytest.approx(0.0, abs=1.0e-14)


def test_shake_harmonic_reaches_liu_steady_multiplier():
    beads = Beads(natoms=2, nbeads=1)
    beads.m[:] = [1.0, 1.0]
    beads.q[0] = [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0]
    beads.p[0] = 0.0
    cell = Cell(np.eye(3) * 10.0)

    motion = SlowGrowthDynamics(
        timestep=0.1,
        mode="nve",
        algorithm="shake",
        fixcom=False,
        atoms=[0, 1],
        start=1.0,
        increment=0.0,
        tolerance=1.0e-12,
        max_iterations=100,
    )
    motion.constraint.bind(beads, cell)
    motion.beads = beads

    force = _CountingPairForce(beads, force_constant=2.0, equilibrium=0.5)
    integrator = SlowGrowthIntegrator()
    integrator.motion = motion
    integrator.beads = beads
    integrator.forces = force
    integrator.ensemble = SimpleNamespace(has_bias=False)
    integrator.enstype = "nve"
    integrator._dt = motion._dt
    integrator.pconstraints = lambda: None

    integrator.step(step=0)
    first_gradient = motion.last_gradient
    integrator.step(step=1)

    # Liu's position-only algorithm has a one-step half-kick transient.
    assert first_gradient == pytest.approx(0.5, abs=2.0e-12)
    assert motion.last_gradient == pytest.approx(1.0, abs=2.0e-12)
    assert force.calls == 4


@pytest.mark.parametrize("increment", [0.1, -0.1])
def test_first_step_uses_start_then_increment(increment):
    beads = Beads(natoms=2, nbeads=1)
    beads.m[:] = [1.0, 1.0]
    beads.q[0] = [-0.5, 0.0, 0.0, 0.5, 0.0, 0.0]
    beads.p[0] = 0.0
    cell = Cell(np.eye(3) * 10.0)

    motion = SlowGrowthDynamics(
        timestep=0.1,
        mode="nve",
        algorithm="rattle",
        fixcom=False,
        atoms=[0, 1],
        start=1.0,
        increment=increment,
        tolerance=1.0e-12,
        max_iterations=100,
    )
    motion.constraint.bind(beads, cell)
    motion.beads = beads

    force = _CountingPairForce(beads, force_constant=0.0, equilibrium=1.0)
    integrator = SlowGrowthIntegrator()
    integrator.motion = motion
    integrator.beads = beads
    integrator.forces = force
    integrator.ensemble = SimpleNamespace(has_bias=False)
    integrator.enstype = "nve"
    integrator._dt = motion._dt
    integrator.pconstraints = lambda: None

    integrator.step(step=0)
    assert motion.last_target == pytest.approx(1.0)
    assert motion.last_cv == pytest.approx(1.0, abs=1.0e-12)
    assert motion.current_target == pytest.approx(1.0 + increment)

    integrator.step(step=1)
    assert motion.last_target == pytest.approx(1.0 + increment)
    assert motion.last_cv == pytest.approx(1.0 + increment, abs=1.0e-12)
    assert motion.current_target == pytest.approx(1.0 + 2.0 * increment)
    assert motion.completed_steps == 2


def _parse_motion(xml):
    document = xml_parse_string("<root>" + xml + "</root>").fields[0][1]
    node = document.fields[0][1]
    parsed = InputMotion()
    parsed.parse(node)
    return parsed.fetch()


def test_slow_growth_xml_and_restart_roundtrip():
    motion = _parse_motion(
        """
        <motion mode="slow_growth">
          <fixcom>false</fixcom>
          <slow_growth mode="nvt" algorithm="rattle">
            <timestep units="femtosecond">1.0</timestep>
            <thermostat mode="svr">
              <tau units="femtosecond">40.0</tau>
            </thermostat>
            <distance>
              <atoms>[0, 1]</atoms>
              <start units="angstrom">1.0</start>
              <increment units="angstrom">0.001</increment>
            </distance>
            <shake>
              <tolerance units="angstrom">1.0e-8</tolerance>
              <max_iterations>321</max_iterations>
            </shake>
          </slow_growth>
        </motion>
        """
    )

    motion.current_target += motion.increment
    motion.last_target = motion.start
    motion.completed_steps = 1
    motion.cumulative_work = 0.25
    motion.previous_cv = motion.start
    motion.previous_gradient = -0.4
    motion.has_previous = True
    motion.last_lambda = 0.4
    motion.last_shake_lambda = 0.3
    motion.last_velocity_lambda = 0.1
    motion.last_gradient = -0.4
    motion.last_error = 1.0e-12
    motion.last_iterations = 4
    motion.last_velocity_error = -2.0e-12

    stored = InputMotion()
    stored.store(motion)
    restored = _parse_motion(stored.write("motion"))

    assert restored.algorithm == "rattle"
    assert restored.enstype == "nvt"
    assert restored.constraint.atoms.tolist() == [0, 1]
    assert restored.constraint.max_iterations == 321
    assert restored.current_target == pytest.approx(motion.current_target)
    assert restored.last_target == pytest.approx(motion.last_target)
    assert restored.completed_steps == 1
    assert restored.cumulative_work == pytest.approx(0.25)
    assert restored.previous_cv == pytest.approx(motion.previous_cv)
    assert restored.previous_gradient == pytest.approx(-0.4)
    assert restored.has_previous is True
    assert restored.last_lambda == pytest.approx(0.4)
    assert restored.last_shake_lambda == pytest.approx(0.3)
    assert restored.last_velocity_lambda == pytest.approx(0.1)
    assert restored.last_gradient == pytest.approx(-0.4)
    assert restored.last_error == pytest.approx(1.0e-12)
    assert restored.last_iterations == 4
    assert restored.last_velocity_error == pytest.approx(-2.0e-12)


def test_distance_difference_xml_and_fixed_atoms():
    motion = _parse_motion(
        """
        <motion mode="slow_growth">
          <fixatoms>[3]</fixatoms>
          <slow_growth mode="nvt" algorithm="shake">
            <timestep units="femtosecond">1.0</timestep>
            <thermostat mode="svr">
              <tau units="femtosecond">40.0</tau>
            </thermostat>
            <distance>
              <atoms>[0, 1, 1, 2]</atoms>
              <coefficients>[1.0, -1.0]</coefficients>
              <start units="angstrom">-1.0</start>
              <increment units="angstrom">0.001</increment>
            </distance>
            <shake>
              <tolerance units="angstrom">1.0e-8</tolerance>
              <max_iterations>1000</max_iterations>
            </shake>
          </slow_growth>
        </motion>
        """
    )

    np.testing.assert_array_equal(motion.constraint.atom_pairs, [[0, 1], [1, 2]])
    np.testing.assert_allclose(motion.constraint.coefficients, [1.0, -1.0])
    np.testing.assert_array_equal(motion.fixatoms_dof, [9, 10, 11])
    assert motion.start < 0.0


def test_cartesian_component_xml_and_restart_roundtrip():
    motion = _parse_motion(
        """
        <motion mode="slow_growth">
          <fixcom>false</fixcom>
          <slow_growth mode="nvt" algorithm="rattle">
            <timestep units="femtosecond">1.0</timestep>
            <thermostat mode="svr">
              <tau units="femtosecond">40.0</tau>
            </thermostat>
            <distance coordinate_type="cartesian">
              <atoms>[0, 1]</atoms>
              <components>[2, 2]</components>
              <coefficients>[1.0, -1.0]</coefficients>
              <start units="angstrom">1.0</start>
              <increment units="angstrom">-0.001</increment>
            </distance>
            <shake>
              <tolerance units="angstrom">1.0e-8</tolerance>
              <max_iterations>1000</max_iterations>
            </shake>
          </slow_growth>
        </motion>
        """
    )

    assert motion.constraint.coordinate_type == "cartesian"
    np.testing.assert_array_equal(motion.constraint.input_atoms, [0, 1])
    np.testing.assert_array_equal(motion.constraint.components, [2, 2])
    np.testing.assert_allclose(motion.constraint.coefficients, [1.0, -1.0])

    stored = InputMotion()
    stored.store(motion)
    restored = _parse_motion(stored.write("motion"))
    assert restored.constraint.coordinate_type == "cartesian"
    np.testing.assert_array_equal(restored.constraint.input_atoms, [0, 1])
    np.testing.assert_array_equal(restored.constraint.components, [2, 2])
    np.testing.assert_allclose(restored.constraint.coefficients, [1.0, -1.0])


def test_midpoint_distance_xml_and_restart_roundtrip():
    motion = _parse_motion(
        """
        <motion mode="slow_growth">
          <fixcom>false</fixcom>
          <slow_growth mode="nvt" algorithm="shake">
            <timestep units="femtosecond">1.0</timestep>
            <thermostat mode="svr">
              <tau units="femtosecond">40.0</tau>
            </thermostat>
            <distance coordinate_type="midpoint">
              <atoms>[0, 1, 2]</atoms>
              <start units="angstrom">1.0</start>
              <increment units="angstrom">0.001</increment>
            </distance>
            <shake>
              <tolerance units="angstrom">1.0e-8</tolerance>
              <max_iterations>1000</max_iterations>
            </shake>
          </slow_growth>
        </motion>
        """
    )

    assert motion.constraint.coordinate_type == "midpoint"
    np.testing.assert_array_equal(motion.constraint.input_atoms, [0, 1, 2])
    beads = SimpleNamespace(nbeads=1, natoms=3)
    cell = Cell(np.eye(3) * 10.0)
    q_before = np.array([0.0, 0.0, 0.0, 0.0, 4.8, 0.0, 0.0, 5.0, 0.0])
    q_after = np.array([0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, -4.8, 0.0])
    motion.constraint.bind(beads, cell)
    motion.constraint.initialize_midpoint_reference(q_before)
    motion.constraint.commit_midpoint_reference(q_after)

    stored = InputMotion()
    stored.store(motion)
    restored = _parse_motion(stored.write("motion"))
    assert restored.constraint.coordinate_type == "midpoint"
    np.testing.assert_array_equal(restored.constraint.input_atoms, [0, 1, 2])
    assert restored.constraint.has_midpoint_reference is True
    np.testing.assert_allclose(
        restored.constraint.midpoint_reference, [0.0, 0.51, 0.0]
    )

    # The restart state selects the same unwrapped branch immediately.
    restored.constraint.bind(beads, cell)
    assert restored.constraint.value(q_after) == pytest.approx(5.1)


@pytest.mark.parametrize(
    "atoms,start,tolerance,max_iterations,match",
    [
        ([0, 0], 1.0, 1.0e-8, 100, "different"),
        ([-1, 1], 1.0, 1.0e-8, 100, "non-negative"),
        ([0, 1], 0.0, 1.0e-8, 100, "positive"),
        ([0, 1], 1.0, 0.0, 100, "tolerance"),
        ([0, 1], 1.0, 1.0e-8, 0, "iteration"),
    ],
)
def test_invalid_constraint_inputs_fail(atoms, start, tolerance, max_iterations, match):
    with pytest.raises(ValueError, match=match):
        SlowGrowthDistanceConstraint(
            atoms=atoms,
            start=start,
            increment=0.0,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )


def test_nonfinite_increment_fails():
    with pytest.raises(ValueError, match="increment"):
        SlowGrowthDistanceConstraint(
            atoms=[0, 1],
            start=1.0,
            increment=np.nan,
            tolerance=1.0e-8,
            max_iterations=100,
        )


def test_invalid_distance_combination_fails():
    with pytest.raises(ValueError, match="pairs"):
        SlowGrowthDistanceConstraint(
            atoms=[0, 1, 2],
            coefficients=[1.0],
            start=1.0,
            increment=0.0,
            tolerance=1.0e-8,
            max_iterations=100,
        )
    with pytest.raises(ValueError, match="coefficients"):
        SlowGrowthDistanceConstraint(
            atoms=[0, 1, 1, 2],
            coefficients=[1.0],
            start=1.0,
            increment=0.0,
            tolerance=1.0e-8,
            max_iterations=100,
        )


@pytest.mark.parametrize(
    "atoms,components,coefficients,match",
    [
        ([], [], [], "at least one atom"),
        ([0, 1], [2], [1.0, -1.0], "components"),
        ([0], [3], [1.0], r"0 \(x\).*2 \(z\)"),
        ([0, 1], [2, 2], [1.0], "coefficients"),
        ([0, 0], [2, 2], [1.0, -1.0], "zero-gradient"),
    ],
)
def test_invalid_cartesian_coordinate_fails(
    atoms, components, coefficients, match
):
    with pytest.raises(ValueError, match=match):
        SlowGrowthDistanceConstraint(
            atoms=atoms,
            components=components,
            coefficients=coefficients,
            coordinate_type="cartesian",
            start=0.0,
            increment=0.0,
            tolerance=1.0e-8,
            max_iterations=100,
        )


def test_components_rejected_for_distance_coordinate():
    with pytest.raises(ValueError, match="do not accept components"):
        SlowGrowthDistanceConstraint(
            atoms=[0, 1],
            components=[2, 2],
            coordinate_type="linear",
            start=1.0,
            increment=0.0,
            tolerance=1.0e-8,
            max_iterations=100,
        )
