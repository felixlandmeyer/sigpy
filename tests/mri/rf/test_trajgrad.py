import math
import unittest

import numpy as np
import numpy.testing as npt

import sigpy.mri.rf as rf

if __name__ == "__main__":
    unittest.main()


class TestTrajGrad(unittest.TestCase):
    def test_min_gradient(self):
        t = np.linspace(0, 1, 1000)
        kx = np.sin(2.0 * math.pi * t)
        ky = np.cos(2.0 * math.pi * t)
        kz = t
        k = np.stack((kx, ky, kz), axis=-1)

        (g, k, s, t) = rf.min_time_gradient(
            k, 0.0, 0.0, gmax=4, smax=15, dt=4e-3, gamma=4.257
        )

        npt.assert_almost_equal(np.max(t), 0.916, decimal=4)

    def test_trap_grad(self):
        dt = 4e-6  # s
        area = 200 * dt
        dgdt = 18000  # g/cm/s
        gmax = 2  # g/cm

        trap, _ = rf.trap_grad(area, gmax, dgdt, dt)

        npt.assert_almost_equal(area, np.sum(trap) * dt, decimal=3)
        npt.assert_almost_equal(gmax, np.max(trap), decimal=1)

    def test_min_trap_grad(self):
        dt = 4e-6  # s
        area = 200 * dt
        dgdt = 18000  # g/cm/s
        gmax = 2  # g/cm

        trap, _ = rf.min_trap_grad(area, gmax, dgdt, dt)

        npt.assert_almost_equal(area, np.sum(trap) * dt, decimal=3)
        npt.assert_almost_equal(gmax, np.max(trap), decimal=1)


class TestSpeedIntegration(unittest.TestCase):
    def test_straight_line_runge_kutta_step(self):
        from sigpy.mri.rf.trajgrad import runge_kutta

        initial_speed = 1.0
        arc_step = 0.001
        gamma = 3.1
        slew_limit = 12.0
        increment = runge_kutta(arc_step, initial_speed, np.zeros(4), slew_limit, gamma)
        expected = np.sqrt(initial_speed**2 + 2 * gamma * slew_limit * arc_step)
        npt.assert_allclose(initial_speed + increment, expected, rtol=1e-8)

    def test_speed_sweeps_use_supplied_gamma(self):
        from sigpy.mri.rf.trajgrad import _speed_sweeps

        arc = np.linspace(0, 0.02, 21)
        gamma = 3.1
        initial_speed = 1.0
        slew_limit = 12.0
        forward, backward = _speed_sweeps(
            arc, np.zeros(arc.size + 3), np.full(arc.size, 100.0),
            arc[1] - arc[0], initial_speed, 0.0, 0.0, 100.0, slew_limit, gamma)
        npt.assert_allclose(forward, np.sqrt(initial_speed**2 + 2 * gamma * slew_limit * arc), rtol=1e-7)
        npt.assert_allclose(backward, np.sqrt(initial_speed**2 + 2 * gamma * slew_limit * (arc[-1] - arc)), rtol=1e-7)
