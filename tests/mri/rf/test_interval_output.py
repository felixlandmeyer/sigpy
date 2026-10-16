"""Regression checks for interval-consistent k-space and common timestamps."""
import unittest

import numpy as np
import numpy.testing as npt

from sigpy.mri.rf import min_time_gradient


class TestIntervalOutput(unittest.TestCase):
    def test_spherical_geometry_and_shared_timestamps(self):
        for ntheta, nphi in [(3, 2), (0.5, 3), (5, 1)]:
            for g0 in [0.0, 0.5]:
                with self.subTest(ntheta=ntheta, nphi=nphi, g0=g0):
                    u = np.linspace(0, 1, 801)
                    theta, phi = 2 * np.pi * ntheta * u, 2 * np.pi * nphi * u
                    curve = 0.5 * np.column_stack((
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ))
                    dt, gamma = 0.01, 4.257
                    g, k, slew, t = min_time_gradient(
                        curve, g0=g0, gmax=2, smax=15, dt=dt, gamma=gamma
                    )
                    for values in (g, k, slew):
                        self.assertEqual(values.shape, (len(t), 3))
                        self.assertTrue(np.all(np.isfinite(values)))
                    npt.assert_array_equal(k[0], np.zeros(3))
                    npt.assert_allclose(np.diff(t), dt)
                    # Check absolute shell geometry, including a nonzero start.
                    radial_error = np.abs(np.linalg.norm(k + curve[0], axis=1) - 0.5)
                    self.assertLess(np.max(radial_error), 1e-5)
                    npt.assert_allclose(
                        g, np.gradient(k, t, axis=0, edge_order=2) / gamma,
                        rtol=1e-10, atol=1e-10,
                    )
                    npt.assert_allclose(
                        slew, np.gradient(g, dt, axis=0, edge_order=1)
                    )

    def test_too_few_output_samples(self):
        u = np.linspace(0, 1, 101)
        curve = np.column_stack((u * 0.1, u * 0, u * 0))
        with self.assertRaisesRegex(ValueError, "At least three output samples"):
            min_time_gradient(curve, dt=0.1)


class TestCompiledSweeps(unittest.TestCase):
    def test_compiled_matches_python_sweeps(self):
        from unittest.mock import patch
        from sigpy.mri.rf import trajgrad

        u = np.linspace(0, 1, 401)
        curve = 0.2 * np.column_stack((
            np.sin(6 * np.pi * u) * np.cos(4 * np.pi * u),
            np.sin(6 * np.pi * u) * np.sin(4 * np.pi * u),
            np.cos(6 * np.pi * u),
        ))
        compiled = trajgrad._speed_sweeps
        for g0, gfin in [(0.0, None), (0.2, 0.0), (None, None)]:
            with self.subTest(g0=g0, gfin=gfin):
                params = dict(g0=g0, gfin=gfin, dt=0.004)
                actual = trajgrad.min_time_gradient(curve, **params)
                with patch.object(trajgrad, "_speed_sweeps", compiled.py_func):
                    expected = trajgrad.min_time_gradient(curve, **params)
                for a, b in zip(actual, expected):
                    npt.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        self.assertTrue(compiled.nopython_signatures)
