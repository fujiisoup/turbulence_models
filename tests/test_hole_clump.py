import numpy as np
import pytest
from turbulence_models.hole_clump import HoleClump


@pytest.mark.parametrize(("xmax", "vmax", "nx", "nv", "nu", "ne"), [
    (10.0, 11.0, 10, 11, 1.0, 1.0),
])
def test_hole_clump(xmax, vmax, nx, nv, nu, ne):
    model = HoleClump(xmax, vmax, nx, nv, nu, ne)
    rng = np.random.RandomState(0)
    finit = rng.randn(nx, nv)
    vinit = rng.randn(nx)
    einit = rng.randn(nx)
    result = model.solve((0, 1e-5), finit, vinit, einit)
    assert np.isfinite(result["y"]).all()
    model.to_xarray(result)
