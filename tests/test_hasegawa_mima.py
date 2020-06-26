import numpy as np
import pytest
from turbulence_models.hasegawa_mima import HasegawaMimaFourier


@pytest.mark.parametrize(("kmax", "nx", "ny"), [(10, 7, 9), (10, 3, 5), (10, 19, 21)])
def test_base(kmax, nx, ny):
    wave = HasegawaMimaFourier(kmax, nx, ny)
    phi = np.ones_like(wave.k2) + 1.0j * np.ones_like(wave.k2)
    result = wave.solve((0, 1e-5), phi)
    assert np.isfinite(result["y"]).all()
    wave.to_xarray(result)
