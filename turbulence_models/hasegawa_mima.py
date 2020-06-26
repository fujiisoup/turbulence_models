import numpy as np
from scipy import integrate


class HasegawaMimaFourier:
    """
    Fourier base hasegawa-mima equation
    """
    def __init__(self, kmax, nx, ny):
        assert nx % 2 == 1
        assert ny % 2 == 1
        self.kx = np.linspace(-kmax, kmax, nx)
        self.ky = np.linspace(-kmax, kmax, ny)
        self._setup()

    @property
    def shape(self):
        return self.k2.shape

    def _setup(self):
        # wavenumber vector [nx, ny]
        self.k = np.stack(np.meshgrid(self.ky, self.kx), axis=-1)
        # square amplitude of the wavenumber
        self.k2 = np.sum(self.k ** 2, axis=-1)
        nx, ny = len(self.kx), len(self.ky)
        # matrix that represents three wave interactions
        self.mat = np.zeros((nx, ny, nx * ny))
        self.index_px = np.zeros((nx, ny, nx * ny), dtype=int)
        self.index_py = np.zeros((nx, ny, nx * ny), dtype=int)
        self.index_qx = np.zeros((nx, ny, nx * ny), dtype=int)
        self.index_qy = np.zeros((nx, ny, nx * ny), dtype=int)

        half_x = (nx - 1) // 2
        half_y = (ny - 1) // 2

        for ix, kx in enumerate(self.kx):
            for iy, ky in enumerate(self.ky):
                jx, jy = np.broadcast_arrays(
                    np.arange(len(self.kx))[:, np.newaxis], np.arange(len(self.ky))
                )
                jx, jy = jx.ravel(), jy.ravel()
                j = jx * ny + jy
                self.index_px[ix, iy, j] = jx
                self.index_py[ix, iy, j] = jy
                lx = 3 * half_x - ix - jx
                ly = 3 * half_y - iy - jy
                l = lx * ny + ly

                valid_x = (0 <= lx) * (lx < nx)
                valid_y = (0 <= ly) * (ly < ny)
                lx = np.where(valid_x, lx, 0)
                ly = np.where(valid_y, ly, 0)
                self.index_qx[ix, iy, j] = np.where(valid_x, lx, 0)
                self.index_qy[ix, iy, j] = np.where(valid_y, ly, 0)
                qx, qy, q2 = self.kx[lx], self.ky[ly], self.k2[lx, ly]
                self.mat[ix, iy, j] = np.where(
                    valid_x * valid_y, (self.kx[jx] * qy - self.ky[jy] * qx) * q2, 0.0
                )
                # sanity check
                kxsum = np.where(valid_x * valid_y, kx + self.kx[jx] + qx, 0)
                assert np.allclose(kxsum, 0)
                kysum = np.where(valid_x * valid_y, ky + self.ky[jy] + qy, 0)
                assert np.allclose(kysum, 0)

    def solve(self, t_span, phiinit, **kwargs):
        """
        Solve HM equation with initial phiinit
        """

        def dfdt(t, phi):
            phi = np.reshape(phi, self.shape)
            phic = np.conj(phi)
            dphi_dt = -1.0j * self.ky * phi + np.sum(
                self.mat
                * phic[self.index_px, self.index_py]
                * phic[self.index_qx, self.index_qy],
                axis=-1,
            )
            return (dphi_dt / (1 + self.k2)).ravel()

        result = integrate.solve_ivp(dfdt, t_span, phiinit.ravel(), **kwargs)
        result["y"] = result["y"].reshape(self.shape + (len(result["t"]),))
        return result

    def to_xarray(self, result):
        """
        Convert the result into xarray format
        """
        import xarray as xr

        return xr.Dataset(
            {
                "phi": (("kx", "ky", "time"), result["y"]),
                "status": result["status"],
                "message": result["message"],
                "success": result["success"],
            },
            coords={"kx": self.kx, "ky": self.ky, "time": result["t"]},
            attrs={},
        )
