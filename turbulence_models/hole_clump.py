"""
Formation of Phase Space Holes and Clumps M. K. Lilley*
PRL 112, 155002 (2014)
"""
import numpy as np
from scipy import integrate


eps0 = 8.85418782e-12  # vacuum permittivity m-3 kg-1 s4 A2
e = 1.60217662e-19  # elementary charge in [C]
me = 9.10938356e-31  # electron mass in [kg]


class HoleClump:
    def __init__(self, xmax, vmax, nx, nv, nu, ne):
        self.nx = nx
        self.nv = nv
        self.x = np.linspace(-xmax, xmax, nx)
        self.dx = np.mean(np.diff(self.x))
        self.v = np.linspace(-vmax, vmax, nv)
        self.dv = np.mean(np.diff(self.v))
        self.eps0 = eps0
        self.nu = nu
        self.ne = ne
        self.omega_p = np.sqrt(ne * e**2 / (me * eps0))

    def _encode(self, f, include_time=False):
        """
        Convert one dimensional vector to F, V, E
        """
        if include_time:
            F = f[:self.nx * self.nv, :].reshape(self.nx, self.nv, -1)
        else: 
            F = f[:self.nx * self.nv].reshape(self.nx, self.nv)
        V = f[self.nx * self.nv]
        E = f[self.nx * self.nv + 1]
        return F, V, E

    def solve(self, t_span, finit, vinit, einit, **kwargs):
        """
        Solve the differential equation
        
        finit: nx * nv
        vinit, einit: scalars
        """
        assert finit.shape == (self.nx, self.nv)
        coef = e**2 / me / eps0

        def dfdt(t, f):
            f, V, E = self._encode(f)
            # periodic boundary along x and zero boundary at v
            f_pad = np.pad(np.pad(f, [(1, 1), (0, 0)], 'wrap'), 
                           [(0, 0), (1, 1)], 'constant', constant_values=0)
            dfdx = (f_pad[2:, 1:-1] - f_pad[:-2, 1:-1]) / (2 * self.dx)
            dfdv = (f_pad[1:-1, 2:] - f_pad[1:-1, :-2]) / (2 * self.dv)
            dfdt = -self.v * dfdx + E * dfdv

            dvdt = -E - self.nu * V
            # TODO is E a scalar? Should we integrate over x?
            dEdt = coef * (self.ne * V + np.sum(f * self.v) * self.dv)
            
            return np.concatenate([dfdt.ravel(), [dvdt], [dEdt]])

        finit = np.concatenate([finit.ravel(), [vinit], [einit]])
        result = integrate.solve_ivp(dfdt, t_span, finit, **kwargs)
        
        result["f"], result["V"], result['E'] = self._encode(
            result["y"], include_time=True)
        return result

    def to_xarray(self, result):
        """
        Convert the result into xarray format
        """
        import xarray as xr

        return xr.Dataset(
            {
                "F": (("x", "v", "time"), result["f"]),
                "V": ("time", result["V"]),
                "E": ("time", result["E"]),
                "status": result["status"],
                "message": result["message"],
                "success": result["success"],
            },
            coords={"x": self.x, "v": self.v, "time": result["t"]},
            attrs={},
        )
