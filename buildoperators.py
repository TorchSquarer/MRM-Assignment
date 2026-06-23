from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import scipy.sparse as sp

from config import ModelConfig

for candidate in (Path.cwd(), Path.cwd().parent):
    pymrm_src = candidate / "pymrm" / "src"
    if pymrm_src.exists() and str(pymrm_src) not in sys.path:
        sys.path.insert(0, str(pymrm_src))

from pymrm import (
    construct_coefficient_matrix,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    non_uniform_grid,
)

# Imports spatial grids and finite-volume transport operators
class TransportOperators:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

        # Shape conventions used by Pymrm
        self.gas_shape        = (cfg.n_z, cfg.n_c)
        self.boundary_shape   = (cfg.n_z, 1, cfg.n_c)
        self.particle_shape   = (cfg.n_z, cfg.n_r_ret, cfg.n_c)

        self._build_grids()
        self._build_operators()
        self._build_gas_transport_operators()
        self._build_permeate_transport_operators()

    # Construc axial and radial grids
    def _build_grids(self) -> None:
        cfg = self.cfg
        
        # Axial faces and cell centers
        self.z_f = np.linspace(0.0, cfg.length, cfg.n_z + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])

    # construc gas, particle, and permeate transport matrices
    def _build_operators(self) -> None:
        cfg = self.cfg
        
        self._build_gas_transport_operators()
        self._build_permeate_transport_operators()

        #self._build_heat_transport_operators(cfg)

    # construct axial convection-dispersion operators for retentate gas
    def _build_gas_transport_operators(self) -> None:
        cfg = self.cfg

        # Boundary conditions:
        bc_axial = (
            {"a": 0.0, "b": 1.0, "d": cfg.inlet_state},  # for z=0: c = c_in and T = T_set
            {"a": 1.0, "b": 0.0, "d": 0.0},               # z=L: dc/dz = 0 and dT/dz = 0
        )

        conv_mat, conv_bc = construct_convflux_upwind(self.gas_shape, self.z_f, self.z_c, bc=bc_axial, v=cfg.v_ret, axis=0)
        div_mat = construct_div(self.gas_shape, self.z_f, axis=0)

        transport_mat = div_mat @ conv_mat 
        transport_const = div_mat @ conv_bc
        return transport_mat, transport_const

    # Construct axial convection operators for permeate gas
    def _build_permeate_transport_operators(self) -> None:
        cfg = self.cfg
        
        # Boundary conditions
        bc_permeate = (
            {"a": 0.0, "b": 1.0, "d": cfg.perm_inlet},   # z=0: c_perm = 0, T = T_in
            {"a": 1.0, "b": 0.0, "d": 0.0},           # z=L: dc/dz = 0
        )

        conv_mat_m, conv_bc_m = construct_convflux_upwind(self.gas_shape, self.z_f, self.z_c, bc=bc_permeate, v=cfg.v_perm, axis=0)
        div_mat_m = construct_div(self.gas_shape, self.z_f, axis=0)
        self.perm_transport_mat = div_mat_m @ conv_mat_m
        self.perm_transport_const = div_mat_m @ conv_bc_m

#    