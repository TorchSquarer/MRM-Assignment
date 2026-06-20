from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

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

    # Construc axial and radial grids
    def _build_grids(self) -> None:
        cfg = self.cfg
        
        # Axial faces and cell centers
        self.z_f = np.linspace(0.0, cfg.length, cfg.n_z + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])

        # Particle radial grid in tetentate catalyst particles.
        self.r_f_ret = non_uniform_grid(
            0.0, cfg.particle_radius, cfg.n_r_ret + 1,
            0.15 * cfg.particle_radius, 0.75
        )
        self.r_c_ret = 0.5 * (self.r_f_ret[:-1] + self.r_f_ret[1:])

        # Permeate radial grid (not used yet)
        self.r_f_perm = non_uniform_grid(
            cfg.R_ret, cfg.R_perm, cfg.n_r_perm + 1,
            0.15 * (cfg.R_perm - cfg.R_ret), 0.75
        )
        self.r_c_perm = 0.5 * (self.r_f_perm[:-1] + self.r_f_perm[1:])

        # spherical particle volume weights, averaging paricle concentratons over the particle volume
        self.volume_weights = np.diff(self.r_f_ret**3) / self.r_f_ret[-1]**3

    # construc gas, particle, and permeate transport matrices
    def _build_operators(self) -> None:
        cfg = self.cfg
        
        self._build_gas_transport_operators(cfg)
        self._build_particle_diffusion_operators(cfg)
        self._build_permeate_transport_operators(cfg)

    # construct axial convection-dispersion operators for retentate gas
    def _build_gas_transport_operators(self, cfg: ModelConfig) -> None:

        # Boundary conditions:
        bc_axial = (
            {"a": 0.0, "b": 1.0, "d": cfg.inlet_concentration},  # z=0: c = c_in
            {"a": 1.0, "b": 0.0, "d": 0.0},                      # z=L: dc/dz = 0
        )

        grad_mat, grad_bc = construct_grad(self.gas_shape, self.z_f, self.z_c, bc=bc_axial, axis=0)
        conv_mat, conv_bc = construct_convflux_upwind(self.gas_shape, self.z_f, self.z_c, bc=bc_axial, v=cfg.v_ret, axis=0)
        div_mat = construct_div(self.gas_shape, self.z_f, axis=0)
        d_ax_mat = construct_coefficient_matrix(cfg.d_ax, self.gas_shape, axis=0)
        self.gas_transport_mat = div_mat @ (conv_mat - d_ax_mat @ grad_mat)
        self.gas_transport_const = div_mat @ (conv_bc - d_ax_mat @ grad_bc)

    def _build_particle_diffusion_operators(self, cfg: ModelConfig) -> None:
        
        # Boundary conditions
        bc_particle = (
            {"a": 1.0, "b": 0.0, "d": 0.0},   # r=0: zero flux (symmetry)
            {"a": 0.0, "b": 1.0, "d": 1.0},   # r=R_p: c = c_boundary (overridden by bc vector)
        )

        grad_p_mat, _, grad_p_bc = construct_grad(self.particle_shape, self.r_f_ret, self.r_c_ret, bc=bc_particle, axis=1, shapes_d=(None, self.boundary_shape), )
        div_p_mat = construct_div(self.particle_shape, self.r_f_ret, nu=2, axis=1)
        d_p = cfg.particle_diffusivity.reshape(1, 1, cfg.n_c)
        d_p_mat = construct_coefficient_matrix(d_p, self.particle_shape, axis=1)
        flux_p_mat = -d_p_mat @ grad_p_mat   
        flux_p_bc = -d_p_mat @ grad_p_bc    
        self.particle_diffusion_mat = div_p_mat @ flux_p_mat
        self.particle_boundary_mat = div_p_mat @ flux_p_bc

        # Extract rows corresponding to the outer particle surface.
        # These are used to calculate the apparent flux from gas into particles.
        face_shape = (cfg.n_z, cfg.n_r_ret + 1, cfg.n_c)
        outer_face_rows = (face_shape[2] * face_shape[1] * np.arange(face_shape[0]).reshape((-1, 1))
            + face_shape[2] * cfg.n_r_ret
            + np.arange(face_shape[2]).reshape((1, -1))).ravel()

        self.particle_apparent_mat = (3.0 / cfg.particle_radius) * flux_p_mat[outer_face_rows, :]
        self.particle_apparent_bc_mat = (3.0 / cfg.particle_radius) * flux_p_bc[outer_face_rows, :]

    # Construct axial convection operators for permeate gas
    def _build_permeate_transport_operators(self, cfg: ModelConfig) -> None:
        
        # Boundary conditions
        bc_permeate = (
            {"a": 0.0, "b": 1.0, "d": 0.0},   # z=0: c_perm = 0
            {"a": 1.0, "b": 0.0, "d": 0.0},   # z=L: dc/dz = 0
        )

        conv_mat_m, conv_bc_m = construct_convflux_upwind(self.gas_shape, self.z_f, self.z_c, bc=bc_permeate, v=cfg.v_perm, axis=0)
        div_mat_m = construct_div(self.gas_shape, self.z_f, axis=0)
        self.perm_transport_mat = div_mat_m @ conv_mat_m
        self.perm_transport_const = div_mat_m @ conv_bc_m