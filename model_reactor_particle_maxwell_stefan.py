from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from modelconfig import ModelConfig as mc
from pymrm import (
    NumJac,
    construct_coefficient_matrix,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    newton,
    non_uniform_grid,
)

class ReactorParticleMaxwellStefanModel:
    species_labels = ("CO2", "H2", "CH3OH", "H2O", "DMC") # CO2, H2, CH3OH, H2O, DMC
    stoich = np.array([[-1.0, -3.0, 1.0, 1.0, 0.0], # r1: A + 3B <-> C + D
                      [-1.0, 0.0, -2.0, 1.0, 1.0]]) # r2: A + 2C <-> D + E 

    def __init__(self, cfg: mc): 
        self.cfg = cfg
        self.ret_shape = (cfg.n_z, cfg.n_c) # col 0 : retentate bulk gas
        self.gas_shape = self.ret_shape
        self.boundary_shape = (cfg.n_z, 1, cfg.n_c) # col 1:  film/boundary layer
        self.particle_shape = (cfg.n_z, cfg.n_r_ret, cfg.n_c) # cols2 ... n_r+r : particle interior
        self.shape = (cfg.n_z, cfg.n_r_ret + 3, cfg.n_c) # total columns: 1(gas) + 1(film) + n_r_ret(particle interior) + 1(permeate)
        self._build_grids()
        self._build_operators()
        self.numjac = NumJac(self.shape, axes_diagonals=[0], axes_blocks=[1, 2])
        self.u0 = self.initial_state()
        self.u = self.u0.copy()
        
    def _build_grids(self):
        cfg = self.cfg
        self.z_f = np.linspace(0.0, cfg.length, cfg.n_z + 1) # axial grid
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.r_f_ret = non_uniform_grid(0.0, cfg.particle_radius, cfg.n_r_ret + 1, 0.15 * cfg.particle_radius, 0.75) # non uniform radial grid in retentate
        self.r_c_ret = 0.5 * (self.r_f_ret[:-1] + self.r_f_ret[1:])
        self.r_f_perm = non_uniform_grid(cfg.R_ret, cfg.R_perm, cfg.n_r_perm + 1, 0.15 * (cfg.R_perm - cfg.R_ret), 0.75) # non uniform radial grid in permeate
        self.r_c_perm = 0.5 * (self.r_f_perm[:-1] + self.r_f_perm[1:])
        self.volume_weights = np.diff(self.r_f_ret**3) / self.r_f_ret[-1] ** 3

    def _build_operators(self):
        cfg = self.cfg
        
        bc_axial = (
            {"a": 0.0, "b": 1.0, "d":cfg.inlet_concentration}, # dirchlet
            {"a": 1.0, "b": 0.0, "d": 0.0}, # Neumann
        )
        grad_mat_ret, grad_bc_ret = construct_grad(self.gas_shape, self.z_f, self.z_c, bc=bc_axial, axis=0) # type: ignore
        conv_mat_ret, conv_bc_ret = construct_convflux_upwind(self.gas_shape, self.z_f, self.z_c, bc=bc_axial, v=cfg.v_ret, axis=0) # type: ignore
        div_mat_ret = construct_div(self.gas_shape, self.z_f, axis=0)
        d_ax_mat_ret = construct_coefficient_matrix(cfg.d_ax, self.gas_shape, axis=0)
        self.gas_transport_mat = div_mat_ret @ (conv_mat_ret - d_ax_mat_ret @ grad_mat_ret)
        self.gas_transport_const = div_mat_ret @ (conv_bc_ret - d_ax_mat_ret @ grad_bc_ret)

        bc_particle = (
            {"a": 1.0, "b": 0.0, "d": 0.0}, # dirichlet at the center of the particle
            {"a": 0.0, "b": 1.0, "d": 1.0}, # Neumann at the outer edge of the particle
        )

        #shape of div and grad: (n_z * n_r_ret * n_c, n_z * n_r_ret * n_c), shape of grad_bc: (n_z * 1 * n_c, n_z * n_r_ret * n_c)
        grad_p_mat, _, grad_p_bc = construct_grad(self.particle_shape, self.r_f_ret, self.r_c_ret, bc=bc_particle, axis=1, shapes_d=(None, self.boundary_shape),) # type: ignore
        div_p_mat = construct_div(self.particle_shape, self.r_f_ret, nu=2, axis=1)
        d_p = cfg.particle_diffusivity.reshape(1, 1, cfg.n_c)
        d_p_mat = construct_coefficient_matrix(d_p, self.particle_shape, axis=1)
        flux_p_mat = -d_p_mat @ grad_p_mat # difusive flux matrix in the particle interior
        flux_p_bc = -d_p_mat @ grad_p_bc # diffusive flux contribution from bc
        self.particle_diffusion_mat = div_p_mat @ flux_p_mat
        self.particle_boundary_mat = div_p_mat @ flux_p_bc

        face_shape = (cfg.n_z, cfg.n_r_ret + 1, cfg.n_c)
        outer_face_rows = (
            face_shape[2] * face_shape[1] * np.arange(face_shape[0]).reshape((-1, 1))
            + face_shape[2] * cfg.n_r_ret
            + np.arange(face_shape[2]).reshape((1, -1))
        ).ravel()
        self.particle_apparent_mat = (3.0 / cfg.particle_radius) * flux_p_mat[outer_face_rows, :]
        self.particle_apparent_bc_mat = (3.0 / cfg.particle_radius) * flux_p_bc[outer_face_rows, :]

        # bc_perm_inlet = cfg.v_perm * np.zeros(cfg.n_c) 
        bc_permeate = (
            {"a": 0.0, "b": 1.0, "d": 0.0}, # c_m = 0 at z = 0 (Dirichlet)
            {"a": 1.0, "b": 0.0, "d": 0.0}, # dc_m/dz = 0 at z = L (Neumann)
        )
        conv_mat_m, conv_bc_m = construct_convflux_upwind(self.gas_shape, self.z_f, self.z_c, bc=bc_permeate, v=cfg.v_perm, axis=0) # type: ignore
        div_mat_m = construct_div(self.gas_shape, self.z_f, axis=0)
        self.perm_transport_mat = div_mat_m @ conv_mat_m
        self.perm_transport_const = div_mat_m @ conv_bc_m 

    def initial_state(self): # intitialise retatentate, film, particle interior, permeate
        u = np.zeros(self.shape)
        u[:, :-1, :] = self.cfg.inlet_concentration.reshape(1, 1, self.cfg.n_c)
        u[:, -1, :] = 0.0
        return u

    def split_state(self, u): # flat state vector to fields (gas, boundary, particle interior, permeate)
        u = u.reshape(self.shape)
        return u[:, 0, :], u[:, 1, :], u[:, 2:-1, :], u[:, -1, :]
    
    def particle_source(self, c_p): # reaction source term in the particle, shape (n_z, n_r_ret, n_c)
        r1 = (self.cfg.k_1 * c_p[..., 0] * (c_p[..., 1]**3) - self.cfg.k_2 * c_p[..., 2] * c_p[..., 3])
        r2 = (self.cfg.k_3 * c_p[..., 0] * (c_p[..., 2]**2) - self.cfg.k_4 * c_p[..., 4] * c_p[..., 3])
        
        rates = np.stack([r1, r2], axis=-1)
        return np.einsum('zrc,co->zro', rates, self.stoich)

    def particle_average_source(self, c_p): # average reaction source term in the particle, shape (n_z, n_c)
        source = self.particle_source(c_p)
        return np.sum(source * self.volume_weights.reshape(1, -1, 1), axis=1)

    def particle_apparent_source(self, c_p, c_b): # apparent reaction source term in the particle, shape (n_z, n_c)
        c_p_vec = c_p.reshape(-1, 1)
        c_b_vec = c_b[:, None, :].reshape(-1, 1)
        return (
            self.particle_apparent_mat @ c_p_vec
            + self.particle_apparent_bc_mat @ c_b_vec
        ).reshape(self.gas_shape)

    def maxwell_stefan_film_residual(self, c_g, c_b, source_reactor):
        c_mid = 0.5 * (c_g + c_b)
        y_mid = c_mid / np.maximum(np.sum(c_mid, axis=-1, keepdims=True), 1.0e-30)
        correction = np.zeros_like(c_g)
        for i in range(self.cfg.n_c):
            for j in range(self.cfg.n_c):
                if i != j:
                    correction[:, i] += (
                        y_mid[:, j] * source_reactor[:, i]
                        - y_mid[:, i] * source_reactor[:, j]
                    ) / self.cfg.kma[i, j]
        return c_b - c_g - correction

    def residual_values(self, u):
        # Block layout (matching self.shape):
          # [0]    retentate bulk ADE  +  membrane sink
          # [1]    Maxwell–Stefan film equation
          # [2:-1] particle diffusion–reaction
          # [-1]   permeate convection  +  membrane source
        c_g, c_b, c_p, c_m = self.split_state(u) 
        source_particle = self.particle_source(c_p)
        source_reactor = self.cfg.eps_s * self.particle_apparent_source(c_p, c_b)
        p_mask = self.cfg.P_vector.reshape(1, -1)
        membrane_flux_density = p_mask * (c_g - c_m)
        residual = np.zeros_like(u).reshape(self.shape)
        residual[:, 0, :] = (
            self.gas_transport_const
            + self.gas_transport_mat @ c_g.reshape(-1, 1)
        ).reshape(self.gas_shape) - source_reactor + (self.cfg.a_ret * membrane_flux_density)
        residual[:, 1, :] = self.maxwell_stefan_film_residual(c_g, c_b, source_reactor)
        residual[:, 2:-1, :] = (
            self.particle_diffusion_mat @ c_p.reshape(-1, 1)
            + self.particle_boundary_mat @ c_b[:, None, :].reshape(-1, 1)
        ).reshape(self.particle_shape) - source_particle
        residual[:, -1, :] = (
            self.perm_transport_const + self.perm_transport_mat @ c_m.reshape(-1, 1)
        ).reshape(self.gas_shape) - (self.cfg.a_perm * membrane_flux_density)
        return residual

    def residual(self, u):
        u = u.reshape(self.shape)
        residual = self.residual_values(u)
        residual, jac = self.numjac(self.residual_values, u, f_value=residual)
        return residual.ravel(), jac

    def solve(self):
        result = newton(self.residual, self.u, tol=self.cfg.tol, maxfev=self.cfg.maxfev, solver="spsolve")
        self.u = result.x.reshape(self.shape)
        self.result = result
        return result

    def fields(self):
        return self.split_state(self.u)

    def effectiveness_profile(self, eps: float = 1.0e-12):
        _c_g, c_b, c_p, _c_m = self.fields()

        w = self.volume_weights.reshape(1, -1)

        r1_fwd_local = self.cfg.k_1 * c_p[..., 0] * (c_p[..., 1] ** 3)
        r1_bwd_local = self.cfg.k_2 * c_p[..., 2] * c_p[..., 3]
        r1_net_local = r1_fwd_local - r1_bwd_local

        r2_fwd_local = self.cfg.k_3 * c_p[..., 0] * (c_p[..., 2] ** 2)
        r2_bwd_local = self.cfg.k_4 * c_p[..., 4] * c_p[..., 3]
        r2_net_local = r2_fwd_local - r2_bwd_local

    # Volume-averaged apparent rates
        r1_app_fwd = np.sum(r1_fwd_local * w, axis=1)
        r1_app_bwd = np.sum(r1_bwd_local * w, axis=1)
        r1_app_net = np.sum(r1_net_local * w, axis=1)

        r2_app_fwd = np.sum(r2_fwd_local * w, axis=1)
        r2_app_bwd = np.sum(r2_bwd_local * w, axis=1)
        r2_app_net = np.sum(r2_net_local * w, axis=1)

    # Surface rates evaluated at pellet boundary concentration c_b
        r1_surf_fwd = self.cfg.k_1 * c_b[:, 0] * (c_b[:, 1] ** 3)
        r1_surf_bwd = self.cfg.k_2 * c_b[:, 2] * c_b[:, 3]
        r1_surf_net = r1_surf_fwd - r1_surf_bwd

        r2_surf_fwd = self.cfg.k_3 * c_b[:, 0] * (c_b[:, 2] ** 2)
        r2_surf_bwd = self.cfg.k_4 * c_b[:, 4] * c_b[:, 3]
        r2_surf_net = r2_surf_fwd - r2_surf_bwd

        def safe_div(num, den, eps):
            out = np.full_like(num, np.nan, dtype=float)
            mask = np.abs(den) > eps
            out[mask] = num[mask] / den[mask]
            return out

        eta_r1_fwd = safe_div(r1_app_fwd, r1_surf_fwd, eps)
        eta_r1_bwd = safe_div(r1_app_bwd, r1_surf_bwd, eps)
        eta_r1_net = safe_div(r1_app_net, r1_surf_net, eps)

        eta_r2_fwd = safe_div(r2_app_fwd, r2_surf_fwd, eps)
        eta_r2_bwd = safe_div(r2_app_bwd, r2_surf_bwd, eps)
        eta_r2_net = safe_div(r2_app_net, r2_surf_net, eps)

        return {
            "eta_r1_net": eta_r1_net,
            "eta_r2_net": eta_r2_net,
            "eta_r1_fwd": eta_r1_fwd,
            "eta_r2_fwd": eta_r2_fwd,
            "eta_r1_bwd": eta_r1_bwd,
            "eta_r2_bwd": eta_r2_bwd,
        }