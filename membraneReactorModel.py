from __future__ import annotations  
from pathlib import Path
import sys
import numpy as np

from config import ModelConfig
from buildoperators import TransportOperators
from reaction import ReactionRates

for candidate in (Path.cwd(), Path.cwd().parent):
    pymrm_src = candidate / "pymrm" / "src"
    if pymrm_src.exists() and str(pymrm_src) not in sys.path:
        sys.path.insert(0, str(pymrm_src))

from pymrm import (
    NumJac,
    newton,
)

STOICH = np.array(
    [
        [-1.0, -3.0,  1.0,  1.0,  0.0],  # R1: CO2 + 3H2 <-> CH3OH + H2O
        [-1.0,  0.0, -2.0,  1.0,  1.0],  # R2: CO2 + 2CH3OH <-> DMC + H2O
    ]
)

SPECIES_LABELS = ("CO2", "H2", "CH3OH", "H2O", "DMC")

# Full membrane reactor model
class MembraneReactorModel:
    species_labels = SPECIES_LABELS
    stoich = STOICH

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.ops = TransportOperators(cfg)
        self.reac = ReactionRates(cfg)

        # State array shapes
        self.shape = (cfg.n_z, cfg.n_r_ret + 3, cfg.n_c)
        self.gas_shape = (cfg.n_z, cfg.n_c)
        self.particle_shape = (cfg.n_z, cfg.n_r_ret, cfg.n_c)

        # Grid coordinates for plotting/post-processing
        self.z_c    = self.ops.z_c
        self.r_c    = self.ops.r_c_ret

        # Numerical Jacobian configuration
        self.numjac = NumJac(self.shape, axes_diagonals=[0], axes_blocks=[1, 2])
        self.u0 = self._initial_state()
        self.u  = self.u0.copy()

    # Build initial concentration field
    def _initial_state(self) -> np.ndarray:
        u = np.zeros(self.shape)
        u[:, :-1, :] = self.cfg.inlet_concentration.reshape(1, 1, self.cfg.n_c)
        u[:, -1, :]  = 0.0
        return u

    # split full state array into model regions:
    # c_g is retentate bulk gas, c_b is particle surface conentration, 
    # c_p is intraparticle concentration and c_m is permeate concentration [mol/m3]
    def split_state(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = u.reshape(self.shape)
        return u[:, 0, :], u[:, 1, :], u[:, 2:-1, :], u[:, -1, :]

    # volume-averages the intraparticle reaction source over the particle radius
    def _particle_average_source(self, c_p: np.ndarray) -> np.ndarray:
        source = self.reac.particle_reaction_rates(c_p, self.cfg)
        return np.sum(source * self.ops.volume_weights.reshape(1, -1, 1), axis=1)

    # calculates apparent gas-particle exchange source from particle surface flux
    def _particle_apparent_source(self, c_p: np.ndarray, c_b: np.ndarray) -> np.ndarray:
        ops = self.ops
        c_p_vec = c_p.reshape(-1, 1)
        c_b_vec = c_b[:, None, :].reshape(-1, 1)
        return (
            ops.particle_apparent_mat    @ c_p_vec
            + ops.particle_apparent_bc_mat @ c_b_vec
        ).reshape(self.gas_shape)

    # Evaluates nonlinear residual calues
    def residual_values(self, u: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        ops = self.ops

        c_g, c_b, c_p, c_m = self.split_state(u)
        residual = np.zeros_like(u).reshape(self.shape)

        # [0] Retentate bulk: convection–dispersion – apparent source + membrane loss
        gas_transport = (
            ops.gas_transport_const
            + ops.gas_transport_mat @ c_g.reshape(-1, 1)
            ).reshape(self.gas_shape)
            
        membrane_flux = cfg.P_membrane.reshape(1, -1) * (c_g - c_m)
        source_reactor = cfg.eps_s * self._particle_apparent_source(c_p, c_b)

        residual[:, 0, :] = gas_transport - source_reactor + cfg.a_ret * membrane_flux
        
        # [1] Boundary layer constraint, no external film resistance
        residual[:, 1, :] = c_b - c_g 
        
        # [2:-1] Intraparticle diffusion–reaction
        particle_diffusion = (
            ops.particle_diffusion_mat @ c_p.reshape(-1, 1)
            + ops.particle_boundary_mat @ c_b[:, None, :].reshape(-1, 1)
            ).reshape(self.particle_shape)

        source_particle = self.reac.particle_reaction_rates(c_p, cfg)

        residual[:, 2:-1, :] = particle_diffusion - source_particle

        # [-1] Permeate gas balance, axial convection + membrane gain
        permeate_transport = (
            ops.perm_transport_const
            + ops.perm_transport_mat @ c_m.reshape(-1, 1)
            ).reshape(self.gas_shape)

        residual[:, -1, :] = permeate_transport - cfg.a_perm * membrane_flux

        return residual

    # Returns residual vector and numerical Jacobion for Newton solver
    def residual(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:   
        u = u.reshape(self.shape)
        f = self.residual_values(u)
        f, jac = self.numjac(self.residual_values, u, f_value=f)
        return f.ravel(), jac

    # Solves the nonlinear Steady-state model
    def solve(self):
        result = newton(self.residual, self.u, tol=self.cfg.tol, maxfev=self.cfg.maxfev, solver="spsolve")
        self.u = result.x.reshape(self.shape)
        self.result = result
        return result
    
    # Return solved concentration fields
    def fields(self,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.split_state(self.u)

    # Calculates axial catalyst effectiveness factors for both reactions
    def effectiveness_profile(self,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _c_g, c_b, c_p, _c_m = self.fields()
        r1_int, r2_int = self.reac.reaction_rates(c_p, self.cfg)
        weights = self.ops.volume_weights.reshape(1, -1)

        r1_apparent = np.sum(r1_int * weights, axis=1)
        r2_apparent = np.sum(r2_int * weights, axis=1)

        r1_surf_full, r2_surf_full = self.reac.reaction_rates(c_b[:, None, :], self.cfg)
      
        r1_surface = r1_surf_full[:, 0]   
        r2_surface = r2_surf_full[:, 0]   

        eta_r1 = r1_apparent / np.maximum(r1_surface, 1.0e-30)
        eta_r2 = r2_apparent / np.maximum(r2_surface, 1.0e-30)
        return eta_r1, eta_r2, r1_surface, r2_surface
    
    # calculates mears criterian for the external mass transfer
    def mears_criterion(self):
        cfg = self.cfg

        c_g, c_b, c_p, c_m = self.fields()

        r1_surface, r2_surface = self.reac.reaction_rates(c_b[:, None, :], cfg)

        r1_surface = r1_surface[:, 0]
        r2_surface = r2_surface[:, 0]

        r1_cat = (r1_surface / cfg.rho_bulk) # (mol/ kg cat * s)
        r2_cat = (r2_surface / cfg.rho_bulk) # (mol/ kg cat * s)

        C_CO2 = np.maximum(c_g[:, 0], 1e-30)  # (mol/m^3)

        mears_r1 = (r1_cat * cfg.particle_radius * cfg.n) / (cfg.K_gs * C_CO2)
        mears_r2 = (r2_cat * cfg.particle_radius * cfg.n) / (cfg.K_gs * C_CO2)

        return mears_r1, mears_r2
    
    # calculates weisz-prater criterion for the internal mass transfer
    def weisz_prater_criterion(self):
        cfg = self.cfg

        c_g, c_b, c_p, c_m = self.fields()

        r1_surface, r2_surface = self.reac.reaction_rates(c_b[:, None, :], cfg)

        r1_surface = r1_surface[:, 0]
        r2_surface = r2_surface[:, 0]

        C_CO2 = np.maximum(c_g[:, 0], 1e-30)

        wp_r1 = (r1_surface * cfg.particle_radius**2) / (cfg.D_eff * C_CO2)
        wp_r2 = (r2_surface * cfg.particle_radius**2) / (cfg.D_eff * C_CO2)

        return wp_r1, wp_r2
