from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class ModelConfig:
    length: float = 1.0 # length
    R_ret: float = 5e-3 # radius of retentate
    R_perm: float = R_ret + 5e-4 # radius of permeate
    R_out: float = 7e-3 # outer radius

    P_vector: np.ndarray = field(default_factory=lambda: np.array([0.000, 0.000, 0.000, 0.002, 0.000])) # membrane permeability
    v_ret: float = 1e-1 # velocity of retentate
    v_perm: float = 1e-1 # velocity of permeate
    v_sup: float = v_perm
    d_ax: float = 2.0e-5 # dispersion in the axial direction
    
    n_z: int = 100 # number of grid points in the axial direction
    n_r_perm: int = 30 # number of grid points in the radial direction of permeate
    n_r_ret: int = 30 # number of grid points in the radial direction of retentate
    n_c: int = 5 # number of components
    
    particle_radius: float = 1.0e-3 # catalyst particle radius
    particle_diffusivity: np.ndarray = field(default_factory=lambda: np.array([2.0e-6, 1.5e-6, 1.5e-6, 1.5e-6, 1.5e-6]))   # diffusivity in the particle per component
    eps_s: float = 0.35  # solid holdup

    k_1: float = 1.0   #  reaction rate constant for reaction 1
    k_2: float = 1.0   #  reaction rate constant for backward reaction 1
    k_3: float = 1.0   #  reaction rate constant for reaction 2
    k_4: float = 1.0   #  reaction rate constant for backward reaction 2

   
    inlet_concentration: np.ndarray = field(default_factory=lambda: np.array([1.0, 3.0, 0.0, 0.0, 0.0])) # inlet concentration of the components, CO2, H2, CH3OH, H2O, DMC
    film_pair_coefficients: np.ndarray = field(  # mass transfer coefficients for the film model, in the order of (CO2, H2, CH3OH, H2O, DMC)
        default_factory=lambda: np.array( 
            [
                [np.inf, 2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3],
                [2.0e-3, np.inf, 4.0e-3, 3.0e-3, 2.5e-3],
                [2.0e-3, 4.0e-3, np.inf, 3.5e-3, 2.0e-3],
                [2.0e-3, 3.0e-3, 3.5e-3, np.inf, 2.2e-3],
                [2.0e-3, 2.5e-3, 2.0e-3, 2.2e-3, np.inf],
                
            ]
        )
    ) # k_ij pair diffusion coefficient
    tol: float = 1.0e-8  # tolerance for convergence
    maxfev: int = 20    # maximum number of function evaluations

    @property # external paricle surface area per unit bed volume
    def external_area(self):
        return 3.0 * self.eps_s / self.particle_radius

    @property # Maxwell-Stefan k_ij * a_vO2, H2, CH3OH, H2O, DMC)
    def kma(self):
        return self.film_pair_coefficients * self.external_area
    
    @property # Membrane interfacial area per unit volume of retentate tube (2 π R_ret L / (π R_ret² L) = 2 / R_ret )
    def a_ret(self):
        return 2.0 / self.R_ret
    
    @property # membrane area per unit volume of the permeate (2 π R_ret L / (π (R_out² - R_perm²) L) = 2 R_ret / (R_out² - R_perm²))
    def a_perm(self):
        return 2.0 * self.R_ret / (self.R_out**2 - self.R_perm**2)

cfg = ModelConfig()

pd.Series(
    {
        "axial cells": cfg.n_z,
        "particle radial cells": [cfg.n_r_ret, cfg.n_r_perm],
        "reactor length [m]": cfg.length,
        "velocity [m/s]": [cfg.v_perm, cfg.v_ret],
        "particle radius [mm]": 1e3 * cfg.particle_radius,
        "solid holdup [-]": cfg.eps_s,
        "external area [1/m]": cfg.external_area,
        "retentate area [1/m]": cfg.a_ret,
        "permeate area [1/m]": cfg.a_perm,
        "reaction rate constant [1/s]": [cfg.k_1, cfg.k_2, cfg.k_3, cfg.k_4], 
    },
    name="settings",
)