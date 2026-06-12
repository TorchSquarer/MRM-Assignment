from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for candidate in (Path.cwd(), Path.cwd().parent):
    pymrm_src = candidate / "pymrm" / "src"
    if pymrm_src.exists() and str(pymrm_src) not in sys.path:
        sys.path.insert(0, str(pymrm_src))

from pymrm import (
    NumJac,
    construct_coefficient_matrix,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    newton,
    non_uniform_grid,
)

@dataclass
class ModelConfig:
    length: float = 1 # length [m]
    R_ret: float = 5e-3 # radius of retentate [m]
    R_perm: float = R_ret + 5e-4 # radius of permeate [m]
    R_out: float = 7e-3 # outer radius [m]

    P_vector: np.ndarray = field(default_factory=lambda: np.array([0.000, 0.000, 0.000, 0.002, 0.000])) # membrane permeability
    v_ret: float = 1 # velocity of retentate [m/s]
    v_perm: float = 1e-1 # velocity of permeate [m/s]
    d_ax: float = 2.0e-5 # dispersion in the axial direction 
    
    n_z: int = 100 # number of grid points in the axial direction
    n_r_perm: int = 30 # number of grid points in the radial direction of permeate
    n_r_ret: int = 30 # number of grid points in the radial direction of retentate
    n_c: int = 5 # number of components

    # Catalyst & Thermodynamic Parameters
    particle_radius: float = 1.0e-3  # Catalyst particle radius [m]
    particle_diffusivity: np.ndarray = field(
        default_factory=lambda: np.array([2.0e-6, 1.5e-6, 1.5e-6, 1.5e-6, 1.5e-6])
    )  # Diffusivity per component [m²/s]
    eps_s: float = 0.4  # Solid holdup (catalyst bed voidage component) [-]
    mu: float = 2.0e-5  # Gas viscosity [Pa·s]
    dp: float = 2.0e-3  # Particle diameter [m]
    T: float = 400.0  # Operating temperature [K]
    R: float = 8.314  # Ideal gas constant [J/(mol·K)]
    D_eff: float = 1.0e-5  # Effective film diffusivity [m²/s]
    p: float = 30e5  # Operational Pressure [Pa]
    p_0: float = 200e5  # Reference Pressure for volume change [Pa]

    # Kinetic Parameters: Reaction 1 (CO2 + 3H2 <-> MeOH + H2O) (Ghosh et al.)
    T1_ref: float = 573.15  # Reference temperature [K] (300 °C)
    k1_pre: float = 6.9e-4  # Forward rate constant at T_ref [mol/(s·bar²·kg_cat)]
    k1_eq: float = 2.5e-4  # Equilibrium constant at T_ref [bar⁻²]
    Ea_1: float = 35.7e3  # Activation energy [J/mol]
    K_CO2_ref: float = 0.79  # CO2 adsorption constant at T_ref [bar⁻¹]
    K_H2_ref: float = 0.76  # H2 adsorption constant at T_ref [bar⁻¹]
    dH_CO2: float = -25.9e3  # Enthalpy of adsorption for CO2 [J/mol]
    dH_H2: float = -12.5e3  # Enthalpy of adsorption for H2 [J/mol]

    # Kinetic Parameters: Reaction 2 (CO2 + 2MeOH <-> DMC + H2O) (ibrahim et al.)  
    T2_ref: float = 298.15
    k2_pre: float = 0.8  # Pre-exponential factor [s⁻¹]
    k_ads1: float = 9
    k_ads2: float = 109
    Ea_2: float = 106e3  # Activation energy [J/mol]
    dH_DMC: float = -20e3
    dG_DMC: float = 31e3
    
    rho_cat: float = 1500 # Catalyst density [kg/m³]
    m_cat: float = 4.6

    inlet_concentration: np.ndarray = field(default_factory=lambda: np.array([900, 2700, 0.0, 0.0, 0.0])) # inlet concentration of the components, CO2, H2, CH3OH, H2O, DMC

    tol: float = 1.0e-6  # tolerance for convergence
    maxfev: int = 30    # maximum number of function evaluations

    def k_eff_r1(self) -> float:
        return self.k1_pre * np.exp(
            (self.Ea_1 / self.R) * (1.0 / self.T_ref - 1.0 / self.T))
    
    def k_eff_r2(self, P_total_Pa: float | np.ndarray) -> float | np.ndarray:
        return self.k2_pre * np.exp(
            (-self.Ea_DMC - self.dV * (P_total_Pa - self.p_0)) / (self.R * self.T))
    
    def K_ads(self, K_ref: float, dH: float) -> float:
        return K_ref * np.exp((-dH / self.R) * (1.0 / self.T_ref - 1.0 / self.T))

    @property # external paricle surface area per unit bed volume
    def external_area(self):
        return 3.0 * self.eps_s / self.particle_radius
    
    @property # Membrane interfacial area per unit volume of retentate tube (2 π R_ret L / (π R_ret² L) = 2 / R_ret )
    def a_ret(self):
        return 2.0 / self.R_ret
    
    @property # membrane area per unit volume of the permeate (2 π R_ret L / (π (R_out² - R_perm²) L) = 2 R_ret / (R_out² - R_perm²))
    def a_perm(self):
        return 2.0 * self.R_ret / (self.R_out**2 - self.R_perm**2)
    
    @property
    def rho_bulk(self):
        return (1-self.eps_s) * self.rho_cat
    

cfg = ModelConfig()





