from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys
import numpy as np

for candidate in (Path.cwd(), Path.cwd().parent):
    pymrm_src = candidate / "pymrm" / "src"
    if pymrm_src.exists() and str(pymrm_src) not in sys.path:
        sys.path.insert(0, str(pymrm_src))

@dataclass
class ModelConfig:
    # reactor model
    length: float = 20 # length [m]
    R_ret: float = 10.0e-3 # radius of retentate [m]
    R_perm: float = R_ret + 5e-4 # radius of permeate [m]
    R_out: float = 12e-3 # outer radius [m]

    P_vector: np.ndarray = field(default_factory=lambda: np.array([0.000, 0.000, 0.000, 0.001, 0.000])) # Membrane permeability
    v_ret: float = 0.05 # velocity of retentate [m/s]
    v_perm: float = 0.1 # velocity of permeate [m/s]
    d_ax: float = 2.0e-5 # dispersion in the axial direction 
    
    n_z: int = 200 # number of grid points in the axial direction
    n_r_perm: int = 30 # number of grid points in the radial direction of permeate
    n_r_ret: int = 30 # number of grid points in the radial direction of retentate
    n_c: int = 5 # number of components

    # Catalyst & Thermodynamic Parameters
    T: float = 573.15  # Operating temperature [K]
    R: float = 8.314  # Ideal gas constant [J/(mol·K)]
    D_eff: float = 1.0e-5  # Effective film diffusivity [m²/s]
    p: float = 200e5  # Operational Pressure [Pa]
    p_0: float = 200e5  # Reference Pressure for volume change [Pa]
    p_stand: float = 1e5 # 1 bar in Pa

    tau: float = length / v_ret

    particle_radius: float = 1.0e-3  # Catalyst particle radius [m]
    particle_diffusivity: np.ndarray = field(default_factory=lambda: np.array([2.0e-6, 1.5e-6, 1.5e-6, 1.5e-6, 1.5e-6]))  # Diffusivity per component [m²/s]
    eps_s: float = 0.4  # Solid holdup (volume fraction catalyst) [-]
    eps_p: float = 0.5  # film porosity / active sites [-]
    rho_cat: float = 7215 # Catalyst density [kg/m³]
    mu: float = 2.0e-5  # Gas viscosity [Pa·s]
    dp: float = 2.0e-3  # Particle diameter [m]

    # Kinetic Parameters: Reaction 1 (CO2 + 3H2 <-> MeOH + H2O) (Ghosh et al.)
    T_ref: float = 573.15  # Reference temperature [K] (300 °C)
    k1_pre: float = 6.9e-4  # Forward rate constant at T_ref [mol/(s·bar²·kg_cat)]
    k1_eq: float = 2.5e-4  # Equilibrium constant at T_ref [bar⁻²]
    Ea_1: float = 35.7e3  # Activation energy [J/mol]
    K_CO2_ref: float = 0.79  # CO2 adsorption constant at T_ref [bar⁻¹]
    K_H2_ref: float = 0.76  # H2 adsorption constant at T_ref [bar⁻¹]
    dH_CO2: float = -25.9e3  # Enthalpy of adsorption for CO2 [J/mol]
    dH_H2: float = -12.5e3  # Enthalpy of adsorption for H2 [J/mol]
    r1_scale: float = 1.0

    # Kinetic Parameters: Reaction 2 (CO2 + 2MeOH <-> DMC + H2O) (ibrahim et al.)  
    k2_pre: float = 0.8  # Pre-exponential factor [s⁻¹]
    k2_eq: float = 3e-4 # guessed value
    k_ads1: float = 9
    k_ads2: float = 109
    Ea_DMC: float = 106e3  # Activation energy [J/mol]
    dV: float = 0.0
    r2_scale: float = 1.0

    feed_y: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.75, 0.0, 0.0, 0.0])) # inlet concentration of the components, CO2, H2, CH3OH, H2O, DMC

    tol: float = 1.0e-6  # tolerance for convergence
    maxfev: int = 30    # maximum number of function evaluations

    @property
    def inlet_concentration(self):
        feed_y = np.asarray(self.feed_y, dtype=float)
        feed_y = feed_y / np.sum(feed_y)

        c_total = self.p / (self.R * self.T)
        return c_total * feed_y

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
    def rho_bulk(self) -> float:
        return self.eps_s * self.eps_p * self.rho_cat

    @property
    def m_cat(self) -> float:
        V_tube = np.pi * self.R_ret**2 * self.length
        return self.rho_bulk * V_tube
    
    @property
    def v_tube(self) -> float:
        return np.pi * self.R_ret**2 * self.length

    def k_eff_r1(self) -> float:
        return self.k1_pre * np.exp(
            (self.Ea_1 / self.R) * (1.0 / self.T_ref - 1.0 / self.T)
        )

    def K_ads(self, K_ref: float, dH: float) -> float:
        return K_ref * np.exp(
            (dH / self.R) * (1.0 / self.T_ref - 1.0 / self.T)
        )

    def k_eff_r2(self, P_total_Pa=None):
        if P_total_Pa is None:
            P_total_Pa = self.p

        # Normal Arrhenius equation
        exponent_T = -self.Ea_DMC / (self.R * self.T)

        # Protection against numerical overflow
        exponent_T = np.clip(exponent_T, -700.0, 700.0)
        k_min = self.k2_pre * np.exp(exponent_T)  # [g_cat^-1 min^-1]

        # Optional pressure correction
        exponent_P = -self.dV * (P_total_Pa - self.p_0) / (self.R * self.T)
        exponent_P = np.clip(exponent_P, -700.0, 700.0)

        k_min = k_min * np.exp(exponent_P)

        # Convert min^-1 to s^-1
        return k_min / 60.0 







