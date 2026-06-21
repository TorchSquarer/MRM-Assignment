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
    length: float = 20              # length [m]
    R_ret: float = 10.0e-3          # radius of retentate [m]
    R_perm: float = R_ret + 5e-4    # radius of permeate [m]
    R_out: float = 12e-3            # outer radius [m]

    P_membrane: np.ndarray = field(default_factory=lambda: np.array([0.000, 0.000, 0.000, 0.005, 0.000])) # Membrane permeability [m/s]
    v_ret: float = 0.05                                                                                 # velocity of retentate [m/s]
    v_perm: float = 0.1                                                                                 # velocity of permeate [m/s]
    
    n_z: int = 200 # number of grid points in the axial direction
    n_r_perm: int = 30 # number of grid points in the radial direction of permeate
    n_r_ret: int = 30 # number of grid points in the radial direction of retentate
    n_c: int = 5 # number of components

    # Catalyst & Thermodynamic Parameters
    T: float = 573.15  # Operating temperature [K]
    R: float = 8.314  # Ideal gas constant [J/(mol·K)]
    D_eff: float = 1.0e-5  # Effective film diffusivity [m²/s]
    p: float = 50e5  # Operational Pressure [Pa]
    p_0: float = 200e5  # Reference Pressure for volume change [Pa]
    p_stand: float = 1e5 # 1 bar in Pa

    tau: float = length / v_ret
    
    particle_radius: float = 1.0e-3  # Catalyst particle radius [m]
    particle_diffusivity: np.ndarray = field(default_factory=lambda: np.array([2.0e-6, 1.5e-6, 1.5e-6, 1.5e-6, 1.5e-6]))  # Diffusivity per component [m²/s]
    eps_s: float = 0.4  # Solid holdup (volume fraction catalyst) [-]
    eps_p: float = 0.5  # film porosity / active sites [-]
    rho_cat: float = 7215 # Catalyst density [kg/m³]
    mu: float = 2.0e-5  # Gas viscosity [Pa·s]
    d_p: float = 2.0e-3  # Particle diameter [m]
    MW: np.ndarray = field(default_factory=lambda: np.array([44.01e-3, 2.016e-3, 32.04e-3, 18.015e-3,90.08e-3]))  #[kg/mol]
    lam: np.ndarray = field(default_factory=lambda: np.array([1, 2, 3, 4, 5]))  #[???] 
    # ^v ABOVE and VALUES NEED TO BE FOUND  YET [Dummyh values currently]!!!
    lam_s: float = 1 #[???] #CeO2 catalyst bed
    eps_bed: float = 1 - eps_s


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
    T_ref_DMC: float = 298.15 #k
    k2_pre: float = 0.8  # Pre-exponential factor [s⁻¹]
    k_ads1: float = 9
    k_ads2: float = 109
    dH_DMC: float = -20.10e3
    dG_DMC: float = 31.50e3
    Ea_DMC: float = 106e3  # Activation energy [J/mol]
    dV: float = -0.24e-6
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
    
    @property
    def Reynolds(self) -> float:
        return self.rho_gas * self.v_ret * self.d_p / self.mu

    @property
    def Schmidt(self) -> np.ndarray:
        return self.mu / (self.rho_gas * self.particle_diffusivity)

    @property
    def d_ax(self) -> np.ndarray:
        Re = self.Reynolds
        Sc = self.Schmidt
        eps_g = 1 - self.eps_s

        D_ax = self.v_ret * self.d_p * (
            (0.73 * eps_g) / (eps_g + 0.5/(Re * Sc))
            + (0.5 / (1 + 9.7 * eps_g / (Re * Sc)))
        )
        return D_ax
    
    @property
    def rho_gas(self) -> float:
        avg_Mw = np.sum(self.feed_y * self.MW)
        return self.p * avg_Mw / (self.R * self.T)
    
    @property
    def Cp_gas(self) -> float: # REDUNDANT AFTER TEST AS THIS WILL CHANGE ALONG Z-DIRECTION
        avg_Mw = np.sum(self.feed_y * self.MW)
        return self.p * avg_Mw / (self.R * self.T)
    
    #@property
    def thermal_conductivity_ax(self): #,y: np.ndarray) -> float:
        """
        1st, Wassilijewa Rule was applied here as the heat conductivity of the 
        gas mixture is dependent on the contribution of all species. The 
        proportions of these species change along of the reactor.
        2nd, the conductivity in the axial direction is also dependent on the 
        catalytic bed.
        """
        y=self.feed_y
        lam_ratios = np.outer(self.lam, 1/self.lam)
        MW_ratios = np.outer(self.MW, 1/self.MW)

        Phi_ij = (1 + np.sqrt(lam_ratios) * MW_ratios**0.25)**2 / np.sqrt(8 * (1 + MW_ratios))
        denom_i = Phi_ij @ y
        lam_fluid = np.sum(y * self.lam / denom_i)
        
        lam_static = (1 + self.eps_s) * lam_fluid + self.eps_s * self.lam_s

        return lam_static + 0.5 * self.rho_gas * self.Cp_gas * self.v_ret * self.d_p

#    def rho_fluid(self, x: np.ndarray) -> float:
#        return np.sum(x * self.rho_gas)
#    
#    def Cp_fluid(self, x: np.ndarray) -> float:
#        return np.sum(x * self.Cp_fluid)