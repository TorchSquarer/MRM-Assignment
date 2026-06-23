from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelConfig:
    # Reactor geometry
    length: float = 10.0 # length [m]
    R_ret: float = 10.0e-3 # Radius of retentate [m]
    R_perm: float = R_ret + 5.0e-4  # Radius of permeate [m]
    R_out: float = 12e-3 # Outer reactor radius [m]

    # Numerical grid
    n_z: int = 100 # number of axial grid points
    n_r_perm: int = 30 # number of radial grid points in permeate
    n_r_ret: int = 30 # number of radial grid points in retentate
    n_species: int = 5 # number of components

    @property
    def n_c(self) -> int:
        return self.n_species + 1
    
    @property
    def iT(self) -> int:
        return self.n_c - 1

    # Membrane and flow parameters
    P_membrane: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0048, 0.0, 0.0])) # Membrane permeability [m/s]
    v_ret: float = 0.05 # Retentate velocity [m/s]                                                                               
    v_perm: float = 0.10 # Permeate velocity [m/s]  
    U_mem: float = 0.0 # overall heat-transfer coefficient [W/(m2 K)]
    thermal_conductivity_gas: float = 0.15  # W/(m K), rough estimate                                                                           
    
    # Operating conditions
    T: float = 573.15  # Operating temperature [K]
    R: float = 8.314  # Ideal gas constant [J/(mol·K)]
    p: float = 50e5  # Operating pressure [Pa]
    p_stand: float = 1e5 # standard pressure, 1 bar [Pa]
    mu: float = 2.0e-5  # Gas viscosity [Pa·s]
    
    # Catalyst and particle properties
    particle_radius: float = 1.0e-3  # Catalyst particle radius [m]
    d_p: float = 2.0e-3  # Catalyst particle diameter [m]
    eps_s: float = 0.4  # Catalyst/solid volume fraction in bed [-]
    eps_p: float = 0.5  # catalyst particle porosity [-]
    Mw_cat: float = 0.177 # Molar weight of the catalyst [kg/mol]
    rho_cat: float = 7215.0 # Catalyst density [kg/m³]
    tortuosity = 2 # tortuosity of catalyst [-]

    particle_diffusivity: np.ndarray = field(
        default_factory=lambda: np.array([2.0e-6, 1.5e-6, 1.5e-6, 1.5e-6, 1.5e-6]))  # Effective diffusivity per component [m²/s]
    
    heat_capacity_gas: np.ndarray = field(
        default_factory=lambda: np.array([48.139, 29.34, 108.95, 79.952, 462.0]))  # [J/(mol K)]
    
    MW: np.ndarray = field(
        default_factory=lambda: np.array([44.01e-3, 2.016e-3, 32.04e-3, 18.015e-3,90.08e-3]))  # Moleculat weight [kg/mol]
    
    thermal_conductivity: float = 1.0  # [W/(m K)], rough packed-bed effective value

    Cp_solid: float = 460.0  #  approximate CeO2 value [J/(kg K)] [??]
    rhoCp_floor: float = 1.0e3  # avoids division by zero for empty permeate
    n: float = 2 # reaction order
    
    # Kinetic Parameters: Reaction 1 (CO2 + 3H2 <-> MeOH + H2O) (Ghosh et al.)
    T_ref_1: float = 573.15  # Reference temperature [K] (300 °C)
    k1_pre: float = 6.9e-4  # Forward rate constant at T_ref [mol/(s·bar²·kg_cat)]
    k1_eq: float = 2.5e-4  # Equilibrium constant at T_ref [bar⁻²]
    Ea_1: float = 35.7e3  # Activation energy [J/mol]
    K_CO2_ref: float = 0.79  # CO2 adsorption constant at T_ref [bar⁻¹]
    K_H2_ref: float = 0.76  # H2 adsorption constant at T_ref [bar⁻¹]
    dH_CO2: float = -25.9e3  # Enthalpy of adsorption for CO2 [J/mol]
    dH_H2: float = -12.5e3  # Enthalpy of adsorption for H2 [J/mol]
    dH_r1: float = -49.5e3 # enthalpy of reaction [J/mol]
    r1_scale: float = 1.0 # optional scaling rate [-]

    # Kinetic Parameters: Reaction 2 (CO2 + 2MeOH <-> DMC + H2O) (ibrahim et al.)  
    T_ref_2: float = 298.15 # Reference temperature [K]
    k2_pre: float = 0.0133  # Pre-exponential factor [s⁻¹]
    k_ads1: float = 9 # adsorption parameter [-]
    k_ads2: float = 109 # adsorption parameter [-]
    dH_DMC: float = -20.10e3 # reaction enthalpy [J/mol]
    dG_DMC: float = 31.50e3 # Gibbs free energy change [J/mol]
    drC_p: float = -170.23
    Ea_DMC: float = 106e3  # Activation energy [J/mol]
    r2_scale: float = 1.0 # optional scaling rate [-]

    # Feed composition
    feed_y: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.75, 0.0, 0.0, 0.0])) # Feed mole fractions [-]
    
    # Solver settings
    method: str = "BDF"
    tol: float = 1.0e-6  # tolerance for convergence
    maxfev: int = 30    # maximum number of function evaluations

    # Derived properties
    @property # Residence time in retentate [s]
    def tau(self) -> float:
        return self.length/self.v_ret
    
    @property # Gas void fraction of the packed bed [-]
    def eps_bed(self) -> float:
        return 1.0 - self.eps_s
    
    @property
    def P_species(self) -> np.ndarray:
        return self.P_membrane[: self.n_species]
    
    @property # Inlet concentration per component [mol/m3]
    def inlet_concentration(self):
        feed_y = np.asarray(self.feed_y, dtype=float)
        feed_y = feed_y / np.sum(feed_y)

        c_total = self.p / (self.R * self.T)
        return c_total * feed_y
    
    @property
    def perm_inlet(self) -> np.ndarray:
        state = np.zeros(self.n_c)
        state[self.iT] = self.T
        return state
    
    @property
    def inlet_state(self) -> np.ndarray:
        state = np.zeros(self.n_c)
        state[: self.n_species] = self.inlet_concentration
        state[self.iT] = self.T
        return state

    @property # external paricle surface area per unit bed volume
    def external_area(self):
        return 3.0 * self.eps_s / self.particle_radius
    
    @property # Membrane area per retentate volume [m2/m3]
    def a_ret(self):
        return 2.0 / self.R_ret
    
    @property # Membrane area per permeate volume [m2/m3]
    def a_perm(self):
        return 2.0 * self.R_ret / (self.R_out**2 - self.R_perm**2)

    @property # Retentate tube volume [m3]
    def v_tube(self) -> float:
        return np.pi * self.R_ret**2 * self.length

    @property # Total catalyst mass in retentate tube [kg_cat]
    def m_cat(self) -> float:
        return self.rho_bulk * self.v_tube * self.eps_s * 1000
    
    @property # bulk concentration of the catalyst
    def rho_bulk(self) -> float:
        return self.eps_p * self.rho_cat
    
    @property # Feed gas density [kg/m3]
    def rho_gas(self) -> float:
        avg_Mw = np.sum(self.feed_y * self.MW)
        return self.p * avg_Mw / (self.R * self.T)
    
    @property # Reynolds number [-]
    def Reynolds(self) -> float:
        return self.rho_gas * self.v_ret * self.d_p / self.mu

    @property # Schmidt number [-]
    def Schmidt(self) -> np.ndarray:
        return self.mu / (self.rho_gas * self.particle_diffusivity)

    @property # axial dispersion coefficient [-]
    def d_ax(self) -> np.ndarray:
        Re = self.Reynolds
        Sc = self.Schmidt
        eps_g = 1 - self.eps_s

        D_ax = self.v_ret * self.d_p * (
            (0.73 * eps_g) / (eps_g + 0.5/(Re * Sc))
            + (0.5 / (1 + 9.7 * eps_g / (Re * Sc)))
        )
        return D_ax
    
    @property # pressure drop from the Ergun equation [Pa/m]
    def ergun_pressure_gradient(self) -> float:
        viscous_term = ((150.0 * self.mu * (1.0 - self.eps_bed)**2 * self.v_ret)
                / (self.eps_bed**3 * self.d_p**2))
        
        Inertial_term = ((1.75 * (1.0 - self.eps_bed) * self.rho_gas * self.v_ret**2)
                         / (self.eps_bed**3 * self.d_p))
        return viscous_term + Inertial_term
    
    @property # calculates the peclet number for mass transfer
    def mass_peclet(self) -> np.ndarray:
        Pe_m = self.Reynolds * self.Schmidt
        return Pe_m
    
    @property # estimated pressure outlet
    def pressure_outlet(self) -> float:
        return self.p - self.ergun_pressure_gradient * self.length
    
    @property # gass solid mass transfer coefficient
    def K_gs(self) -> float:
        return (self.particle_diffusivity[1] / self.d_p) * (2 + 1.1 * self.Reynolds**0.6 * self.Schmidt[1]**(1.3))
    
    @property # effecitve difussivity inside catalyst
    def D_eff(self) -> float:
        return (self.eps_p * self.particle_diffusivity[1]) / self.tortuosity
    
