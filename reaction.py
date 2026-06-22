from __future__ import annotations
import numpy as np
from config import ModelConfig

# component order used throughout the model
SPECIES_LABELS = ("CO2", "H2", "CH3OH", "H2O", "DMC")

# stofhiometic coefficients
STOICH = np.array(
    [
        [-1.0, -3.0,  1.0,  1.0,  0.0],
        [-1.0,  0.0, -2.0,  1.0,  1.0],
    ]
)

# number to avoid division by zero
EPS = 1.0e-30

class ReactionRates:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        

    # Composistion and pressure helpers
    # Convert concentration to mole fractions
    def mole_fractions(self, c: np.ndarray) -> np.ndarray: 
        c_pos = np.maximum(c, 0.0)
        c_tot = np.maximum(np.sum(c_pos, axis=-1, keepdims=True), EPS)
        return c_pos / c_tot

    def partial_pressures(self, c: np.ndarray, cfg: ModelConfig|None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg is None:
            cfg = self.cfg

        y = self.mole_fractions(c)

        P_total_pa = np.asarray(cfg.p, dtype=float)
        P_pa = y * P_total_pa
        P_bar = P_pa / 1.0e5

        return P_pa, P_bar, y

    # Temperature dependent adsorption constant
    def adsorption_constant(self, K_ref: float, dH: float, cfg: ModelConfig, T: float|np.ndarray|None=None) -> float|np.ndarray:
        if T is None:
            T = cfg.T
        return K_ref * np.exp((dH / cfg.R) * (1.0 / cfg.T_ref - 1.0 / T))
    
    # Effective rate constant for methanol formation
    def k_eff_r1(self, cfg: ModelConfig, T: float|np.ndarray|None=None,) -> float|np.ndarray:
        if T is None:
            T = cfg.T
        return cfg.k1_pre * np.exp(
            (cfg.Ea_1 / cfg.R) * (1.0 / cfg.T_ref - 1.0 / T))

    # Temperature dependent equillibrium constant for DMC formation
    def k2_eq_T(self, cfg: ModelConfig, T: float|np.ndarray|None=None,) -> float|np.ndarray:
        if T is None:
            T = cfg.T

        K = (
            -cfg.dG_DMC / (cfg.R * cfg.T_ref_DMC)
            + (cfg.dH_DMC / cfg.R) * (1.0 / cfg.T_ref_DMC - 1.0 / T)
            )
        return np.exp(K) 
    
    # Effective rate constant for DMC formation
    def k_eff_r2(self, cfg: ModelConfig, P_total_pa: float|np.ndarray|None=None, T: float|np.ndarray|None=None,) -> float | np.ndarray:
        if T is None:
            T = cfg.T

        if P_total_pa is None:
            P_total_pa = cfg.p

        # Arrhenius temperature term
        exponent_T = -cfg.Ea_DMC / (cfg.R * T)
        exponent_T = np.clip(exponent_T, -700.0, 700.0)

        k_min = cfg.k2_pre * np.exp(exponent_T)

        # Pressure correction term
        exponent_P = -cfg.dV * (P_total_pa - cfg.p_0) / (cfg.R * T)
        exponent_P = np.clip(exponent_P, -700.0, 700.0)

        k_min *= np.exp(exponent_P)

        # Convert from min^-1 to s^-1
        return k_min / 60.0


    # Reaction 1: Methanol formation
    def r1_methanol(self, c: np.ndarray, cfg: ModelConfig|None=None, T: float|np.ndarray|None=None) -> np.ndarray:
        if cfg is None:
            cfg = self.cfg
        if T is None:
            T = self.cfg.T

        _P_pa, P_bar, _y = self.partial_pressures(c, cfg)

        P_CO2   = np.maximum(P_bar[..., 0], 0.0)
        P_H2    = np.maximum(P_bar[..., 1], 1e-8)
        P_CH3OH = np.maximum(P_bar[..., 2], 0.0)
        P_H2O   = np.maximum(P_bar[..., 3], 0.0)

        K_CO2 = self.adsorption_constant(cfg.K_CO2_ref, cfg.dH_CO2, cfg, T)
        K_H2  = self.adsorption_constant(cfg.K_H2_ref,  cfg.dH_H2,  cfg, T)

        driving_force = (P_CO2 * P_H2**3 - (P_CH3OH * P_H2O) / np.maximum(cfg.k1_eq, 1.0e-12))

        inhibition = (1.0 + K_CO2 * P_CO2 + np.sqrt(np.maximum(K_H2 * P_H2, 0.0))) ** 2
        denominator = P_H2**2 * np.maximum(inhibition, 1e-12)
    
        r1_mass = self.k_eff_r1(cfg) * driving_force / denominator
        r1_vol = cfg.r1_scale * r1_mass * cfg.rho_bulk
        return np.nan_to_num(r1_vol, nan=0.0, posinf=1.0e-8, neginf=-1.0e-8)

    # Reaction 2: DMC formation
    def r2_dmc(self, c: np.ndarray, cfg: ModelConfig|None=None, T: float|np.ndarray|None=None) -> np.ndarray:
        if cfg is None:
            cfg = self.cfg
        if T is None:
            T = self.cfg.T

        P_pa, _P_bar, _y = self.partial_pressures(c, cfg)

        P_CO2   = np.maximum(P_pa[..., 0], 0.0)
        P_CH3OH = np.maximum(P_pa[..., 2], 1e-3)
        P_H2O   = np.maximum(P_pa[..., 3], 0.0)
        P_DMC   = np.maximum(P_pa[..., 4], 0.0)

        K_eq = self.k2_eq_T(cfg, T)
        k2 = self.k_eff_r2(cfg, T)

        driving_force = ((P_CO2/cfg.p_stand) * (P_CH3OH/cfg.p_stand)**2 - ((P_DMC/cfg.p_stand) * (P_H2O/cfg.p_stand)) / np.maximum(K_eq, EPS))
        denominator = np.maximum(1.0 + cfg.k_ads1 * (P_CO2/cfg.p_stand) + cfg.k_ads2 * (P_CH3OH/cfg.p_stand), EPS) ** 3

        rho_cat_g_m3 = cfg.rho_bulk * 1000.0

        c_ref = np.minimum(np.maximum(c[..., 0], 0.0),
                           np.maximum(c[..., 2], 0.0) / 2.0)

        r2_vol = cfg.r2_scale * c_ref * rho_cat_g_m3 * k2 * driving_force / denominator
        return np.nan_to_num(r2_vol, nan=0.0, posinf=1.0e30, neginf=-1.0e30)

    #combined reaction rates and source terms
    def reaction_rates(self, c: np.ndarray, cfg: ModelConfig|None=None) -> tuple[np.ndarray, np.ndarray]:
        if cfg is None:
            cfg = self.cfg
            
        r1 = self.r1_methanol(c[...,:-1], cfg, c[...,-1]) # Methanol formation rate [mol/m3 s]
        r2 = self.r2_dmc(c[...,:-1], cfg, c[...,-1]) # DMC formation rate [mol/m3 s]

        return r1, r2

    # converts reaction rates into component source terms
    def particle_reaction_rates(self, c_p: np.ndarray, cfg: ModelConfig|None=None) -> np.ndarray:
        if cfg is None:
            cfg = self.cfg

        r1, r2 = self.reaction_rates(c_p, cfg)
        rates = np.stack([r1, r2], axis=-1)  
        source = np.einsum("...r,rs->...s", rates, STOICH)

        heat = np.zeros(source.shape[:-1] + (1,))
        return np.concatenate([source, heat], axis=-1)
    
   