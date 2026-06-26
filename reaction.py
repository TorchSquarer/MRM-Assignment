from __future__ import annotations
import numpy as np
from config import ModelConfig

# component order used throughout the model
SPECIES_LABELS = ("CO2", "H2", "CH3OH", "H2O", "DMC", "N2")

# stofhiometic coefficients
STOICH = np.array(
    [
        [-1.0, -3.0,  1.0,  1.0,  0.0, 0.0],
        [-1.0,  0.0, -2.0,  1.0,  1.0, 0.0],
    ]
)

# number to avoid division by zero
EPS = 1.0e-12

class ReactionRates:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        
    # Convert concentration to mole fractions
    def mole_fractions(self, c_species: np.ndarray) -> np.ndarray: 
        c_pos = np.maximum(np.asarray(c_species, dtype=float), 0.0)
        c_tot = np.maximum(np.sum(c_pos, axis=-1, keepdims=True), EPS)
        return c_pos / c_tot

    def partial_pressures(self, c_species: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = self.mole_fractions(c_species)
        P_pa = y * self.cfg.p
        P_bar = P_pa / 1.0e5
        return P_pa, P_bar, y

    # Temperature dependent adsorption constant
    def adsorption_constant(self, K_ref: float, dH: float, T: float | np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = np.maximum(np.asarray(T, dtype=float), 250.0)
        return K_ref * np.exp((dH / cfg.R) * (1.0 / cfg.T_ref_1 - 1.0 / T))
    
    # Effective rate constant for methanol formation
    def k_eff_r1(self, T: float | np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = np.maximum(np.asarray(T, dtype=float), 250.0)
        return cfg.k1_pre * np.exp(
            (cfg.Ea_1 / cfg.R) * (1.0 / cfg.T_ref_1 - 1.0 / T))

    # Temperature dependent equillibrium constant for DMC formation
    def k2_eq_T(self, T: float | np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = np.maximum(np.asarray(T, dtype=float), 250.0)

        lnK = (
            - cfg.dG_DMC / (cfg.R * cfg.T_ref_2)
            + cfg.dH_DMC / (cfg.R * cfg.T_ref_2) * (1 - cfg.T_ref_2 / T)
            - cfg.drC_p * (T - cfg.T_ref_2) / (cfg.R * T) 
            + cfg.drC_p / cfg.R * np.log(T / cfg.T_ref_2)
            )

        return np.exp(np.clip(lnK, -100.0, 100.0))
    
    # Effective rate constant for DMC formation
    def k_eff_r2(self, T: float | np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T = np.maximum(np.asarray(T, dtype=float), 250.0)
        exp = cfg.Ea_DMC / cfg.R * (1.0 / cfg.T_ref_2 - 1 / T)
        exp = np.clip(exp, -50.0, 50.0)
        return  cfg.k2_pre * np.exp(exp)


    # Reaction 1: Methanol formation
    def r1_methanol(self, c_species: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        cfg = self.cfg

        _P_pa, P_bar, _y = self.partial_pressures(c_species)

        P_CO2   = np.maximum(P_bar[..., 0], 0.0)
        P_H2    = np.maximum(P_bar[..., 1], 1e-8)
        P_CH3OH = np.maximum(P_bar[..., 2], 0.0)
        P_H2O   = np.maximum(P_bar[..., 3], 0.0)

        K_CO2 = self.adsorption_constant(cfg.K_CO2_ref, cfg.dH_CO2, T)
        K_H2 = self.adsorption_constant(cfg.K_H2_ref, cfg.dH_H2, T)

        driving_force = P_CO2 * P_H2**3 - (P_CH3OH * P_H2O) / max(cfg.k1_eq, EPS)
        inhibition = (1.0 + K_CO2 * P_CO2 + np.sqrt(np.maximum(K_H2 * P_H2, 0.0))) ** 2
        denominator = np.maximum(P_H2**2 * inhibition, EPS)

        r_mass = self.k_eff_r1(T) * driving_force / denominator
        r_vol = cfg.r1_scale * r_mass * cfg.rho_bulk * cfg.eps_s
        return np.nan_to_num(r_vol, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

    # Reaction 2: DMC formation
    def r2_dmc(self, c_species: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        cfg = self.cfg
        P_pa, _P_bar, _y = self.partial_pressures(c_species)

        P_CO2 = np.maximum(P_pa[..., 0], 0.0)
        P_CH3OH = np.maximum(P_pa[..., 2], 1e-8)
        P_H2O = np.maximum(P_pa[..., 3], 0.0)
        P_DMC = np.maximum(P_pa[..., 4], 0.0)

        K_eq = np.maximum(self.k2_eq_T(T), EPS)
        k2 = self.k_eff_r2(T)

        driving_force = ((P_CO2 / cfg.p_stand) * (P_CH3OH / cfg.p_stand) ** 2
                        - ((P_DMC / cfg.p_stand) * (P_H2O / cfg.p_stand)) / K_eq)
        

        denominator = (1.0 + cfg.k_ads1 * (P_CH3OH / cfg.p_stand) 
                        + cfg.k_ads2 * (P_CO2 / cfg.p_stand))**3

        r_mass = (k2 * cfg.m_cat * driving_force / denominator)
        r_vol = cfg.r2_scale * r_mass * cfg.rho_bulk / cfg.Mw_cat
        return np.nan_to_num(r_vol, nan=0.0, posinf=1.0e20, neginf=-1.0e20)
    
    #combined reaction rates and source terms
    def reaction_rates(self, c_species: np.ndarray, T: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.r1_methanol(c_species, T), self.r2_dmc(c_species, T)

    # converts reaction rates into component source terms
    def species_source(self, c_species: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        r1, r2 = self.reaction_rates(c_species, T)
        rates = np.stack([r1, r2], axis=-1)
        return np.einsum("...r,rs->...s", rates, STOICH)

    def heat_source(self, c_species: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        r1, r2 = self.reaction_rates(c_species, T)
        return -(self.cfg.dH_r1 * r1 + self.cfg.dH_DMC * r2)