import numpy as np
from config import ModelConfig

STOICH = np.array(
    [
        [-1.0, -3.0,  1.0,  1.0,  0.0],  # R1: CO2 + 3H2 <-> CH3OH + H2O
        [-1.0,  0.0, -2.0,  1.0,  1.0],  # R2: CO2 + 2CH3OH <-> DMC + H2O
    ]
)

SPECIES_LABELS = ("CO2", "H2", "CH3OH", "H2O", "DMC")

class ReactionRates:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def reaction_rates(self, c_p: np.ndarray, cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
        RT = cfg.R * cfg.T
        P_Pa = c_p * RT                  # partial pressures [Pa], shape (n_z, n_r_ret, n_c)

        P_CO2 = P_Pa[..., 0]
        P_H2 = P_Pa[..., 1]
        P_CH3OH = P_Pa[..., 2]
        P_H2O = P_Pa[..., 3]
        P_DMC = P_Pa[..., 4]
        P_total_Pa = P_Pa.sum(axis=-1)      

        # R1: CO2 + 3H2 <-> CH3OH + H2O
        alpha_1 = P_CO2 * P_H2**3 - (P_CH3OH * P_H2O) / cfg.k1_eq
        inhibition = (1.0 + cfg.K_ads(cfg.K_CO2_ref, cfg.dH_CO2) * P_CO2 + np.sqrt(cfg.K_ads(cfg.K_H2_ref, cfg.dH_H2) * P_H2)) ** 2
        r1 = np.zeros_like(alpha_1)
        r1 = (cfg.k_eff_r1() * alpha_1 / (P_H2 ** 2 * inhibition)) 
        r1 = r1 * cfg.rho_bulk  # [mol/s/bar²/kg_cat] → [mol/s/m³_reactor]   

        # R2: CO2 + 2CH3OH <-> DMC + H2O  (Ibrahim et al., Eq. 6, no adsorption terms)
        eps = 1e-10 * cfg.p_stand  # small pressure floor

        alpha_2 = P_DMC * P_H2O / (cfg.k2_eq * cfg.p_stand)
        inhibition = (1 + cfg.k_ads1 * (P_CH3OH / cfg.p_stand) 
               + cfg.k_ads2 * (P_CH3OH / cfg.p_stand) * (P_CO2 / cfg.p_stand))

        P_CH3OH_safe = np.maximum(P_CH3OH, eps)
        r2 = cfg.m_cat * cfg.k_eff_r2(P_total_Pa) * (P_CO2 * P_CH3OH**2 - alpha_2) / ((P_CH3OH_safe / cfg.p_stand) * inhibition)
        return r1, r2

    def particle_reaction_rates(self, c_p: np.ndarray, cfg: ModelConfig) -> np.ndarray:
        r1, r2 = self.reaction_rates(c_p, cfg)
        rates = np.stack([r1, r2], axis=-1)             
        return np.einsum("zrc,co->zro", rates, STOICH)