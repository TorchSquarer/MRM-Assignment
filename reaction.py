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

from config import ModelConfig

cfg = ModelConfig()


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
        P_bar = P_Pa /1e5

        P_CO2 = P_bar[..., 0]
        P_H2 = P_bar[..., 1]
        P_CH3OH = P_bar[..., 2]
        P_H2O = P_bar[..., 3]
        P_DMC = P_bar[..., 4]
        P_total_Pa = P_Pa.sum(axis=-1)      

        # R1: CO2 + 3H2 <-> CH3OH + H2O
        eps = 1e-8
        P_H2_safe = np.maximum(P_H2,eps)

        alpha_1 = P_CO2 * P_H2_safe**3 - (P_CH3OH * P_H2O) / cfg.k1_eq
        k_ads_co2 = cfg.K_ads(cfg.K_CO2_ref, cfg.dH_CO2)
        k_ads_h2  = cfg.K_ads(cfg.K_H2_ref, cfg.dH_H2)
        inhibition_r1 = (1.0 + k_ads_co2 * P_CO2 + np.sqrt(k_ads_h2 * P_H2_safe)) ** 2
        r1 = cfg.rho_bulk * (cfg.k_eff_r1() * alpha_1) / (P_H2_safe**2 * inhibition_r1)

        # R2: CO2 + 2CH3OH <-> DMC + H2O  (Ibrahim et al., Eq. 6, no adsorption terms)
        alpha_2 = P_CO2 * P_CH3OH**2 - (P_DMC * P_H2O) / cfg.k_eq_r2()
        inhibition_r2 = (1.0 + cfg.k_ads1 * P_CH3OH + cfg.k_ads2 * P_CH3OH * P_CO2) ** 2
        r2 = cfg.m_cat * cfg.k2_pre * alpha_2 / inhibition_r2
        return r1, r2

    def particle_reaction_rates(self, c_p: np.ndarray, cfg: ModelConfig) -> np.ndarray:
        r1, r2 = self.reaction_rates(c_p, cfg)
        rates = np.stack([r1, r2], axis=-1)             
        return np.einsum("zrc,co->zro", rates, STOICH)