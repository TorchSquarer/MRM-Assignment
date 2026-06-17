from __future__ import annotations

import numpy as np
from config import ModelConfig


SPECIES_LABELS = ("CO2", "H2", "CH3OH", "H2O", "DMC")

STOICH = np.array(
    [
        [-1.0, -3.0,  1.0,  1.0,  0.0],
        [-1.0,  0.0, -2.0,  1.0,  1.0],
    ]
)

EPS = 1.0e-30

class ReactionRates:
    def __init__(self, cfg: ModelConfig, r2_mechanism: str = "LH") -> None:
        self.cfg = cfg
        self.r2_mechanism = r2_mechanism.upper()

    def mole_fractions(self, c: np.ndarray) -> np.ndarray:
        c_pos = np.maximum(c, 0.0)
        c_tot = np.maximum(np.sum(c_pos, axis=-1, keepdims=True), EPS)
        return c_pos / c_tot

    def partial_pressures(self, c: np.ndarray, cfg: ModelConfig | None = None):
        if cfg is None:
            cfg = self.cfg

        y = self.mole_fractions(c)

        # Isobaric pressure from config.py
        P_total_pa = np.asarray(cfg.p, dtype=float)

        # Broadcast pressure to match concentration shape
        P_pa = y * P_total_pa
        P_bar = P_pa / 1.0e5

        return P_pa, P_bar, y

    def adsorption_constant(self, K_ref: float, dH: float, cfg: ModelConfig) -> float:
        return K_ref * np.exp((dH / cfg.R) * (1.0 / cfg.T_ref - 1.0 / cfg.T))

    def r1_methanol(self, c: np.ndarray, cfg: ModelConfig | None = None) -> np.ndarray:
        if cfg is None:
            cfg = self.cfg

        _P_pa, P_bar, _y = self.partial_pressures(c, cfg)

        P_CO2   = np.maximum(P_bar[..., 0], 0.0)
        P_H2    = np.maximum(P_bar[..., 1], EPS)
        P_CH3OH = np.maximum(P_bar[..., 2], 0.0)
        P_H2O   = np.maximum(P_bar[..., 3], 0.0)

        K_CO2 = self.adsorption_constant(cfg.K_CO2_ref, cfg.dH_CO2, cfg)
        K_H2  = self.adsorption_constant(cfg.K_H2_ref,  cfg.dH_H2,  cfg)

        driving_force = (P_CO2 * P_H2**3 - (P_CH3OH * P_H2O) / np.maximum(cfg.k1_eq, EPS))

        inhibition = (1.0 + K_CO2 * P_CO2 + np.sqrt(np.maximum(K_H2 * P_H2, 0.0))) ** 2
        denominator = P_H2**2 * np.maximum(inhibition, EPS)
        # Rate per kg catalyst
        r1_mass = cfg.k_eff_r1() * driving_force / denominator
        # Convert to rate per reactor volume
        r1_vol = cfg.r1_scale * r1_mass * cfg.rho_bulk

        return np.nan_to_num(r1_vol, nan=0.0, posinf=1.0e30, neginf=-1.0e30)

    def r2_dmc(self, c: np.ndarray, cfg: ModelConfig | None = None) -> np.ndarray:
        if cfg is None:
            cfg = self.cfg

        P_pa, _P_bar, _y = self.partial_pressures(c, cfg)

        P_CO2   = np.maximum(P_pa[..., 0], 0.0)
        P_CH3OH = np.maximum(P_pa[..., 2], EPS)
        P_H2O   = np.maximum(P_pa[..., 3], 0.0)
        P_DMC   = np.maximum(P_pa[..., 4], 0.0)

        K_eq = cfg.k2_eq_T()

        driving_force = ((P_CO2/cfg.p_stand) * (P_CH3OH/cfg.p_stand)**2 - ((P_DMC/cfg.p_stand) * (P_H2O/cfg.p_stand)) / np.maximum(K_eq, EPS))
        denominator = np.maximum(1.0 + cfg.k_ads1 * (P_CO2/cfg.p_stand) + cfg.k_ads2 * (P_CH3OH/cfg.p_stand), EPS) ** 3
        k2 = cfg.k_eff_r2(cfg.p)  # [1/s] after min^-1 to s^-1 conversion

        rho_cat_g_m3 = cfg.rho_bulk * 1000.0

        c_ref = np.minimum(np.maximum(c[..., 0], 0.0),
                           np.maximum(c[..., 2], 0.0) / 2.0)

        r2_vol = cfg.r2_scale * c_ref * rho_cat_g_m3 * k2 * driving_force / denominator
        return np.nan_to_num(r2_vol, nan=0.0, posinf=1.0e30, neginf=-1.0e30)


    def reaction_rates(self,c: np.ndarray, cfg: ModelConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
        if cfg is None:
            cfg = self.cfg
        r1 = self.r1_methanol(c, cfg)
        r2 = self.r2_dmc(c, cfg)

        return r1, r2

    def particle_reaction_rates(self, c_p: np.ndarray, cfg: ModelConfig | None = None) -> np.ndarray:
        if cfg is None:
            cfg = self.cfg

        r1, r2 = self.reaction_rates(c_p, cfg)
        rates = np.stack([r1, r2], axis=-1)  # [..., n_reactions]
        source = np.einsum("...r,rs->...s", rates, STOICH)

        return source
    
    def reaction_diagnostics(self, c: np.ndarray, cfg: ModelConfig | None = None) -> dict:
        """
        Diagnostic terms for the two reaction-rate expressions.

        Returns forward term, reverse term, reverse/forward ratio,
        driving force and denominator for R1 and R2.
        This does not change the model; it only helps interpret it.
        """
        if cfg is None:
            cfg = self.cfg

        P_pa, P_bar, _y = self.partial_pressures(c, cfg)

    # ---------- R1: CO2 + 3H2 <-> CH3OH + H2O ----------
        P_CO2   = np.maximum(P_bar[..., 0], 0.0)
        P_H2    = np.maximum(P_bar[..., 1], EPS)
        P_CH3OH = np.maximum(P_bar[..., 2], 0.0)
        P_H2O   = np.maximum(P_bar[..., 3], 0.0)

        K_CO2 = self.adsorption_constant(cfg.K_CO2_ref, cfg.dH_CO2, cfg)
        K_H2  = self.adsorption_constant(cfg.K_H2_ref,  cfg.dH_H2,  cfg)

        r1_forward = P_CO2 * P_H2**3
        r1_reverse = (P_CH3OH * P_H2O) / np.maximum(cfg.k1_eq, EPS)
        r1_driving = r1_forward - r1_reverse

        r1_inhibition = (1.0 + K_CO2 * P_CO2 + np.sqrt(np.maximum(K_H2 * P_H2, 0.0))) ** 2
        r1_denominator = P_H2**2 * np.maximum(r1_inhibition, EPS)

    # ---------- R2: CO2 + 2CH3OH <-> DMC + H2O ----------
        # ---------- R2: CO2 + 2CH3OH <-> DMC + H2O ----------
        P_CO2   = np.maximum(P_pa[..., 0], 0.0)
        P_CH3OH = np.maximum(P_pa[..., 2], EPS)
        P_H2O   = np.maximum(P_pa[..., 3], 0.0)
        P_DMC   = np.maximum(P_pa[..., 4], 0.0)

        theta_CO2   = P_CO2 / cfg.p_stand
        theta_CH3OH = P_CH3OH / cfg.p_stand

        if hasattr(cfg, "k2_eq_T"):
            K_eq = cfg.k2_eq_T()
        else:
            K_eq = cfg.k2_eq

        r2_forward = P_CO2 * P_CH3OH**2
        r2_reverse = (P_DMC * P_H2O * cfg.p_stand) / np.maximum(K_eq, EPS)
        r2_driving = r2_forward - r2_reverse

        r2_adsorption_sum = 1.0 + cfg.k_ads1 * theta_CO2 + cfg.k_ads2 * theta_CH3OH
        r2_denominator = np.maximum(r2_adsorption_sum, EPS) ** 3
        return {
            "r1_forward": r1_forward,
            "r1_reverse": r1_reverse,
            "r1_reverse_over_forward": r1_reverse / np.maximum(r1_forward, EPS),
            "r1_driving": r1_driving,
            "r1_denominator": r1_denominator,

            "r2_forward": r2_forward,
            "r2_reverse": r2_reverse,
            "r2_reverse_over_forward": r2_reverse / np.maximum(r2_forward, EPS),
            "r2_driving": r2_driving,
            "r2_denominator": r2_denominator,
        }