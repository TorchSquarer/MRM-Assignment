from __future__ import annotations  
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass

from config import ModelConfig
from buildoperators import TransportOperators
from reaction import ReactionRates, SPECIES_LABELS, STOICH

for candidate in (Path.cwd(), Path.cwd().parent):
    pymrm_src = candidate / "pymrm" / "src"
    if pymrm_src.exists() and str(pymrm_src) not in sys.path:
        sys.path.insert(0, str(pymrm_src))

from pymrm import (
    NumJac,
    newton,
)

@dataclass
class SimpleResult:
    success: bool
    message: str
    nfev: int
    x: np.ndarray

# Full membrane reactor model
class MembraneReactorModel:
    species_labels = SPECIES_LABELS
    stoich = STOICH

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.ops = TransportOperators(cfg)
        self.reac = ReactionRates(cfg)

        # State array shapes
        self.shape = (cfg.n_z, 2, cfg.n_c)

        # Grid coordinates
        self.z_c    = self.ops.z_c

        self.u0 = self._initial_state()
        self.y0 = self._initial_state().ravel()
        self.u  = np.zeros(self.shape)
        self.result: SimpleResult | None = None

    # Build initial concentration field and temperature field
    def _initial_state(self) -> np.ndarray:
        cfg = self.cfg
        u0 = np.zeros((2, cfg.n_c), dtype=float)
        # Initial Concentration Disctribution
        u0[0, :] = cfg.inlet_state
        u0[1, :]  = cfg.perm_inlet
        return u0.ravel()

    # split full state array into model regions:
    # c_ret is retentate bulk gas, c_perm is permeate concentration [mol/m3]
    def split_state(self, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
        cfg = self.cfg
        y = np.asarray(y, dtype=float).reshape(2, cfg.n_c)

        ret = y[0, :]
        perm = y[1, :]

        c_ret = np.maximum(ret[: cfg.n_species], 0.0)
        T_ret = float(max(ret[cfg.iT], 250.0))

        c_perm = np.maximum(perm[: cfg.n_species], 0.0)
        T_perm = float(max(perm[cfg.iT], 250.0))

        return c_ret, T_ret, c_perm, T_perm
    
    def reaction_source(self, c_ret: np.ndarray, T: float,) -> tuple[np.ndarray, float, float, float]:
        cfg = self.cfg

        c_2d = np.asarray(c_ret, dtype=float).reshape(1, cfg.n_species)
        T_1d = np.array([T], dtype=float)

        source = self.reac.species_source(c_2d, T_1d)[0]
        q_rxn = float(self.reac.heat_source(c_2d, T_1d)[0])

        r1, r2 = self.reac.reaction_rates(c_2d, T_1d)

        return source, q_rxn, float(r1[0]), float(r2[0])

    def rhoCp_retentate(self, c_ret: np.ndarray) -> float:
        cfg = self.cfg

        c_ret = np.asarray(c_ret, dtype=float)
        gas = cfg.eps_bed * float(c_ret @ cfg.heat_capacity_gas)
        solid = cfg.rho_bulk * cfg.Cp_solid

        return max(gas + solid, cfg.rhoCp_floor)


    def rhoCp_permeate(self, c_perm: np.ndarray) -> float:
        cfg = self.cfg

        c_perm = np.asarray(c_perm, dtype=float)
        gas = float(c_perm @ cfg.heat_capacity_gas)

        return max(gas, cfg.rhoCp_floor)

    # Evaluates nonlinear residual calues
    def residual_values(self, y: np.ndarray) -> np.ndarray:
        cfg = self.cfg

        c_ret, T_ret, c_perm, T_perm = self.split_state(y)

        source_ret, q_ret, _r1_ret, _r2_ret = self.reaction_source(c_ret, T_ret)
        source_perm = np.zeros(cfg.n_species)
        q_perm = 0.0

        P_species = np.asarray(cfg.P_species, dtype=float)

        J_mem = np.zeros(cfg.n_species, dtype = float)
        mask = P_species != 0.0

        ctot_ret = np.sum(c_ret)
        ctot_perm = np.sum(c_perm)

        y_ret = c_ret/ctot_ret
        y_perm = c_perm/ctot_perm

        c_mem_ret = y_ret * cfg.p/(cfg.R*T_ret)
        c_mem_perm = y_perm * cfg.p_perm/(cfg.R*T_perm)

        J_mem[mask] = P_species[mask] * (c_mem_ret[mask]- c_mem_perm[mask])

        # Positive q_mem means heat retentate -> permeate
        q_mem = cfg.U_mem * (T_ret - T_perm)  # [W/m2]

        rhoCp_ret = self.rhoCp_retentate(c_ret)
        rhoCp_perm = self.rhoCp_permeate(c_perm)

        dy = np.zeros((2, cfg.n_c), dtype=float)

        # Retentate species balances:
        # v_ret dc_ret/dz = reaction source - membrane removal
        dy[0, :cfg.n_species] = (source_ret - cfg.a_ret * J_mem) / cfg.v_ret

        # Retentate energy balance:
        # v_ret rhoCp_ret dT_ret/dz = reaction heat - heat loss through membrane
        dy[0, cfg.iT] = (q_ret - cfg.a_ret * q_mem) / (cfg.v_ret * rhoCp_ret)

        # Permeate species balances:
        # v_perm dc_perm/dz = membrane addition
        dy[1, :cfg.n_species] = (source_perm + cfg.a_perm * J_mem) / cfg.v_perm

        # Permeate energy balance:
        # v_perm rhoCp_perm dT_perm/dz = heat received through membrane
        dy[1, cfg.iT] = (q_perm + cfg.a_perm * q_mem) / (cfg.v_perm * rhoCp_perm)

        return np.nan_to_num(dy.reshape(-1), nan=0.0, posinf=1.0e20, neginf=-1.0e20)

    def rhs(self, z: float, y: np.ndarray) -> np.ndarray:
        return self.residual_values(y)

    # Solves the nonlinear Steady-state model
    def solve(self) -> SimpleResult:
        cfg = self.cfg

        y0 = self._initial_state()
        z_eval = self.z_c
       
        z_eval[0] = 0.0
        z_eval[-1] = cfg.length

        sol = solve_ivp(self.rhs, 
                        t_span=(0.0, cfg.length), 
                        y0=y0, method=cfg.method, 
                        t_eval=z_eval, 
                        rtol=cfg.tol,
                        atol=cfg.tol*1e-3,
                        )
        
        if sol.y.shape[1] != cfg.n_z:
            # Fall back to interpolation if the solver returns a different grid.
            y_grid = np.vstack([np.interp(z_eval, sol.t, sol.y[i, :]) for i in range(sol.y.shape[0])])
        else:
            y_grid = sol.y

        u = np.zeros(self.shape)
        for i in range(cfg.n_z):
            y = y_grid[:, i]
            u[i, 0, :] = y[: cfg.n_c]
            u[i, 1, :] = y[cfg.n_c :]

        # Remove tiny negative numerical noise in concentrations, but not in temperature.
        u[:, :, : cfg.n_species] = np.maximum(u[:, :, : cfg.n_species], 0.0)
        u[:, :, cfg.iT] = np.maximum(u[:, :, cfg.iT], 250.0)

        self.u = u
        self.result = SimpleResult(
            success=bool(sol.success),
            message=str(sol.message),
            nfev=int(sol.nfev),
            x=u.ravel(),
        )
        return self.result
    
    # Return solved concentration fields
    def fields(self,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        c_ret = self.u[:, 0, : cfg.n_species]
        T_ret = self.u[:, 0, cfg.iT]
        c_perm = self.u[:, 1, : cfg.n_species]
        T_perm = self.u[:, 1, cfg.iT]
        return c_ret, T_ret, c_perm, T_perm
    
    def outlet_values(self) -> dict[str, np.ndarray | float]:
        c_ret, T_ret, c_perm, T_perm = self.fields()
        return {
            "c_ret_out": c_ret[-1, :].copy(),
            "T_ret_out": float(T_ret[-1]),
            "c_perm_out": c_perm[-1, :].copy(),
            "T_perm_out": float(T_perm[-1]),
        }

    def conversions(self) -> dict[str, float]:
        c_ret, _T_ret, _c_perm, _T_perm = self.fields()
        c_in = self.cfg.inlet_concentration
        out = c_ret[-1, :]
        return {
            "X_CO2_retentate": float((c_in[0] - out[0]) / max(c_in[0], 1.0e-30)),
            "X_H2_retentate": float((c_in[1] - out[1]) / max(c_in[1], 1.0e-30)),
        }
    
    # calculates mears criterian for the external mass transfer
    def mears_criterion(self):
        cfg = self.cfg

        c_ret, T_ret, _c_perm, _T_perm = self.fields()

        r1, r2 = self.reac.reaction_rates(c_ret, T_ret)

        C_CO2 = np.maximum(c_ret[:, 0], 1.0e-30)

        mears_r1 = np.abs(r1) * cfg.particle_radius * cfg.n / (cfg.K_gs * C_CO2)
        mears_r2 = np.abs(r2) * cfg.particle_radius * cfg.n / (cfg.K_gs * C_CO2)

        return mears_r1, mears_r2
    
    # calculates weisz-prater criterion for the internal mass transfer
    def weisz_prater_criterion(self):
        cfg = self.cfg

        c_ret, T_ret, _c_perm, _T_perm = self.fields()

        r1, r2 = self.reac.reaction_rates(c_ret, T_ret)

        C_CO2 = np.maximum(c_ret[:, 0], 1.0e-30)

        D_eff_CO2 = cfg.eps_p * cfg.particle_diffusivity[0] / cfg.tortuosity

        wp_r1 = np.abs(r1) * cfg.particle_radius**2 / (D_eff_CO2 * C_CO2)
        wp_r2 = np.abs(r2) * cfg.particle_radius**2 / (D_eff_CO2 * C_CO2)

        return wp_r1, wp_r2
    
    
    # calculates the peclet number for heat transfer
    def thermal_peclet(self):
        cfg = self.cfg
        c_ret, T_ret, c_perm, T_perm = self.fields()
        Pe_T = cfg.rho_gas * self.rhoCp_retentate(c_ret, cfg) * cfg.v_ret * cfg.length / cfg.thermal_conductivity
        return Pe_T