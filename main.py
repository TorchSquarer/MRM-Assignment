from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   18,
    'axes.labelsize':   16,
    'xtick.labelsize':  13,
    'ytick.labelsize':  13,
    'legend.fontsize':  13,
    'figure.titlesize': 20,
})
import pandas as pd

from config import ModelConfig
from membraneReactorModel import MembraneReactorModel
from reaction import SPECIES_LABELS

cfg = ModelConfig()

def run_simulation(T=None, p=None, P_membrane=None, v_ret=None):
    cfg = ModelConfig()

    if T is not None:
        cfg.T = T

    if p is not None:
        cfg.p = p

    if P_membrane is not None:
        cfg.P_membrane = P_membrane

    if v_ret is not None:
        cfg.v_ret = v_ret

    model = MembraneReactorModel(cfg)
    model.solve()

    c_ret, T_ret, c_perm, T_perm = model.fields()

    z = model.z_c
    c_CO2_in = cfg.inlet_concentration[0]
    c_DMC = c_ret[:, 4]

    Y_DMC = c_DMC / c_CO2_in
    Y_DMC_percent = Y_DMC * 100

    return z, Y_DMC_percent

temperatures_K = [423.15, 498.15, 573.15, 598.15, 623.15]

plt.figure(figsize=(7, 5))

for T_K in temperatures_K:
    z, Y_DMC_percent = run_simulation(T=T_K, p=200e5, P_membrane=cfg.P_membrane)
    plt.plot(z, Y_DMC_percent, label=f"{T_K} K")

plt.xlabel("Reactor length z [m]")
plt.ylabel("Y_DMC [%]")
plt.title("Influence of temperature on DMC yield")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

pressures_bar = [25, 50, 100, 150, 200]
pressures_Pa = [p_bar * 1e5 for p_bar in pressures_bar]

plt.figure(figsize=(7, 5))

for p_bar, p_Pa in zip(pressures_bar, pressures_Pa):
    z, Y_DMC_percent = run_simulation(T=573.15, p=p_Pa, P_membrane=cfg.P_membrane)
    plt.plot(z, Y_DMC_percent, label=f"{p_bar} bar")

plt.xlabel("Reactor length z [m]")
plt.ylabel("Y_DMC [%]")
plt.title("Influence of pressure on DMC yield")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

P_H2O_values = [0.0, 0.001, 0.0025, 0.0048, 0.0075, 0.01]

plt.figure(figsize=(7, 5))

for P_H2O in P_H2O_values:
    P_membrane = np.array([0, 0, 0, P_H2O, 0, 0])
    z, Y_DMC_percent = run_simulation(T=573.15, p=200e5, P_membrane=P_membrane)
    plt.plot(z, Y_DMC_percent, label = f"{P_H2O} m/s")

plt.xlabel("Reactor length z [m]")
plt.ylabel("Y_DMC [%]")
plt.title("Influence of membrane permeability on DMC yield")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

velocity = [0.01, 0.02, 0.05, 0.10, 0.15]

plt.figure(figsize=(7, 5))

for v_ret in velocity:
    z, Y_DMC_percent = run_simulation(T=573.15, p=200e5, P_membrane=cfg.P_membrane, v_ret=v_ret)
    plt.plot(z, Y_DMC_percent, label=f"{v_ret} m/s")

plt.xlabel("Reactor length z [m]")
plt.ylabel("Y_DMC [%]")
plt.title("Influence of velocity on DMC yield")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

cfg = ModelConfig()

model = MembraneReactorModel(cfg)
model.solve()

c_ret, T_ret, c_perm, T_perm = model.fields()

z = model.z_c

species = list(model.species_labels)

plt.figure(figsize=(8,5))
plt.plot(z, T_ret, label="T retentate")
plt.plot(z, T_perm, label="T permeate")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profiles")
plt.grid(True)

plt.show()

print("pressure drop", cfg.ergun_pressure_gradient)
print("final pressure", cfg.pressure_outlet)

# fig, ax = plt.subplots(1, 2, figsize=(12, 8))

# for i, sp in enumerate(species):
#     ax[0].plot(z, c_ret[:,i], label=sp)
# ax[0].set_xlabel("Reactor length z [m]")
# ax[0].set_ylabel("Concentration [mol/m³]")
# ax[0].set_title("retenate concentration concentrations")
# ax[0].grid(True)
# ax[0].legend()

# c_CO2_in = cfg.inlet_concentration[0]
# c_DMC = c_ret[:, 4]
# Y_DMC = c_DMC / c_CO2_in
# Y_DMC_percent = Y_DMC * 100

# ax[1].plot(z, Y_DMC_percent, label="y_DMC")
# ax[1].set_xlabel("Reactor length z [m]")
# ax[1].set_ylabel("Yield DMC [mol/m³]")
# ax[1].set_title("retenate concentration concentrations")
# ax[1].grid(True)
# ax[1].legend()

# plt.tight_layout()
# plt.show()

# def run_case(name, feed_y, r1_scale, r2_scale, T, P, P_membrane):
#     cfg = ModelConfig()
       
#     cfg.r1_scale = r1_scale
#     cfg.r2_scale = r2_scale

#     cfg.feed_y = np.array(feed_y, dtype=float)
#     cfg.feed_y /= cfg.feed_y.sum()

#     cfg.T = T
#     cfg.p = P
#     cfg.P_membrane = np.array(P_membrane, dtype=float)

#     model = MembraneReactorModel(cfg)
#     result = model.solve()

#     c_ret, T_ret, c_perm, T_perm = model.fields()
#     species = list(model.species_labels)
#     z = model.z_c

#     table = pd.DataFrame({
#         "inlet retentate [mol/m3]": cfg.inlet_concentration,
#         "outlet retentate [mol/m3]": c_ret[-1, :],
#         "outlet permeate [mol/m3]": c_perm[-1, :],
#         "retentate change [mol/m3]": c_ret[-1, :] - cfg.inlet_concentration,
#     }, index=species)

#     plt.figure(figsize=(8,5))
#     for i, sp in enumerate(species):
#         plt.plot(z, c_ret[:,i], label=sp)
#     plt.xlabel("Reactor length z [m]")
#     plt.ylabel("Concentration [mol/m³]")
#     plt.title("retenate concentration concentrations")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     print("\n" + "=" * 80)
#     print(name)
#     print("success:", result.success)
#     print("message:", result.message)
#     print("nfev:", result.nfev)
#     print(table.to_string(float_format=lambda x: f"{x:.4e}"))

#     return cfg, model, result, c_ret, T_ret, c_perm, T_perm




# cases = {
#     # "R1 only without membrane: CO2 + H2 feed": {
#     #     "feed_y": [0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
#     #     "r1_scale": 1.0,
#     #     "r2_scale": 0.0,
#     #     "T": cfg.T,
#     #     "P": cfg.p,
#     #     "P_membrane": [0, 0, 0, 0, 0, 0],

#     },
#     "R2 only with membrane : CO2 + CH3OH feed": {
#         "feed_y": [1/3, 0.0, 2/3, 0.0, 0.0, 0.0],
#         "r1_scale": 0.0,
#         "r2_scale": 1.0,
#         "T": 400,
#         "P": 200e5,
#         "P_membrane": cfg.P_membrane,
#     },
#     "R2 only without membrane : CO2 + CH3OH feed": {
#         "feed_y": [1/3, 0.0, 2/3, 0.0, 0.0, 0.0],
#         "r1_scale": 0.0,
#         "r2_scale": 1.0,
#         "T": 400,
#         "P": 200e5,
#         "P_membrane": [0,0,0,0,0,0]

    # },
#     "R1 + R2: CO2 + H2 feed": {
#         "feed_y": [0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
#         "r1_scale": 1.0,
#         "r2_scale": 1.0,
#         "T": cfg.T,
#         "P": cfg.p,
#         "P_membrane": cfg.P_membrane,
#     },
# }

# results = {
#     name: run_case(name, **settings)
#     for name, settings in cases.items()
# }


# c_ret, T_ret, c_perm, T_perm = model.fields()
# z = model.z_c
# species = list(model.species_labels)
# n_species = cfg.n_species

# r1, r2 = model.reac.reaction_rates(c_ret, T_ret)
# source = model.reac.species_source(c_ret, T_ret)

# k2_profile = model.reac.k_eff_r2(T_ret)
# k_eq = model.reac.k2_eq_T(T_ret)

# c_in = cfg.inlet_concentration

# # voor de plots, bereken de mole fractions en partiele drukken van H2O in retentate en permeate
# y_ret = c_ret / np.maximum(
#     np.sum(c_ret, axis=1, keepdims=True),
#     1e-12
# )

# y_perm = c_perm / np.maximum(
#     np.sum(c_perm, axis=1, keepdims=True),
#     1e-12
# )

# p_h2o_ret = y_ret[:, 3] * cfg.p / 1e5      # bar
# p_h2o_perm = y_perm[:, 3] * cfg.p_perm / 1e5   # bar
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# # -------------------------------------------------
# # (1) Sweep-gas side concentrations
# # -------------------------------------------------

# axs[0,0].plot(z, c_perm[:,3], label="H2O")
# axs[0,0].plot(z, c_perm[:,5], label="N2")

# axs[0,0].set_xlabel("Reactor length z [m]")
# axs[0,0].set_ylabel("Concentration [mol/m³]")
# axs[0,0].set_title("Permeate concentrations")
# axs[0,0].grid(True)
# axs[0,0].legend()

# # -------------------------------------------------
# # (2) Reaction rates
# # -------------------------------------------------

# axs[0,1].plot(z, r1, label="r1")
# axs[0,1].plot(z, r2, label="r2")

# axs[0,1].set_xlabel("Reactor length z [m]")
# axs[0,1].set_ylabel("Rate [mol/m³ s]")
# axs[0,1].set_title("Reaction-rate profiles")
# axs[0,1].grid(True)
# axs[0,1].legend()

# # -------------------------------------------------
# # (3) H2O partial pressures
# # -------------------------------------------------

# axs[1,0].plot(z, p_h2o_ret, label="Retentate")
# axs[1,0].plot(z, p_h2o_perm, label="Permeate")

# axs[1,0].set_xlabel("Reactor length z [m]")
# axs[1,0].set_ylabel("H2O partial pressure [bar]")
# axs[1,0].set_title("Water partial-pressure driving force")
# axs[1,0].grid(True)
# axs[1,0].legend()

# # -------------------------------------------------
# # (4) H2O mole fractions
# # -------------------------------------------------

# delta_p_h2o = p_h2o_ret - p_h2o_perm

# axs[1,1].plot(z, delta_p_h2o)

# axs[1,1].set_xlabel("Reactor length z [m]")
# axs[1,1].set_ylabel("Δp_H2O [bar]")
# axs[1,1].set_title("Water driving force")
# axs[1,1].grid(True)

# plt.tight_layout()
# plt.show()

# # plt.figure(figsize=(8, 5))
# # plt.plot(z, k_eq, label="k_eq")
# # plt.xlabel("Reactor length z [m]")
# # plt.ylabel("k_eq [mol/m3s]")
# # plt.title("Reaction-rate profiles")
# # plt.grid(True)
# # plt.legend()
# # plt.show()

# concentration_table = pd.DataFrame({
#     "inlet retentate [mol/m3]": c_in,
#     "outlet retentate [mol/m3]": c_ret[-1, :],
#     "outlet permeate [mol/m3]": c_perm[-1, :],
#     "retentate change [mol/m3]": c_ret[-1, :] - c_in,
# }, index=species)

# print("\n=== COMPONENT CONCENTRATIONS ===")
# print(concentration_table.to_string())
