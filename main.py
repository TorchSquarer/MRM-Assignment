from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import ModelConfig
from membraneReactorModel import MembraneReactorModel
from reaction import SPECIES_LABELS

cfg = ModelConfig()

cfg = ModelConfig()
    
cfg.T = 400
cfg.P = 200e5
cfg.r1_scale = 0.0
cfg.r2_scale = 1.0
cfg.feed_y = np.array([1/3, 0.0, 2/3, 0.0, 0.0, 0.0])

model = MembraneReactorModel(cfg)
result = model.solve()


print("success:", result.success)
print("message:", result.message)
print("number of function evaluations:", result.nfev)

c_ret, T_ret, c_perm, T_perm = model.fields()
z = model.z_c
species = list(model.species_labels)
n_species = cfg.n_species

r1, r2 = model.reac.reaction_rates(c_ret, T_ret)
source = model.reac.species_source(c_ret, T_ret)
r_CO2_consumption = -source[:, 0]

k2_profile = model.reac.k_eff_r2(T_ret)
k_eq = model.reac.k2_eq_T(T_ret)

c_in = cfg.inlet_concentration

# voor de plots, bereken de mole fractions en partiele drukken van H2O in retentate en permeate
y_ret = c_ret / np.maximum(
    np.sum(c_ret, axis=1, keepdims=True),
    1e-12
)

y_perm = c_perm / np.maximum(
    np.sum(c_perm, axis=1, keepdims=True),
    1e-12
)

p_h2o_ret = y_ret[:, 3] * cfg.p / 1e5      # bar
p_h2o_perm = y_perm[:, 3] * cfg.p_perm / 1e5   # bar
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# -------------------------------------------------
# (1) Sweep-gas side concentrations
# -------------------------------------------------

axs[0,0].plot(z, c_perm[:,3], label="H2O")
axs[0,0].plot(z, c_perm[:,5], label="N2")

axs[0,0].set_xlabel("Reactor length z [m]")
axs[0,0].set_ylabel("Concentration [mol/m³]")
axs[0,0].set_title("Permeate concentrations")
axs[0,0].grid(True)
axs[0,0].legend()

# -------------------------------------------------
# (2) Reaction rates
# -------------------------------------------------

axs[0,1].plot(z, r1, label="r1")
axs[0,1].plot(z, r2, label="r2")

axs[0,1].set_xlabel("Reactor length z [m]")
axs[0,1].set_ylabel("Rate [mol/m³ s]")
axs[0,1].set_title("Reaction-rate profiles")
axs[0,1].grid(True)
axs[0,1].legend()

# -------------------------------------------------
# (3) H2O partial pressures
# -------------------------------------------------

axs[1,0].plot(z, p_h2o_ret, label="Retentate")
axs[1,0].plot(z, p_h2o_perm, label="Permeate")

axs[1,0].set_xlabel("Reactor length z [m]")
axs[1,0].set_ylabel("H2O partial pressure [bar]")
axs[1,0].set_title("Water partial-pressure driving force")
axs[1,0].grid(True)
axs[1,0].legend()

# -------------------------------------------------
# (4) H2O mole fractions
# -------------------------------------------------

delta_p_h2o = p_h2o_ret - p_h2o_perm

axs[1,1].plot(z, delta_p_h2o)

axs[1,1].set_xlabel("Reactor length z [m]")
axs[1,1].set_ylabel("Δp_H2O [bar]")
axs[1,1].set_title("Water driving force")
axs[1,1].grid(True)

plt.tight_layout()
plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(z, k_eq, label="k_eq")
# plt.xlabel("Reactor length z [m]")
# plt.ylabel("k_eq [mol/m3s]")
# plt.title("Reaction-rate profiles")
# plt.grid(True)
# plt.legend()
# plt.show()

concentration_table = pd.DataFrame({
    "inlet retentate [mol/m3]": c_in,
    "outlet retentate [mol/m3]": c_ret[-1, :],
    "outlet permeate [mol/m3]": c_perm[-1, :],
    "retentate change [mol/m3]": c_ret[-1, :] - c_in,
}, index=species)

print("\n=== COMPONENT CONCENTRATIONS ===")
print(concentration_table.to_string())