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
cfg.feed_y = np.array([1/3, 0.0, 2/3, 0.0, 0.0])

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

plt.figure(figsize=(8, 5))
plt.plot(z, r1, label="r1")
plt.plot(z, r2, label="r2")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Rate [mol/m3s]")
plt.title("Reaction-rate profiles")
plt.grid(True)
plt.legend()
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