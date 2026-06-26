from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import ModelConfig
from membraneReactorModel import MembraneReactorModel
from reaction import SPECIES_LABELS

cfg = ModelConfig()

def run_simulation(T=None, p=None, P_membrane=None, v_ret=None, v_perm=None):
    cfg = ModelConfig()

    if T is not None:
        cfg.T = T

    if p is not None:
        cfg.p = p

    if P_membrane is not None:
        cfg.P_membrane = P_membrane

    if v_ret is not None:
        cfg.v_ret = v_ret

    if v_perm is not None:
        cfg.v_perm = v_perm

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

plt.figure(figsize=(7, 5))

for v_perm in velocity:
    z, Y_DMC_percent = run_simulation(T=573.15, p=200e5, P_membrane=cfg.P_membrane, v_ret=cfg.v_ret, v_perm=v_perm)
    plt.plot(z, Y_DMC_percent, label=f"{v_perm} m/s")

plt.xlabel("Reactor length z [m]")
plt.ylabel("Y_DMC [%]")
plt.title("Influence of permeate velocity on DMC yield")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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

fig, ax = plt.subplots(1, 2, figsize=(12, 8))

for i, sp in enumerate(species):
    ax[0].plot(z, c_ret[:,i], label=sp)
ax[0].set_xlabel("Reactor length z [m]")
ax[0].set_ylabel("Concentration [mol/m³]")
ax[0].set_title("retenate concentration concentrations")
ax[0].grid(True)
ax[0].legend()

c_CO2_in = cfg.inlet_concentration[0]
c_DMC = c_ret[:, 4]
Y_DMC = c_DMC / c_CO2_in
Y_DMC_percent = Y_DMC * 100

ax[1].plot(z, Y_DMC_percent, label="y_DMC")
ax[1].set_xlabel("Reactor length z [m]")
ax[1].set_ylabel("Yield DMC [mol/m³]")
ax[1].set_title("DMC yield")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()

def run_case(name, feed_y, r1_scale, r2_scale, T, P, P_membrane):
    cfg = ModelConfig()
       
    cfg.r1_scale = r1_scale
    cfg.r2_scale = r2_scale

    cfg.feed_y = np.array(feed_y, dtype=float)
    cfg.feed_y /= cfg.feed_y.sum()

    cfg.T = T
    cfg.p = P
    cfg.P_membrane = np.array(P_membrane, dtype=float)

    model = MembraneReactorModel(cfg)
    result = model.solve()

    c_ret, T_ret, c_perm, T_perm = model.fields()
    species = list(model.species_labels)
    z = model.z_c

    table = pd.DataFrame({
        "inlet retentate [mol/m3]": cfg.inlet_concentration,
        "outlet retentate [mol/m3]": c_ret[-1, :],
        "outlet permeate [mol/m3]": c_perm[-1, :],
        "retentate change [mol/m3]": c_ret[-1, :] - cfg.inlet_concentration,
    }, index=species)

    plt.figure(figsize=(8,5))
    for i, sp in enumerate(species):
        plt.plot(z, c_ret[:,i], label=sp)
    plt.xlabel("Reactor length z [m]")
    plt.ylabel("Concentration [mol/m³]")
    plt.title("retenate concentration concentrations")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\n" + "=" * 80)
    print(name)
    print("success:", result.success)
    print("message:", result.message)
    print("nfev:", result.nfev)
    print(table.to_string(float_format=lambda x: f"{x:.4e}"))

    return cfg, model, result, c_ret, T_ret, c_perm, T_perm

cases = {
    "R1 only without membrane: CO2 + H2 feed": {
        "feed_y": [0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
        "r1_scale": 1.0,
        "r2_scale": 0.0,
        "T": cfg.T,
        "P": 50e5,
        "P_membrane": [0, 0, 0, 0, 0, 0],

    },
    "R2 only with membrane : CO2 + CH3OH feed": {
        "feed_y": [1/3, 0.0, 2/3, 0.0, 0.0, 0.0],
        "r1_scale": 0.0,
        "r2_scale": 1.0,
        "T": 400,
        "P": 200e5,
        "P_membrane": cfg.P_membrane,
    },
    "R2 only without membrane : CO2 + CH3OH feed": {
        "feed_y": [1/3, 0.0, 2/3, 0.0, 0.0, 0.0],
        "r1_scale": 0.0,
        "r2_scale": 1.0,
        "T": 400,
        "P": 200e5,
        "P_membrane": [0,0,0,0,0,0]

    },
    "R1 + R2: CO2 + H2 feed": {
        "feed_y": [0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
        "r1_scale": 1.0,
        "r2_scale": 1.0,
        "T": cfg.T,
        "P": cfg.p,
        "P_membrane": cfg.P_membrane,
    },
}

results = {
    name: run_case(name, **settings)
    for name, settings in cases.items()
}

n_species = cfg.n_species

r1, r2 = model.reac.reaction_rates(c_ret, T_ret)
source = model.reac.species_source(c_ret, T_ret)
r_CO2_consumption = -source[:, 0]

P_species = np.asarray(cfg.P_species, dtype=float)
J_mem = P_species[None, :] * (c_ret - c_perm)  # [mol/(m2 s)]

c_in = cfg.inlet_concentration

D_eff_CO2 = cfg.eps_p * cfg.particle_diffusivity[0] / cfg.tortuosity

# Mears criterion:
mears_r1, mears_r2 = model.mears_criterion()
wp_r1, wp_r2 = model.weisz_prater_criterion()

# 1. Retentate concentrations
plt.figure(figsize=(8,5))
for i, name in enumerate(species):
    plt.plot(z, c_ret[:, i], label=f"{name} ret")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Retentate concentration [mol/m³]")
plt.title("Retentate concentration profiles")
plt.grid(True)
plt.legend()

# 2. Permeate concentrations
plt.figure(figsize=(8,5))
for i, name in enumerate(species):
    plt.plot(z, c_perm[:, i], label=f"{name} perm")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Permeate concentration [mol/m³]")
plt.title("Permeate concentration profiles")
plt.grid(True)
plt.legend()

# 3. Membrane flux
plt.figure(figsize=(8,5))
for i, name in enumerate(species):
    if np.any(np.abs(J_mem[:, i]) > 1e-20):
        plt.plot(z, J_mem[:, i], label=f"{name}")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Membrane flux [mol/(m² s)]")
plt.title("Membrane flux profiles")
plt.grid(True)
plt.legend()

# 4. Reaction rates
plt.figure(figsize=(8,5))
plt.plot(z, r1, label="r1: methanol synthesis")
plt.plot(z, r2, label="r2: DMC synthesis")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Rate [mol/(m³ s)]")
plt.title("Reaction-rate profiles")
plt.grid(True)
plt.legend()

# 5. Temperature profile
plt.figure(figsize=(8,5))
plt.plot(z, T_ret, label="T retentate")
plt.plot(z, T_perm, label="T permeate")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profiles")
plt.grid(True)
plt.legend()

# 6. Criteria and conversions
plt.figure(figsize=(8,5))
plt.plot(z, mears_r1, label="Mears r1")
plt.plot(z, mears_r2, label="Mears r2")
plt.axhline(0.15, linestyle=":", label="Mears guideline 0.15")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Criterion value [-]")
plt.title("External mass-transfer criteria")
plt.grid(True)
plt.legend()

plt.figure(figsize=(8,5))
plt.plot(z, wp_r1, label="Weisz-Prater r1")
plt.plot(z, wp_r2, label="Weisz-Prater r2")
plt.axhline(1, linestyle=":", label="WP guideline 1")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Criterion value [-]")
plt.title("Internal mass-transfer criteria")
plt.grid(True)
plt.tight_layout()
plt.show()
