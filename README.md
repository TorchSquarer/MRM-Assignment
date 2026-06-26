# MRM-Assignment
This code is associated with the course project of Multiphase Reactor Modelling (6EMA05) at Technical University of Eindhoven ([TU/e](https://tue.nl)). The code of this project is based on the [code example](https://computational-chemical-engineering.github.io/pymrm-book/pymrm/examples/membrane-module-2d/) for membranes written in on the [pymrm](https://computational-chemical-engineering.github.io/pymrm-book/) website manual.

Goals:
* survive the course
* get a passing grade
* bbq 

# Model description
The code simulates a one-dimensional membrane reactor with a retentate side and a permeate side. The retentate contains the reacting gas mixture, while the permeate side contains a sweep gas. The membrane allows selected components, mainly water, to permeate from the retentate to the permeate side.

The following reactions are included:

Methanol formation:
CO₂ + 3 H₂ ⇌ CH₃OH + H₂O
Dimethyl carbonate formation:
CO₂ + 2 CH₃OH ⇌ DMC + H₂O

The model includes species balances, energy balances, membrane transport, reaction kinetics, and several checks for mass transfer limitations.

# File structure
config.py
Contains all model settings and parameters, such as:

* reactor geometry
* grid size
* membrane permeability
* operating pressure and temperature
* catalyst and particle properties
* kinetic parameters
* feed composition
* solver settings
* derived properties such as Reynolds number, Peclet number, pressure drop, and inlet concentrations

Most values that need to be changed for simulations can be changed in this file.

buildoperators.py
Builds the numerical transport operators used in the model. These operators describe axial convection in the retentate and permeate side. The file also defines the axial grid of the reactor.

reaction.py
Contains the reaction kinetics for the two reactions. It calculates:

* mole fractions
* partial pressures
* temperature-dependent rate constants
* equilibrium constant for DMC formation
* reaction rates
* species source terms
* heat source terms

The component order used throughout the model is:

CO2, H2, CH3OH, H2O, DMC, N2
membraneReactorModel.py

Contains the main reactor model. It combines the configuration, transport operators, reaction rates, membrane transport, and energy balance.

This file solves the model along the reactor length using solve_ivp from SciPy. It also contains functions to extract useful results such as:

* retentate and permeate concentration profiles
* temperature profiles
* outlet values
* CO₂ and H₂ conversion
* Mears criterion
* Weisz-Prater criterion
* thermal Peclet number

# Required packages
The main required Python packages are:

* numpy
* scipy
* pymrm
'
The standard Python packages dataclasses, pathlib, and sys are also used.

# How to run

A typical simulation can be run by creating a configuration object, creating the reactor model, and solving it:

from config import ModelConfig
from membraneReactorModel import MembraneReactorModel

cfg = ModelConfig()
model = MembraneReactorModel(cfg)

result = model.solve()

c_ret, T_ret, c_perm, T_perm = model.fields()
outlet = model.outlet_values()
conversions = model.conversions()

print(result.message)
print(outlet)
print(conversions)
Main outputs

The most important outputs of the model are:

* concentration profiles in the retentate
* concentration profiles in the permeate
* temperature profiles
* outlet concentrations
* CO₂ and H₂ conversion
* reaction rates
* membrane fluxes
* mass transfer criteria

These results can be used to study how the membrane permeability, reactor length, temperature, pressure, and feed composition influence the reactor performance.

Notes

The bbq is not included in the code.