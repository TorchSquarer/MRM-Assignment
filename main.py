from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import ModelConfig
from membraneReactorModel import MembraneReactorModel
from reaction import SPECIES_LABELS

cfg = ModelConfig()
model = MembraneReactorModel(cfg)
result = model.solve()

print("success:", result.success)
print("message:", result.message)
print("number of function evaluations:", result.nfev)