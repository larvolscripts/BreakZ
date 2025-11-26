import sys
import platform
import sklearn
import pandas as pd
import numpy as np

# Try imports safely
def safe_import(pkg):
    try:
        mod = __import__(pkg)
        return mod.__version__
    except:
        return "NOT INSTALLED"

packages = [
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "joblib",
]

print("======== SYSTEM ========")
print("Python:", sys.version)
print("OS:", platform.platform())

print("\n======== CORE LIBRARIES ========")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)

print("\n======== MODEL LIBRARIES ========")
for p in packages:
    print(f"{p}: {safe_import(p)}")
