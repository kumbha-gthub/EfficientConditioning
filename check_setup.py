#!/usr/bin/env python3
"""Check the installation (takes a few seconds)."""
import os, sys, time
print(f"Python {sys.version.split()[0]}")
if sys.version_info[:2] != (3, 12):
    print("  note: the results in the paper were computed with Python 3.12; other "
          "versions can change individual numbers.")
pins = {"numpy": "2.5.1", "scipy": "1.18.0", "sklearn": "1.9.0", "matplotlib": "3.11.0",
        "torch": "2.11.0", "botorch": "0.18.1", "gpytorch": "1.15.2", "onnxruntime": "1.30.0"}
bad = 0
for mod, want in pins.items():
    try:
        v = __import__(mod).__version__
        flag = "ok" if v.startswith(want) else f"differs from {want}"
        print(f"  {mod:<11} {v:<14} {flag}")
    except Exception as e:
        bad += 1
        print(f"  {mod:<11} NOT INSTALLED  ({type(e).__name__}: {e})")
t0 = time.time()
import numpy as np
from utils import hartmann6, latin_hypercube, GPSurrogate, MQRBFPowerSurrogate
X = latin_hypercube(30, 6, np.array([[0.0, 1.0]] * 6), seed=42)
y = hartmann6(X)
gp = GPSurrogate(); gp.fit(X, y)
mu, sd = gp.predict(X[:3])
mq = MQRBFPowerSurrogate(); mq.fit(X, y)
print(f"  GP and MQ-RBF test ok ({time.time() - t0:.1f}s); GP mean at x0 = {mu[0]:.6f}")
import yahpo_data
missing = [rel for rel in yahpo_data.SHA256
           if not os.path.exists(os.path.join(yahpo_data.DATA, *rel.split("/")))]
if missing:
    print(f"  YAHPO files: {len(missing)} missing; run  python yahpo_data.py  "
          "(needed by exp23 and exp24)")
else:
    print("  YAHPO files: present")
print("SETUP OK" if bad == 0 else "SETUP INCOMPLETE - see messages above")
