import pandas as pd
import numpy as np

score_file = "biored_mul_score.txt"

rows = []

with open(score_file) as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 7:
            continue

        if parts[0] == "Overall":
            continue

        try:
            support = int(parts[1])
            f1 = float(parts[-1])
            rows.append((support, f1))
        except:
            pass

supports = np.array([r[0] for r in rows])
f1s = np.array([r[1] for r in rows])

macro_f1 = np.mean(f1s)
weighted_f1 = np.average(f1s, weights=supports)

print("Macro-F1   =", round(macro_f1, 4))
print("Weighted-F1 =", round(weighted_f1, 4))