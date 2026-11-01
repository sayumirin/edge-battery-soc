Seeds: 17, 42, 2026. RMSE in SoC percentage points, mean ± sample SD over seeds.

## SoC 40–100% (main-table scope)

| Method | 25°C | 10°C | 0°C | -10°C | -20°C |
|---|---|---|---|---|---|
| A original model (z-scored temperature), direct transfer | 2.70 ± 0.46 | 38.92 ± 20.56 | 45.73 ± 25.16 | 54.89 ± 34.66 | 59.66 ± 35.82 |
| A original model + all parameters, 300 steps | — | 12.48 ± 3.06 | 8.10 ± 0.62 | 13.55 ± 3.62 | 20.01 ± 2.03 |
| B fixed temperature scaling, direct transfer | 2.31 ± 1.05 | 18.90 ± 3.93 | 36.78 ± 6.29 | 50.32 ± 8.23 | 69.81 ± 10.11 |
| B + output layer, 300 steps | — | 6.07 ± 2.13 | 10.82 ± 3.01 | 11.40 ± 2.44 | 14.46 ± 1.62 |
| B + all parameters, 300 steps | — | 3.06 ± 0.71 | 4.63 ± 0.27 | 8.26 ± 0.47 | 8.44 ± 0.37 |
| D multi-temperature training (target held out), direct | — | 7.89 ± 1.23 | 7.78 ± 2.38 | 12.05 ± 3.53 | 12.79 ± 1.37 |

## SoC 40–100% and |I| > 0.05 A (rest periods removed)

| Method | 25°C | 10°C | 0°C | -10°C | -20°C |
|---|---|---|---|---|---|
| A original model (z-scored temperature), direct transfer | 2.73 ± 0.46 | 34.53 ± 18.70 | 40.73 ± 19.34 | 46.96 ± 22.85 | 52.32 ± 25.26 |
| A original model + all parameters, 300 steps | — | 13.89 ± 3.38 | 9.14 ± 0.94 | 17.52 ± 4.10 | 24.34 ± 1.77 |
| B fixed temperature scaling, direct transfer | 2.31 ± 1.04 | 20.26 ± 4.10 | 40.91 ± 6.67 | 62.59 ± 9.02 | 83.87 ± 10.54 |
| B + output layer, 300 steps | — | 5.80 ± 2.01 | 6.08 ± 1.43 | 10.84 ± 3.46 | 18.75 ± 1.57 |
| B + all parameters, 300 steps | — | 3.06 ± 0.60 | 2.30 ± 0.10 | 3.65 ± 0.68 | 10.48 ± 1.13 |
| D multi-temperature training (target held out), direct | — | 6.53 ± 0.73 | 5.59 ± 1.30 | 8.21 ± 0.57 | 13.96 ± 0.23 |