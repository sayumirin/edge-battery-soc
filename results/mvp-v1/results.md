# Results: configs/mvp.yaml

Label: nominal_2.9Ah_shared_reference_not_temperature_calibrated

RMSE in percentage points, SoC 40–100%, mean ± sample SD over seeds. Adaptation at 25°C is not applicable.

| Method | 25°C | 10°C | 0°C | -10°C | -20°C |
|---|---|---|---|---|---|
| GRU direct | 2.696 ± 0.455 | 38.920 ± 20.557 | 45.728 ± 25.162 | 54.891 ± 34.662 | 59.656 ± 35.824 |
| GRU int8 | 2.718 ± 0.432 | 38.683 ± 20.534 | 45.608 ± 25.226 | 54.762 ± 34.665 | 59.398 ± 35.643 |
| Coulomb +5pp | 5.230 ± 0.000 | 5.111 ± 0.000 | 5.044 ± 0.000 | 5.014 ± 0.000 | 5.070 ± 0.000 |
| GRU target stats | — | 58.871 ± 3.803 | 47.968 ± 1.667 | 32.125 ± 1.120 | 25.949 ± 0.938 |
| GRU tune head | — | 34.082 ± 16.998 | 40.741 ± 20.617 | 49.185 ± 30.600 | 53.853 ± 32.257 |
| GRU tune input | — | 38.639 ± 20.460 | 45.600 ± 25.100 | 54.682 ± 34.568 | 59.436 ± 35.692 |
| GRU tune all | — | 28.385 ± 12.043 | 34.812 ± 18.082 | 43.209 ± 26.843 | 47.191 ± 26.917 |