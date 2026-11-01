# Results: configs/no_temperature.yaml

Label: nominal_2.9Ah_shared_reference_not_temperature_calibrated

RMSE in percentage points, SoC 40–100%, mean ± sample SD over seeds. Adaptation at 25°C is not applicable.

| Method | 25°C | 10°C | 0°C | -10°C | -20°C |
|---|---|---|---|---|---|
| GRU direct | 1.954 ± 0.133 | 4.324 ± 0.548 | 10.619 ± 0.463 | 15.941 ± 0.327 | 27.235 ± 0.088 |
| GRU int8 | 2.014 ± 0.143 | 4.295 ± 0.518 | 10.604 ± 0.445 | 15.886 ± 0.333 | 27.192 ± 0.252 |
| Coulomb +5pp | 5.230 ± 0.000 | 5.111 ± 0.000 | 5.044 ± 0.000 | 5.014 ± 0.000 | 5.070 ± 0.000 |
| GRU target stats | — | 59.148 ± 1.922 | 48.641 ± 1.101 | 33.190 ± 0.875 | 26.530 ± 0.521 |
| GRU tune head | — | 4.387 ± 0.648 | 9.776 ± 0.623 | 15.488 ± 0.469 | 26.350 ± 0.111 |
| GRU tune input | — | 4.256 ± 0.565 | 10.494 ± 0.470 | 15.889 ± 0.338 | 26.993 ± 0.102 |
| GRU tune all | — | 3.959 ± 0.488 | 11.070 ± 0.833 | 17.407 ± 0.840 | 21.964 ± 0.956 |