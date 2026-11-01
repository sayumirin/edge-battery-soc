# Edge Battery SoC

State-of-charge (SoC) estimation for lithium-ion cells under ambient-temperature shift, covering few-shot adaptation, a physical baseline, uncertainty diagnostics, int8 quantization and a portable C inference kernel.


A technical report is in [`report/battery-shift-report.pdf`](report/battery-shift-report.pdf).

## Key results

The table reports SoC RMSE in percentage points over the SoC 40–100% range, as mean ± sample SD over seeds 17, 42 and 2026. Adaptation uses only the first 1,800 s of one target-temperature drive cycle.

| Method | 25 °C | 10 °C | 0 °C | −10 °C | −20 °C |
|---|---|---|---|---|---|
| GRU, direct transfer | 2.31 ± 1.05 | 18.90 ± 3.93 | 36.78 ± 6.29 | 50.32 ± 8.23 | 69.81 ± 10.11 |
| GRU, output layer tuned (300 steps) | — | 6.07 ± 2.13 | 10.82 ± 3.01 | 11.40 ± 2.44 | 14.46 ± 1.62 |
| GRU, all parameters tuned (300 steps) | — | 3.06 ± 0.71 | 4.63 ± 0.27 | 8.26 ± 0.47 | 8.44 ± 0.37 |
| 1D-CNN, direct transfer | 2.08 ± 0.37 | 16.17 ± 4.91 | 29.14 ± 10.26 | 34.86 ± 14.72 | 43.93 ± 19.33 |
| ECM/EKF, no target data | 4.34 | 2.55 | 1.35 | 1.65 | 4.43 |
| LG HG2, GRU direct transfer (E7) | 4.02 ± 0.78 | 19.03 ± 4.30 | 35.11 ± 6.81 | 53.70 ± 9.19 | 76.25 ± 11.05 |
| LG HG2, all parameters tuned (E7) | 7.09 ± 1.02 | 6.51 ± 0.77 | 5.63 ± 0.38 | 6.12 ± 0.25 | 11.15 ± 0.78 |

- **Source accuracy does not imply transfer.** The 25 °C training data span only about ±1 °C, so the network cannot learn the effect of temperature, and direct transfer fails at low temperature. Fine-tuning on 30 minutes of labelled target data brings the error down to 3.1–8.4 pp. A source-calibrated equivalent-circuit model with an extended Kalman filter (ECM/EKF) remains more accurate without any target data.
- **Cross-cell transfer.** On an LG HG2 cell, all-parameter adaptation reaches 5.6–11.2 pp from 10 to −20 °C. At 25 °C, where the source model already fits, it overfits a single drive profile (4.02 → 7.09 pp), while output-layer tuning stays at 4.02 pp.
- **Online updates forget.** Three chronological updates reduce −20 °C error from 69.81 to 14.17 pp, but 25 °C error rises from 2.31 to 14.6–16.3 pp.
- **Ensemble intervals flag the shift but do not bound the error.** Five-model interval width grows from 5.83 pp at 25 °C to 62.55 pp at −20 °C. All members err in the same direction, so cold-temperature coverage is near zero.
- **Edge cost.** Dynamic int8 quantization reduces the 3,597-parameter model from 17,549 to 9,097 serialized bytes, with 25 °C RMSE moving from 2.31 to 2.34 pp. The float32 C99 kernel matches PyTorch to within 4.8 × 10⁻⁷ over 75,000 real-data steps on five models. It needs 128 bytes of persistent state and 14,424 bytes of weights and preprocessing arrays.
- **State of health (E5).** On NASA cells, the SoH RMSE (capacity ratio) is 0.024 with random cycle splits and 0.023–0.097 when a whole cell is held out. Random splits are optimistic because the same cell appears in training and test data.

## Data

| Dataset | Use | Source |
|---|---|---|
| Panasonic 18650PF (Kollmeyer, 2018) | SoC at 25, 10, 0, −10 and −20 °C | DOI [10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1), CC BY 4.0 |
| LG 18650HG2 (Kollmeyer, Naguib and Skells, 2020) | Cross-cell transfer (E7) | DOI [10.17632/b5mj79w5w9.2](https://doi.org/10.17632/b5mj79w5w9.2), CC BY 4.0 |
| NASA Ames PCoE Battery Data Set (Saha and Goebel, 2007) | State of health (E5) | PHM Society archive |

- **Panasonic splits.** The source model trains on 25 °C Cycle 1–4 and NN, and validates on HWFTa/HWFTb. Testing uses US06, UDDS and LA92 at all five temperatures. Twenty-six files are pinned by mirror revision and checksum (`data/splits/manifest.json`).
- **Leakage controls.** Windows never cross files, and overlapping recording intervals across files are rejected.
- **Normalization.** Voltage and current are standardized on the 25 °C training files. Temperature uses the fixed rule (T − 25) / 20.
- **Labels.** SoC labels use a nominal capacity of 2.9 Ah. The official Panasonic archive contains only one 25 °C C/20 capacity reference, and the label sensitivity to it is reported separately.
- **Redistribution.** Raw and processed data are not included in this repository.

## Experiments

| ID | Question | Configuration |
|---|---|---|
| E1 | Same-temperature accuracy at 25 °C | `configs/e1.yaml` |
| E2 | Direct transfer to 10, 0, −10 and −20 °C | `configs/e2.yaml` |
| E3 | Few-shot adaptation: target statistics, output layer, input layer, all parameters | `configs/e3.yaml` |
| E4 | Voltage noise, signed current/temperature bias, dropout, held samples, truncation; noise-augmented training | `configs/e4.yaml` |
| E5 | NASA SoH: random cycle split vs leave-one-cell-out | `configs/e5.yaml` |
| E6 | Dynamic int8 quantization and ONNX export | `configs/e6.yaml` |
| E7 | Panasonic → LG HG2 cross-cell transfer and adaptation | `configs/e7.yaml` |
| E8 | Five-model ensemble prediction intervals | `configs/e8.yaml` |
| E9 | Chronological online updates with measured or Coulomb pseudo-labels; forgetting | `configs/e9.yaml` |
| — | Full-recording stateful inference vs 300 s reset windows | `configs/streaming.yaml` |

**Model.** The model is a GRU with a 300 s window at 1 Hz, an input affine layer, 32 hidden units and a scalar output, 3,597 parameters in total. Its inputs are voltage, current and temperature. Early stopping uses source validation data only.

**Fine-tuning.** Fine-tuning runs exactly 300 Adam steps with a 10 s window stride, batch size 32 and learning rate 10⁻³, with no early stopping. E9 runs 100 steps per update for three updates.

**Baselines.** The baselines are:
- Coulomb counting with a +5 pp initial-state error;
- a first-order RC equivalent-circuit model with an EKF, fitted to 25 °C HPPC data;
- a 1D-CNN;
- a voltage/current-only GRU ablation.


## Reproducing

**Requirements.**
- Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.
- A C compiler (MSVC Build Tools, gcc or clang) for the C parity check.
- The official LG and Panasonic archives. Download them in a browser with "Download All" from their Mendeley Data pages, and save them as `data/raw/lg/lg.zip` and `data/raw/panasonic-official.zip`. The scripts verify their hashes.

Install dependencies and run the tests:

```bash
uv sync --frozen --group report
uv run pytest
```

Download data, preprocess, train five source models and run every experiment stage:

```bash
uv run python scripts/download.py
uv run python scripts/run_pipeline.py --output runs/pipeline
```

Alternatively, train the source models once and reuse them:

```bash
uv run python -m battery_shift.experiments --config configs/train.yaml --output runs/source
uv run python scripts/run_pipeline.py --reuse-source runs/source --output runs/experiments
```

Each stage can also run alone, for example `uv run python -m battery_shift.experiments --config configs/e3.yaml --source runs/source --output runs/e3`. The experiment entry points refuse to write into an existing output folder.

C parity, the CNN baseline, the report and verification:

```bash
uv run python scripts/verify_c.py --run runs/source --output runs/c-inference
uv run python scripts/cnn_baseline.py
uv run --group report python scripts/build_report.py
uv run python scripts/validate_modular.py --results runs/experiments
```

`scripts/reproduce_all.py` chains every stage into one run.

## Protocol notes

Two settings were revised after an initial run had revealed implementation problems:

1. **Temperature scaling.** Temperature was originally z-scored against the 1.04 °C spread of the source data, which placed cold data 35–45 standard deviations outside the training range. It now uses the fixed rule (T − 25) / 20.
2. **Fine-tuning length.** Fine-tuning originally amounted to three optimizer steps per setting, and E9 used one step per update. It now runs a fixed budget of 300 steps, and 100 per update for E9.

Both revisions were chosen after seeing test results. No hyperparameters were swept, and there is no independent confirmation set. The current results reproduce the diagnostic check in `results/fix-check-v1/` seed by seed. Results under the earlier settings remain in the folders marked above.

## Limitations

- **Labels.** Labels use nominal capacity, and they share the current-integration source with the Coulomb-counting baseline.
- **Single cell.** Each temperature comes from one Panasonic cell, tested from 25 °C down to −20 °C. Temperature and ageing are therefore confounded.
- **Resampling.** Resampling to 1 Hz is offline interpolation, not a causal online implementation.
- **Intervals.** Ensemble intervals reuse validation data for early stopping, so their coverage is descriptive only.
- **Deployment.** Dynamic int8 is a PyTorch CPU format; the portable C kernel is float32. Serialized size and host timing are not measurements of MCU Flash, RAM, power or latency.
- **Validation coverage.**
  - The current results were produced on Windows. Linux/GCC cross-validation covers the earlier protocol only.
  - `reproduce_all.py` has not yet been run end to end with the current stages.
  - The workflow in `.github/workflows/tests.yml` has not been verified on GitHub.

## Citation of data

- P. Kollmeyer, *Panasonic 18650PF Li-ion Battery Data*, Mendeley Data, V1, 2018. doi:10.17632/wykht8y7tg.1
- P. Kollmeyer, M. Naguib and M. Skells, *LG 18650HG2 Li-ion Battery Data*, Mendeley Data, V2, 2020. doi:10.17632/b5mj79w5w9.2
- B. Saha and K. Goebel, *Battery Data Set*, NASA Ames Prognostics Data Repository, 2007.
