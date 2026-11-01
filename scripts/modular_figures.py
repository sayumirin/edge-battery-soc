"""Figures read saved experiment tables only; no training or raw-data access."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/modular-v3")
    args = parser.parse_args()
    root = Path(args.results)
    e4 = pd.read_csv(root / "E4/metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, kind in zip(axes.flat, ["voltage", "current", "temperature", "dropout", "hold", "crop"]):
        for label in ["clean_training", "augmented_training"]:
            g = e4[(e4.noise == kind) & (e4.ambient_c == 25) & (e4.scope == "common") & (e4.model == label)]
            s = g.groupby("level").rmse.agg(["mean", "std"])
            scale = {"voltage": 5, "current": 50, "temperature": 2, "dropout": 100, "hold": 1, "crop": 100}[
                kind
            ]
            units = {
                "voltage": "Noise SD (mV)",
                "current": "Bias (mA)",
                "temperature": "Bias (C)",
                "dropout": "Dropped (%)",
                "hold": "Hold (s)",
                "crop": "Start trimmed (%)",
            }[kind]
            ax.errorbar(s.index * scale, s["mean"], yerr=s["std"], marker="o", label=label)
        ax.set(title=kind, xlabel=units, ylabel="RMSE (pp)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "E4/robustness.png", dpi=160)
    plt.close(fig)
    calibration = pd.read_csv(root / "E8/calibration.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for temp, g in calibration[calibration.scope == "full"].groupby("ambient_c"):
        ax.plot(g.nominal, g.empirical_coverage, marker="o", label=f"{temp} C")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set(
        xlabel="Validation residual quantile",
        ylabel="Empirical coverage",
        title="Five-model ensemble: descriptive calibration (reused validation)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "E8/calibration.png", dpi=160)
    plt.close(fig)
    stream = pd.read_csv(root / "streaming/metrics.csv")
    stream = stream[(stream.scope == "common") & stream["mode"].isin(["window_reset", "stream_carry"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, g in stream.groupby("mode"):
        s = g.groupby("ambient_c").rmse.agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", label=mode)
    ax.set(
        xlabel="Ambient temperature (C)",
        ylabel="SoC RMSE (pp)",
        title="Full recording history vs 300 s window: matched endpoints",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "streaming/comparison.png", dpi=160)
    plt.close(fig)
    differences = pd.read_csv(root / "streaming/history_difference.csv")
    differences.groupby("ambient_c")[["max_abs_delta_pp", "mean_abs_delta_pp"]].max().to_csv(
        root / "streaming/difference_summary.csv"
    )
    temps = [25, 10, 0, -10, -20]

    def spread(values):
        return f"{values.mean():.2f} ± {values.std():.2f}" if len(values) else "—"

    def table(data, methods, column="method"):
        rows = ["| Method | " + " | ".join(f"{t}°C" for t in temps) + " |", "| --- " * 6 + "|"]
        for method, label in methods.items():
            cells = [spread(data[(data[column] == method) & (data.ambient_c == t)].rmse) for t in temps]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
        return rows

    shift = pd.concat([pd.read_csv(root / s / "metrics.csv") for s in ["E1", "E2", "E3"]])
    shift = shift[shift.scope == "common"].drop_duplicates(["seed", "ambient_c", "method"])
    adaptation = {
        "GRU direct": "Direct transfer",
        "GRU target stats": "Target statistics (V, I)",
        "GRU tune head": "Output layer, 300 steps",
        "GRU tune input": "Input layer, 300 steps",
        "GRU tune all": "All parameters, 300 steps",
        "Coulomb +5pp": "Coulomb counting (+5 pp initial error)",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, label in adaptation.items():
        s = shift[shift.method == method].groupby("ambient_c").rmse.agg(["mean", "std"])
        style = "--" if method.startswith("Coulomb") else "-"
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", linestyle=style, capsize=3, label=method)
    ax.set(xlabel="Ambient temperature (C)", ylabel="SoC RMSE, SoC 40-100% (pp)", xticks=temps)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "E3/temperature_shift.png", dpi=160)
    plt.close(fig)
    e6 = pd.read_csv(root / "E6/metrics.csv")
    sizes = pd.read_csv(root / "E6/model_sizes.csv")
    e7 = pd.read_csv(root / "E7/metrics.csv")
    e9 = pd.read_csv(root / "E9/metrics.csv")
    e9 = e9[e9.scope == "common"]
    lines = [
        "# Experiment results (E1–E9, streaming)",
        "",
        "Configurations and commands are described in the top-level README. Source models scale temperature "
        "with the fixed rule (T − 25) / 20 and standardize voltage and current on the 25°C training data. "
        "E3/E7 fine-tuning runs exactly 300 Adam steps (10 s window stride, batch 32, learning rate 1e-3); "
        "E9 runs 100 steps per update, 300 in total. Both settings were revised after an initial run "
        "(see ../fix-check-v1); results under the earlier settings remain in ../modular-v2.",
        "",
        "## Temperature transfer and adaptation (E1/E2/E3)",
        "",
        "SoC 40–100%, RMSE in percentage points; mean ± sample SD over three seeds. Adaptation uses only the "
        "first 1,800 s of each target-temperature Cycle 1 and never touches test data. Target statistics "
        "re-estimate the voltage and current mean and SD only; temperature keeps the fixed rule.",
        "",
        *table(shift, adaptation),
        "",
        "![Temperature transfer and adaptation](E3/temperature_shift.png)",
        "",
        "## Dynamic int8 (E6)",
        "",
        *table(e6[e6.scope == "common"], {"GRU direct": "float32", "GRU int8": "Dynamic int8"}),
        "",
        f"{int(sizes.parameters.iloc[0]):,} parameters; float32 file {int(sizes.float32_bytes.mean()):,} bytes, "
        f"dynamic int8 {int(sizes.int8_bytes.mean()):,} bytes. File size is not MCU Flash/RAM usage.",
        "",
        "## LG cross-cell transfer (E7)",
        "",
        "Panasonic source models applied to LG HG2; adaptation uses the first 1,800 s of each temperature's "
        "Mixed1 recording with the E3 settings.",
        "",
        *table(
            e7[e7.scope == "common"],
            {
                "direct": "Direct transfer",
                "target stats": "Target statistics (V, I)",
                "tune head": "Output layer, 300 steps",
                "tune input": "Input layer, 300 steps",
                "tune all": "All parameters, 300 steps",
            },
        ),
        "",
        "## Online updates and forgetting (E9)",
        "",
        "Target Cycle 1 is split chronologically into three 600 s chunks with 100 steps per update; "
        "errors before and after the three updates (mean over three seeds).",
        "",
        "| Target / supervision | Target before → after | 25°C source before → after |",
        "| --- | --- | --- |",
    ]
    for temp in temps[1:]:
        for supervision in ["true_label", "coulomb_pseudo"]:
            cells = []
            for domain in ["target", "source"]:
                g = e9[
                    (e9.target_temp == temp)
                    & (e9.supervision == supervision)
                    & (e9.evaluation_domain == domain)
                ]
                before, after = (g[g["update"] == u].rmse.mean() for u in [0, e9["update"].max()])
                cells.append(f"{before:.2f} → {after:.2f}")
            lines.append(f"| {temp}°C / {supervision} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Signed input bias (E4)",
        "",
        "25°C, SoC 40–100%, RMSE in percentage points; mean ± sample SD over three seeds.",
        "",
        "| Input bias | Clean training | Augmented training |",
        "| --- | --- | --- |",
    ]
    for kind, level in [("current", -4), ("current", 4), ("temperature", -4), ("temperature", 4)]:
        cells = []
        for label in ["clean_training", "augmented_training"]:
            v = e4[
                (e4.noise == kind)
                & (e4.level == level)
                & (e4.ambient_c == 25)
                & (e4.scope == "common")
                & (e4.model == label)
            ].rmse
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        name = f"Current {level * 50:+g} mA" if kind == "current" else f"Temperature {level * 2:+g}°C"
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The augmentation recipe stays at 5 mV, +50 mA and +2°C; negative offsets are an added control. "
        "Positive-direction augmentation does not guarantee robustness to the opposite direction; "
        "unfavourable results are kept.",
        "",
        "![Signed robustness](E4/robustness.png)",
        "",
        "## Five-model uncertainty (E8)",
        "",
        "Seeds 17, 42, 2026, 101 and 314. This is one five-model ensemble; validation data are reused for "
        "early stopping, so only empirical coverage is reported, with no independent calibration guarantee.",
        "",
        "| Ambient | Raw interval coverage | Validation-residual coverage | Residual interval width (pp) |",
        "| --- | --- | --- | --- |",
    ]
    intervals = pd.read_csv(root / "E8/intervals.csv")
    for temp in [25, 10, 0, -10, -20]:
        a = intervals[
            (intervals.ambient_c == temp)
            & (intervals.scope == "common")
            & (intervals.method == "raw_ensemble")
        ].iloc[0]
        b = intervals[
            (intervals.ambient_c == temp)
            & (intervals.scope == "common")
            & (intervals.method == "validation_residual")
        ].iloc[0]
        lines.append(
            f"| {temp}°C | {a.empirical_coverage:.1%} | {b.empirical_coverage:.1%} | {b.mean_width_pp:.3f} |"
        )
    lines += [
        "",
        "![Empirical coverage](E8/calibration.png)",
        "",
        "## Full-recording streaming",
        "",
        "Compared at matched window endpoints, SoC 40–100%. State is carried within a file and reset between "
        "files; it is not reset between execution chunks.",
        "",
        "| Ambient | Reset-window RMSE | Streaming RMSE | Max pointwise difference (pp) |",
        "| --- | --- | --- | --- |",
    ]
    for temp in [25, 10, 0, -10, -20]:
        cells = []
        for mode in ["window_reset", "stream_carry"]:
            v = stream[(stream.ambient_c == temp) & (stream["mode"] == mode)].rmse
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        maximum = differences[differences.ambient_c == temp].max_abs_delta_pp.max()
        lines.append(f"| {temp}°C | " + " | ".join(cells) + f" | {maximum:.6g} |")
    lines += [
        "",
        f"{int(differences.samples_processed.sum()):,} test samples over three seeds and 15 independent test "
        "files. This compares state history on the same offline-resampled data; it is neither an MCU real-time "
        "measurement nor a test of online sensor interpolation.",
        "",
        "![Streaming comparison](streaming/comparison.png)",
        "",
        "## Verification evidence",
        "",
        "verification.json records the configurations, ensemble size, temperature rule and adaptation step "
        "budgets, signed-bias ranges and streaming endpoint alignment. It also holds two numerical regression "
        "checks: E5, which does not depend on the source models, against the earlier run; and direct transfer "
        "plus 300-step tuning against the diagnostic check's variant B, seed by seed (tolerance 1e-4). "
        "pipeline.json records the command of every independent stage; each stage folder holds its "
        "configuration and a COMPLETE marker.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
