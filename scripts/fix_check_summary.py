import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results/fix-check-v1"
d = pd.read_csv(OUT / "metrics.csv")
TEMPS = [25, 10, 0, -10, -20]
LABELS = {
    ("A_original_zscoreT", "direct"): "A original model (z-scored temperature), direct transfer",
    ("A_original_zscoreT", "tune_all_300"): "A original model + all parameters, 300 steps",
    ("B_phys_T", "direct"): "B fixed temperature scaling, direct transfer",
    ("B_phys_T", "tune_head_300"): "B + output layer, 300 steps",
    ("B_phys_T", "tune_all_300"): "B + all parameters, 300 steps",
    ("D_multi_temp_phys_T", "direct"): "D multi-temperature training (target held out), direct",
}


def table(scope):
    lines = ["| Method | " + " | ".join(f"{t}°C" for t in TEMPS) + " |", "|---" * (len(TEMPS) + 1) + "|"]
    for (variant, method), label in LABELS.items():
        cells = []
        for t in TEMPS:
            v = d.query("variant==@variant and method==@method and ambient_c==@t and scope==@scope").rmse
            cells.append(f"{v.mean():.2f} ± {v.std():.2f}" if len(v) else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


seeds = sorted(d.seed.unique())
md = [
    "Seeds: " + ", ".join(map(str, seeds)) + ". RMSE in SoC percentage points, mean ± sample SD over seeds.",
    "",
    "## SoC 40–100% (main-table scope)",
    "",
    table("common"),
    "",
    "## SoC 40–100% and |I| > 0.05 A (rest periods removed)",
    "",
    table("active"),
]
(OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

plt.rcParams["font.family"] = ["DejaVu Sans"]
fig, ax = plt.subplots(figsize=(9, 5))
names = {
    ("A_original_zscoreT", "direct"): "original, direct",
    ("A_original_zscoreT", "tune_all_300"): "original + fine-tune (300 steps)",
    ("B_phys_T", "direct"): "physical T scaling, direct",
    ("B_phys_T", "tune_all_300"): "physical T + fine-tune (300 steps)",
    ("D_multi_temp_phys_T", "direct"): "multi-temperature training, direct",
}
for (variant, method), name in names.items():
    g = d.query("variant==@variant and method==@method and scope=='common'").groupby("ambient_c").rmse
    m, s = g.mean(), g.std()
    ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, label=name)
ax.set(xlabel="Ambient temperature (°C)", ylabel="SoC RMSE, SoC 40–100% (pp)")
ax.set_xticks(TEMPS)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fix_check.png", dpi=150)
