"""Build the six-page technical report from saved measured artifacts."""

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
V3 = "results/modular-v3/"
TEMPS = [25, 10, 0, -10, -20]


def read(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scores(rows, **match):
    """RMSE values in the common SoC 40-100% scope for rows matching every given field."""
    return [
        float(r["rmse"])
        for r in rows
        if r["scope"] == "common" and all(r[k] == str(v) for k, v in match.items())
    ]


def spread(values):
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def main():
    output = ROOT / "report"
    output.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=9, alignment=TA_LEFT
        )
    )
    styles["Title"].fontSize = 27
    styles["Title"].leading = 31
    styles["Title"].textColor = colors.HexColor("#16334b")
    styles["Heading1"].textColor = colors.HexColor("#16334b")
    story = []

    def p(text):
        story.append(Paragraph(text, styles["BodySmall"]))

    def title(text):
        story.append(Paragraph(text, styles["Heading1"]))
        story.append(Spacer(1, 8))

    def table(rows, widths=None):
        wrapped = [[Paragraph(str(x), styles["BodySmall"]) for x in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eef4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#7293aa")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cdd8df")),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    def figure(path, w=510):
        from PIL import Image as PILImage

        image_path = ROOT / path
        with PILImage.open(image_path) as im:
            h = w * im.height / im.width
        story.append(Image(str(image_path), width=w, height=h))
        story.append(Spacer(1, 8))

    def page():
        story.append(PageBreak())

    # Protocol v3 results; first-round (v1) tables are used only for unchanged baselines and history.
    direct = read(V3 + "E1/metrics.csv") + read(V3 + "E2/metrics.csv")
    adapt = read(V3 + "E3/metrics.csv")
    lg = read(V3 + "E7/metrics.csv")
    physics = read("results/physics-v1/metrics.csv")
    c = json.loads((ROOT / "results/c-inference-v3/verification.json").read_text())
    sizes = read(V3 + "E6/model_sizes.csv")[0]
    modular = json.loads((ROOT / V3 / "verification.json").read_text())
    gru = {t: scores(direct, ambient_c=t, method="GRU direct") for t in TEMPS}
    tuned = {t: scores(adapt, ambient_c=t, method="GRU tune all") for t in TEMPS[1:]}
    ekf = [mean(scores(physics, ambient_c=t)) for t in TEMPS]
    lg_direct = {t: scores(lg, ambient_c=t, method="direct") for t in TEMPS}
    lg_tuned = {t: scores(lg, ambient_c=t, method="tune all") for t in TEMPS}
    int8, fp32 = int(sizes["int8_bytes"]), int(sizes["float32_bytes"])

    title("Battery Shift | Technical Report")
    p("1 November 2026 | Local research prototype | Nominal-capacity SoC reference | Protocol v3")
    p(
        "A reproducible study of battery state-of-charge estimation under temperature shift, with adaptation, a physical baseline, uncertainty diagnostics and a portable C inference kernel. The project is intended as evidence of experimental and engineering practice, not as a validated vehicle BMS."
    )
    table(
        [
            ["Measured deliverable", "Result"],
            [
                "Panasonic preprocessing",
                "26 pinned files; five ambient temperatures; 293,124 samples at 1 Hz",
            ],
            [
                "Source-only GRU",
                f"{int(sizes['parameters']):,} parameters; 25 C RMSE {spread(gru[25])} percentage points",
            ],
            [
                "Temperature transfer",
                f"At -10 C: direct {spread(gru[-10])} pp; all-parameter fine-tuning for 300 steps on "
                f"1,800 s of target data {spread(tuned[-10])} pp",
            ],
            [
                "Physical baseline",
                f"Source-calibrated ECM/EKF, no target data: {min(ekf):.2f}-{max(ekf):.2f} pp at all five temperatures",
            ],
            [
                "Dynamic int8",
                f"{fp32:,} to {int8:,} serialized bytes ({(1 - int8 / fp32) * 100:.1f}% reduction)",
            ],
            [
                "C float32",
                f"{sum(r['real_stream_steps'] for r in c):,} real-stream steps across five seeds; "
                f"max C/PyTorch difference {max(r['max_abs_error'] for r in c):.2g}",
            ],
            [
                "Cross-cell LG HG2",
                f"At -10 C: direct {spread(lg_direct[-10])} pp; all-parameter adaptation {spread(lg_tuned[-10])} pp",
            ],
            [
                "Reproduction",
                "Protocol v3 run once on Windows and checked seed by seed against the post-hoc check; "
                "fresh-environment and Linux/GCC validation cover the first-round protocol",
            ],
        ],
        [155, 355],
    )
    tuned_means = [mean(v) for v in tuned.values()]
    p(
        "<b>Interpretation.</b> Source accuracy does not imply transfer reliability. This report uses protocol v3, which fixes two first-round issues found by a post-hoc check (Section 2): temperature z-scored against a 1.04 C source SD, and fine-tuning that took only three optimizer steps. Both fixes were chosen after seeing test results. "
        f"With temperature scaled by a fixed rule, direct transfer still fails (-20 C: {mean(gru[-20]):.2f} pp): a source range of about +/-1 C gives no basis for extrapolation. "
        f"Labelled adaptation on 1,800 s of target data reaches {min(tuned_means):.2f}-{max(tuned_means):.2f} pp. A source-calibrated ECM/EKF remains better without any target data."
    )
    lg_cold = [mean(lg_tuned[t]) for t in TEMPS[1:]]
    lg_cold_direct = [mean(lg_direct[t]) for t in TEMPS[1:]]
    p(
        "<b>E7 protocol.</b> Panasonic models transfer to one LG HG2 cell: 19 held-out drive files, with separate Mixed1 adaptation (first 1,800 s) at each temperature using the E3 settings. The corrupt 10 C LA92 file is excluded before fitting. "
        f"From 10 to -20 C, all-parameter adaptation gives {min(lg_cold):.2f}-{max(lg_cold):.2f} pp (direct: {min(lg_cold_direct):.2f}-{max(lg_cold_direct):.2f} pp). "
        f"At 25 C it worsens RMSE from {mean(lg_direct[25]):.2f} to {mean(lg_tuned[25]):.2f} pp, while head-only tuning stays at "
        f"{mean(scores(lg, ambient_c=25, method='tune head')):.2f} pp: 300 steps on one profile overfit when the source model already fits. "
        "This is a cross-dataset test: cell chemistry, instruments and driving profiles also change."
    )
    p(
        "<b>Scope boundary.</b> Panasonic labels retain 2.9 Ah; LG uses 3 Ah with per-temperature C/20 label sensitivity reported separately. Protocol v3 ran on Windows only. Remote CI is deferred by the user. Real MCU measurements and another physical computer remain unverified. All reported timing is host timing."
    )
    streaming = read(V3 + "streaming/metrics.csv")
    differences = read(V3 + "streaming/history_difference.csv")
    earlier = read("results/modular-v2/streaming/history_difference.csv")
    cells = [
        f"{mean(scores(streaming, ambient_c=-10, mode=m)):.3f}" for m in ["window_reset", "stream_carry"]
    ]
    largest = max(float(r["max_abs_delta_pp"]) for r in differences)
    p(
        "<b>Streaming accuracy.</b> Complete test recordings are processed with state reset only at file boundaries, "
        "using chunked causal GRU execution. At matched window endpoints, -10 C RMSE is "
        + cells[0]
        + " pp with 300 s reset windows and "
        + cells[1]
        + f" pp with carried state; the maximum endpoint difference across all files and seeds is {largest:.3g} pp. "
        f"Under protocol v2 carried state differed by up to {max(float(r['max_abs_delta_pp']) for r in earlier):.1f} pp in cold data; "
        "the gap disappears with the fixed temperature rule, consistent with it coming from out-of-range temperature inputs. "
        "The input remains the same offline-resampled 1 Hz data; this does not validate an online sensor-resampling implementation."
    )
    page()
    title("1 | Data, labels and leakage controls")
    p(
        "Source train: 25 C Cycle 1-4 and NN. Source validation: HWFTa/HWFTb. Held-out evaluation: US06, UDDS and LA92 at 25, 10, 0, -10 and -20 C. Adaptation uses the first 1,800 seconds of each target-temperature Cycle 1. A fixed file manifest and checksum verification precede processing."
    )
    p(
        "Raw discharge current and Ah are negative; processing makes discharge positive. SoC = 1 - discharged Ah / 2.9. Original Ah offsets are retained. Twenty-one duplicate within-file relative timestamps are removed deterministically; overlapping absolute recording intervals across files are rejected. Windows never cross files. Voltage and current are standardized on source training data only; temperature uses the fixed rule (T - 25) / 20."
    )
    p(
        "Official-archive auditing confirms all 26 mirror files byte-for-byte and only one 25 C C/20 reference. The original protocol repeats steps 3-5 across temperatures, not C/20 step 2; the original plan's five-temperature capacity assumption is corrected. The 25 C counter difference is 2.99732 Ah, recorded later than the source drive tests. Replacing 2.9 Ah shifts labels by at most 3.133 pp; primary labels remain unchanged."
    )
    figure("results/diagnostics-v1/eda_-10.png", 430)
    p(
        "Example audit: target files can include about two hours of preceding rest. Plots display every tenth sample; inference still uses the 1 Hz series. Offline interpolation is not a causal online resampling implementation."
    )
    page()
    title("2 | Transfer, adaptation and model comparison")
    p(
        "The GRU uses a 300-second window, 32 hidden units, an input affine layer and a scalar output. Training windows stride 30 seconds; evaluation windows stride 10 seconds. Seeds are 17, 42 and 2026. Early stopping uses source validation only. Metrics use the common SoC 40-100% range, in percentage points; columns marked v1 keep the first-round protocol."
    )
    cnn = read("results/cnn-v2/metrics.csv")
    ablation = read("results/no-temperature-v1/metrics.csv")
    rows = [["Ambient", "GRU direct", "GRU tune all, 300 steps", "VI-only GRU (v1)", "ECM/EKF", "1D-CNN"]]
    for temp in TEMPS:
        rows.append(
            [
                str(temp) + " C",
                f"{mean(gru[temp]):.2f}",
                f"{mean(tuned[temp]):.2f}" if temp in tuned else "-",
                f"{mean(scores(ablation, ambient_c=temp, method='GRU direct')):.2f}",
                f"{mean(scores(physics, ambient_c=temp)):.2f}",
                f"{mean(scores(cnn, ambient_c=temp)):.2f}",
            ]
        )
    table(rows, [55, 80, 105, 100, 80, 90])
    cnn_v1 = read("results/cnn-v1/metrics.csv")
    cnn_cold = [mean(scores(cnn_v1, ambient_c=t)) for t in TEMPS[1:]]
    p(
        f"Without target data, the VI-only ablation transfers better than the v3 GRU (-10 C: {mean(scores(ablation, ambient_c=-10, method='GRU direct')):.2f} vs {mean(gru[-10]):.2f} pp). "
        "It was designed after observing first-round test results and is exploratory; its -10 C RMSE rises to 24.53 pp when evaluation is restricted to endpoints with |I| &gt; 0.05 A."
    )
    p(
        "The physical baseline fits one RC branch to separate 25 C HPPC data, with a quasi-OCV curve from C/20 discharge. It transfers unchanged to other temperatures. Ageing, hysteresis, sparse HPPC gaps and fixed covariance choices limit physical interpretation. "
        f"The CNN uses the v3 input scaling (first round: {min(cnn_cold):.0f}-{max(cnn_cold):.0f} pp in cold data) and a 15-step endpoint receptive field, so it is not a matched-history architecture comparison."
    )
    p(
        "<b>Protocol revision.</b> Two issues surfaced after the first-round runs: temperature z-scored on the 1.04 C source SD (cold data lie 35-45 SD away and seeds collapse to different constants), and E3 fine-tuning that took three Adam steps (51 windows, batch 128, 3 epochs). "
        "A post-hoc check with unchanged data, splits, windows and seeds tested both fixes; protocol v3 adopted them and reran E1-E9. Temperature is scaled as (T - 25) / 20. Adaptation uses 151 windows (10 s stride), batch 32 and exactly 300 Adam steps at lr 1e-3. "
        f"Both changes were chosen after seeing test results; no hyperparameters were swept and there is no independent confirmation set. Protocol v3 reproduces the check seed by seed (max difference {modular['fix_check_reproduction']['max_rmse_delta']:.1g} pp)."
    )
    primary = read("results/mvp-v1/metrics.csv")
    fix = read("results/fix-check-v1/metrics.csv")
    rows = [
        [
            "Ambient",
            "v1 tune all, 3 steps",
            "v3 target V/I stats",
            "v3 head only",
            "v3 input only",
            "v3 all",
            "D, no target data",
        ]
    ]
    for temp in TEMPS[1:]:
        rows.append(
            [
                str(temp) + " C",
                f"{mean(scores(primary, ambient_c=temp, method='GRU tune all')):.2f}",
                *[
                    f"{mean(scores(adapt, ambient_c=temp, method='GRU ' + m)):.2f}"
                    for m in ["target stats", "tune head", "tune input", "tune all"]
                ],
                f"{mean(scores(fix, ambient_c=temp, variant='D_multi_temp_phys_T', method='direct')):.2f}",
            ]
        )
    table(rows, [50, 75, 80, 75, 75, 70, 85])
    sds = [stdev(v) for v in tuned.values()]
    p(
        f"Head-only tuning recovers most of the shift; all-parameter tuning is best at every target temperature, with seed SD {min(sds):.2f}-{max(sds):.2f} pp. "
        f"Re-estimating voltage/current statistics on target data without tuning makes 10 C worse ({mean(gru[10]):.2f} to {mean(scores(adapt, ambient_c=10, method='GRU target stats')):.2f} pp). "
        "D trains on 25 C plus the three other cold Cycle 1 files, never the target temperature; it uses more training files, and the cell was tested from 25 C down to -20 C, so temperature and ageing stay mixed."
    )
    page()
    title("3 | Robustness and SoH split sensitivity")
    p(
        "E4 tests voltage noise (0-20 mV SD), signed current offsets (-200 to +200 mA), temperature offsets (-8 to +8 C), packet dropout (0-10%), held observations (1-10 seconds) and truncated starts (0-50%). Labels are unchanged. Training augmentation combines 5 mV, +50 mA and +2 C perturbations, with matched initialization. E4 evaluates every 30 seconds (other main experiments use 10 seconds); both models share endpoints. Crop experiments have different endpoints and are not direct point-for-point degradation estimates."
    )
    e4 = read(V3 + "E4/metrics.csv")
    rows = [["25 C setting", "Clean train RMSE", "Augmented train RMSE"]]
    for kind, level in [
        ("voltage", 0),
        ("voltage", 4),
        ("current", -4),
        ("current", 4),
        ("temperature", -4),
        ("temperature", 4),
    ]:
        values = []
        for model in ["clean_training", "augmented_training"]:
            x = [
                float(r["rmse"])
                for r in e4
                if int(r["ambient_c"]) == 25
                and r["scope"] == "common"
                and r["noise"] == kind
                and float(r["level"]) == level
                and r["model"] == model
            ]
            values.append(f"{mean(x):.2f} +/- {stdev(x):.2f}")
        units = {"voltage": (5, "mV SD"), "current": (50, "mA bias"), "temperature": (2, "C bias")}
        scale, unit = units[kind]
        rows.append([f"{kind}: {level * scale:g} {unit}", *values])
    table(rows, [170, 170, 170])
    p(
        "Under the fixed temperature rule an 8 C sensor offset is a small input change, and clean-model RMSE rises to about 9-10 pp (protocol v2: 22.8-30.6 pp). Augmented training is worse than clean training in every setting above; the augmentation recipe was kept unchanged and the result is reported as observed."
    )
    p(
        "E5 uses NASA cells B0005, B0006, B0007 and B0018. A fixed ridge regression predicts capacity / 2 Ah from only the first 600 seconds of discharge: voltage levels, mean current and temperature rise. Full-discharge duration and measured capacity are excluded from the input features. Scaling is fit on training rows only."
    )
    soh = read(V3 + "E5/metrics.csv")
    rows = [["Split", "SoH RMSE (capacity ratio)"]]
    for protocol, fold in [("random_cycles", "all")] + [
        ("leave_one_cell_out", c) for c in ["B0005", "B0006", "B0007", "B0018"]
    ]:
        values = [float(r["rmse"]) for r in soh if r["protocol"] == protocol and r["fold"] == fold]
        rows.append([protocol + " / " + fold, f"{mean(values):.4f} +/- {stdev(values):.4f}"])
    table(rows, [310, 200])
    p(
        "Random cycle splitting intentionally permits same-cell information in train and test. Held-cell folds do not. Deterministic ridge gives zero repeated-seed SD for fixed held-cell folds; this is not independent replication. Cell-specific cutoff voltages and only four cells limit generalization claims."
    )
    page()
    title("4 | Empirical intervals and online forgetting")
    p(
        "E8 evaluates five independently seeded source models (17, 42, 2026, 101, 314). Raw intervals use mean +/- 1.645 ensemble SD. A second interval uses the validation absolute-residual 90th percentile as constant half-width. Validation is reused after early stopping and observations are correlated: coverage is descriptive, not an independent conformal or safety guarantee. This is one five-model ensemble, not repeated independent ensemble experiments."
    )
    intervals = read(V3 + "E8/intervals.csv")
    rows = [["Ambient", "Raw coverage", "Raw width (pp)", "Validation-scaled coverage"]]
    widths = []
    for temp in TEMPS:
        a, b = (
            next(
                r
                for r in intervals
                if int(r["ambient_c"]) == temp and r["scope"] == "common" and r["method"] == method
            )
            for method in ["raw_ensemble", "validation_residual"]
        )
        widths.append(float(a["mean_width_pp"]))
        rows.append(
            [
                str(temp) + " C",
                f"{float(a['empirical_coverage']) * 100:.1f}%",
                f"{float(a['mean_width_pp']):.2f}",
                f"{float(b['empirical_coverage']) * 100:.1f}%",
            ]
        )
    table(rows, [80, 130, 125, 175])
    p(
        f"Ensemble width grows with the temperature gap ({widths[0]:.2f} pp at 25 C, {widths[-1]:.2f} pp at -20 C), so spread can flag a shift, but coverage falls to near zero because all members err in the same direction. "
        f"The validation-scaled interval keeps a fixed {float(b['mean_width_pp']):.2f} pp width and does not adapt to the shift."
    )
    p(
        "E9 updates on disjoint chronological 600-second chunks from target Cycle 1: three updates of 100 Adam steps each (batch 32, 10 s stride, lr 1e-3; first round: one step per update at lr 1e-4). Every update is followed by held-out target and 25 C evaluations. Two supervision modes are compared: measured-Ah labels and full-charge-anchor Coulomb pseudo-labels. The pseudo-label route requires a verified initial near-full state and retains nominal-capacity assumptions. No target test samples participate in an update."
    )
    online = read(V3 + "E9/metrics.csv")
    rows = [["Target / supervision", "Target before -> after", "Source before -> after"]]
    forgotten = []
    for supervision in ["true_label", "coulomb_pseudo"]:
        for temp in [-10, -20]:
            cells = []
            for domain in ["target", "source"]:
                before, after = (
                    mean(
                        scores(
                            online,
                            target_temp=temp,
                            supervision=supervision,
                            evaluation_domain=domain,
                            update=step,
                        )
                    )
                    for step in [0, 3]
                )
                cells.append(f"{before:.2f} -> {after:.2f}")
            forgotten.append(after)
            rows.append([f"{temp} C / {supervision}", *cells])
    table(rows, [220, 145, 145])
    p(
        f"Target error falls sharply, but 25 C source error rises from {mean(gru[25]):.2f} to {min(forgotten):.2f}-{max(forgotten):.2f} pp: unconstrained updates overwrite source behaviour, so deployment would need replay or regularization. "
        "Pseudo-labels track measured labels closely because both come from current integration over a short high-SoC segment. Three updates do not establish long-term continual learning performance."
    )
    page()
    title("5 | Edge inference, reproduction and remaining work")
    table(
        [
            ["C float32 check", "Measured result"],
            ["Real stream steps", f"{sum(r['real_stream_steps'] for r in c):,} (five v3 source models)"],
            ["Maximum C/PyTorch absolute difference", f"{max(r['max_abs_error'] for r in c):.3g}"],
            ["Persistent hidden state", str(c[0]["state_bytes"]) + " bytes"],
            ["Weights and preprocessing arrays", str(c[0]["weight_float_bytes"]) + " bytes"],
            [
                "Explicit float workspace",
                str(c[0]["explicit_step_float_workspace_bytes"]) + " bytes; excludes compiler stack overhead",
            ],
            [
                "Host runtime-clock time per step",
                f"{mean(r['host_seconds_per_step'] for r in c) * 1e6:.2f} microseconds (mean of five models)",
            ],
        ],
        [245, 265],
    )
    p(
        "The C99 kernel implements PyTorch reset-after GRU gate ordering, input normalization, affine transform and persistent state without heap allocation. Host checks use contiguous float buffers. Generated headers and the C source are supplied. Dynamic int8 remains a PyTorch CPU format; the portable C kernel is float32. "
        "Protocol v3 timing ran while other experiments shared the host (first round, alone: 1.87 microseconds). Neither serialization size nor host latency is a measurement of MCU Flash, peak RAM, power or real-time behavior."
    )
    p(
        "Protocol v3 ran once on Windows from persisted data and checkpoints. The first-round protocol reproduced exactly in a fresh Windows environment, and a separate WSL2 Ubuntu 24.04 environment completed all three seeds and GCC checks over another 45,000 steps (max error 1.23e-6); Windows/Linux float32 direct RMSE differed by at most 1.53e-5 pp and dynamic int8 by 0.1583 pp. "
        "Protocol v3 has not been re-validated on Linux, and scripts/reproduce_all.py has not been run end to end with the v3 stages."
    )
    p(
        "<b>Remaining boundaries:</b> actual STM32/ESP32 deployment, an independent confirmatory temperature set (the v3 changes were chosen after seeing test results), Linux and end-to-end re-validation of v3, and another physical computer. GitHub publication/CI is explicitly deferred by the user. Panasonic temperature-specific capacity measurements are absent from the official archive; their absence is a dataset limitation, not an unimplemented extraction step."
    )
    title("Sources and reproducibility")
    p(
        "Kollmeyer, Panasonic 18650PF Li-ion Battery Data, 2018. DOI: 10.17632/wykht8y7tg.1. CC BY 4.0. Pinned mirror revision: 6d720489c9b5b95624e448d44e8e2c0e48bdacaf. See data/splits/manifest.json and results/diagnostics-v1/capacity_reference.json for checksums."
    )
    p(
        "Saha and Goebel, NASA Ames PCoE Battery Data Set, 2007. PHM Society archive: phm-datasets.s3.amazonaws.com/NASA/5.%20Battery%20Data%20Set.zip. Archive SHA-256 and extraction protocol are saved in results/soh-v1/provenance.json; original README retained with raw data."
    )
    p(
        "Kollmeyer, Naguib and Skells, LG 18650HG2 Li-ion Battery Data, 2020, DOI: 10.17632/b5mj79w5w9.2, CC BY 4.0. Official browser download; 24 valid MAT files, separate CSV timestamp audit. Five C/20 capacities: 2.78074, 2.64881, 2.59862, 2.56366 and 2.49482 Ah (25 to -20 C). See results/cross-cell-v1 for checksums and sensitivity results."
    )
    p(
        "Independent E1-E9 YAML configurations and CLI stages consume persisted, checksum-verified datasets and checkpoints. Training, evaluation and figure rendering can rerun separately. Entry: scripts/run_pipeline.py; evidence: results/modular-v3/verification.json. Claims remain restricted to these laboratory protocols."
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#637888"))
        canvas.drawString(42, 25, "BATTERY SHIFT / LOCAL RESEARCH PROTOTYPE")
        canvas.drawRightString(A4[0] - 42, 25, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output / "battery-shift-report.pdf"),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=40,
        bottomMargin=42,
        title="Battery Shift Technical Report",
        author="Battery Shift Project",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output / "battery-shift-report.pdf")


if __name__ == "__main__":
    main()
