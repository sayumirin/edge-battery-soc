"""Check a protocol-v3 modular run from its saved evidence (v2 results keep their own verification.json)."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from battery_shift.artifacts import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/modular-v3")
    args = parser.parse_args()
    root = Path(args.results)
    stages = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "streaming"]
    checked = {}
    for stage in stages:
        folder = root / stage
        assert (folder / "COMPLETE").exists(), stage
        cfg = OmegaConf.load(folder / "experiment.yaml")
        assert cfg.experiment == stage
        table = folder / ("intervals.csv" if stage == "E8" else "metrics.csv")
        data = pd.read_csv(table)
        assert len(data) and not data.isna().any().any(), stage
        checked[stage] = dict(rows=len(data), sha256=digest(table))
    e4 = pd.read_csv(root / "E4/metrics.csv")
    cfg = OmegaConf.load(root / "E4/experiment.yaml")
    for noise in ["current", "temperature"]:
        for seed in cfg.seeds:
            for temp in cfg.temperatures:
                for model in ["clean_training", "augmented_training"]:
                    g = e4[
                        (e4.noise == noise) & (e4.seed == seed) & (e4.ambient_c == temp) & (e4.model == model)
                    ]
                    assert set(g.level) == set(cfg.noise[noise])
                    assert min(g.level) < 0 < max(g.level)
    e8 = pd.read_csv(root / "E8/intervals.csv")
    ens = OmegaConf.load(root / "E8/experiment.yaml")
    assert ens.ensemble_size == 5 and len(set(ens.seeds)) == 5 and e8.ensemble_size.eq(5).all()
    for temp in ens.temperatures:
        data = pd.read_csv(root / f"E8/predictions_{temp}.csv")
        assert len([c for c in data if c.startswith("seed_")]) == 5
    stream = pd.read_csv(root / "streaming/metrics.csv")
    diffs = pd.read_csv(root / "streaming/history_difference.csv")
    assert len(diffs) == 45
    a = stream[stream["mode"] == "window_reset"].reset_index(drop=True)
    b = stream[stream["mode"] == "stream_carry"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        a[["seed", "ambient_c", "scope", "n"]], b[["seed", "ambient_c", "scope", "n"]]
    )
    source = Path(json.loads((root / "pipeline.json").read_text())["source"])
    model_map = json.loads((source / "models.json").read_text())
    assert len(set(model_map[str(s)]["sha256"] for s in ens.seeds)) == 5
    # Protocol v3: fixed temperature rule in the source models, fixed adaptation step budgets.
    training = OmegaConf.load(source / "config.yaml")
    norms = json.loads((source / "normalization.json").read_text())
    assert [norms["mean"][2], norms["std"][2]] == list(training.temperature_scale)
    e3 = OmegaConf.load(root / "E3/experiment.yaml")
    histories = sorted((root / "E3").glob("adapt_*.json"))
    assert len(histories) == len(e3.seeds) * len(e3.temperatures) * len(e3.adapt.modes)
    assert all(json.loads(p.read_text())[-1]["steps"] == e3.adapt.steps for p in histories)
    e7 = OmegaConf.load(root / "E7/experiment.yaml")
    e9 = OmegaConf.load(root / "E9/experiment.yaml")
    protocol = dict(
        temperature_scale=list(training.temperature_scale),
        adaptation=dict(
            E3=dict(runs=len(histories), **OmegaConf.to_container(e3.adapt)),
            E7=OmegaConf.to_container(e7.adapt),
            E9=dict(
                steps_per_update=e9.steps_per_update,
                updates=e9.chunks,
                batch_size=e9.batch_size,
                stride=e9.stride,
                learning_rate=e9.learning_rate,
            ),
        ),
    )
    # Only E5 is independent of the Panasonic source models, so only it must match the original run.
    regressions = {}
    new = pd.read_csv(root / "E5/metrics.csv")
    previous = pd.read_csv("results/soh-v1/metrics.csv")
    joined = new.merge(
        previous, on=["seed", "protocol", "fold"], suffixes=("_new", "_old"), validate="one_to_one"
    )
    assert len(joined) == len(new)
    for metric in ["rmse", "mae"]:
        np.testing.assert_allclose(joined[metric + "_new"], joined[metric + "_old"], atol=1e-4, rtol=0)
    regressions["E5"] = dict(
        matched_rows=len(joined), max_rmse_delta=float(np.max(abs(joined.rmse_new - joined.rmse_old)))
    )
    # The post-hoc check's variant B trained its own models with the same seeds, temperature rule and
    # 300-step tuning, so E1/E2 direct transfer and E3 head/all tuning must reproduce it seed by seed.
    fix = pd.read_csv("results/fix-check-v1/metrics.csv")
    fix = fix[(fix.variant == "B_phys_T") & (fix.scope == "common")]
    names = {"direct": "GRU direct", "tune_all_300": "GRU tune all", "tune_head_300": "GRU tune head"}
    ours = pd.concat(pd.read_csv(root / s / "metrics.csv") for s in ["E1", "E2", "E3"])
    ours = ours[ours.scope == "common"].drop_duplicates(["seed", "ambient_c", "method"])
    joined = fix.assign(method=fix.method.map(names)).merge(
        ours, on=["seed", "ambient_c", "method"], suffixes=("_fix", "_v3"), validate="one_to_one"
    )
    assert len(joined) == len(fix)
    np.testing.assert_allclose(joined.rmse_v3, joined.rmse_fix, atol=1e-4, rtol=0)
    reproduction = dict(
        reference="results/fix-check-v1 variant B_phys_T",
        matched_rows=len(joined),
        max_rmse_delta=float(np.max(abs(joined.rmse_v3 - joined.rmse_fix))),
    )
    report = dict(
        status="passed",
        stages=checked,
        ensemble_members=list(ens.seeds),
        bidirectional_bias=True,
        streaming_files=len(diffs),
        stream_samples=int(diffs.samples_processed.sum()),
        matched_stream_window_endpoints=True,
        independent_configs=True,
        persisted_inputs_only=True,
        source=str(source),
        source_manifest_sha256=digest(source / "models.json"),
        protocol=protocol,
        unchanged_protocol_regression=regressions,
        fix_check_reproduction=reproduction,
        limitations=[
            "MCU not measured",
            "Five-model intervals reuse early-stopping validation",
            "Offline 1 Hz interpolation remains noncausal; streaming evaluation changes state lifetime only",
        ],
    )
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
