"""Independent, configuration-driven stages operating on persisted artifacts only."""

import argparse
import copy
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .artifacts import digest, load_panasonic
from .data import Windows, statistics
from .evaluate import metrics, predict
from .export import export_models
from .models import GRU, coulomb_count
from .noise import perturb
from .train import fit, seed_all


def dataset(frames, stats, cfg, training=False, stride=None):
    return Windows(
        frames,
        *stats,
        cfg.window,
        stride if stride is not None else (cfg.train_stride if training else cfg.eval_stride),
        feature_mask=cfg.get("feature_mask"),
    )


def save_rows(rows, out, name="metrics.csv"):
    pd.DataFrame(rows).to_csv(out / name, index=False)


def record(rows, y, pred, **identity):
    for scope, values in metrics(y, pred).items():
        rows.append(dict(**identity, scope=scope, **values))


def source_model(source, seed, training):
    entry = json.loads((source / "models.json").read_text())[str(seed)]
    path = source / str(seed) / "float32.pt"
    if digest(path) != entry["sha256"]:
        raise ValueError("Source checkpoint checksum mismatch")
    m = GRU(training.hidden).eval()
    m.load_state_dict(torch.load(path, weights_only=True))
    return m


def train_source(cfg, out):
    training = OmegaConf.load(cfg.training_config)
    training.seeds = cfg.seeds
    torch.set_num_threads(training.threads)
    frames = load_panasonic(cfg.prepared, training)
    train = [f for f in frames if f.split.iloc[0] == "train"]
    val = [f for f in frames if f.split.iloc[0] == "validation"]
    stats = statistics(train, training.get("temperature_scale"))
    OmegaConf.save(training, out / "config.yaml")
    (out / "normalization.json").write_text(json.dumps(dict(mean=stats[0].tolist(), std=stats[1].tolist())))
    models = {}
    for seed in cfg.seeds:
        seed_all(seed)
        model, history = fit(
            GRU(training.hidden),
            dataset(train, stats, training, True),
            training,
            dataset(val, stats, training),
        )
        folder = out / str(seed)
        folder.mkdir()
        path = folder / "float32.pt"
        torch.save(model.state_dict(), path)
        (folder / "training.json").write_text(json.dumps(history, indent=2))
        models[str(seed)] = dict(sha256=digest(path), parameters=sum(p.numel() for p in model.parameters()))
        print(f"Source model {seed} saved", flush=True)
    (out / "models.json").write_text(json.dumps(models, indent=2))


def evaluate_source(cfg, out, source, training, frames, stats):
    rows, sizes = [], []
    for seed in cfg.seeds:
        model = source_model(source, seed, training)
        if cfg.experiment == "E6":
            folder = out / str(seed)
            folder.mkdir()
            example = dataset([f for f in frames if f.split.iloc[0] == "train"], stats, training)
            x = torch.from_numpy(np.stack([example[i][0] for i in range(4)]))
            quantized, size = export_models(model, x, folder)
            sizes.append(dict(seed=seed, parameters=sum(p.numel() for p in model.parameters()), **size))
        for temp in cfg.temperatures:
            tests = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
            ds = dataset(tests, stats, training)
            y, p = predict(model, ds, training.batch_size)
            outputs = {"GRU direct": p}
            if cfg.experiment in {"E1", "E2"}:
                cc = [
                    coulomb_count(
                        f.time_s,
                        f.current_a,
                        float(f.soc.iloc[0]) + cfg.initial_soc_error_pp / 100,
                        training.capacity_ah,
                    )
                    for f in tests
                ]
                outputs[f"Coulomb {cfg.initial_soc_error_pp:+g}pp"] = np.array(
                    [cc[j][end] for j, end in ds.index]
                )
            if cfg.experiment == "E3":
                adapt = [
                    f.loc[f.time_s < cfg.adapt.seconds].copy()
                    for f in frames
                    if f.split.iloc[0] == "adapt" and f.ambient_c.iloc[0] == temp
                ]
                target = statistics(adapt, training.get("temperature_scale"))
                _, outputs["GRU target stats"] = predict(
                    model, dataset(tests, target, training), training.batch_size
                )
                fitting = copy.deepcopy(training)
                fitting.batch_size, fitting.adapt_steps = cfg.adapt.batch_size, cfg.adapt.steps
                fitting.learning_rate = cfg.adapt.learning_rate
                adapt_windows = dataset(adapt, stats, training, stride=cfg.adapt.stride)
                for mode in cfg.adapt.modes:
                    seed_all(seed)
                    tuned, history = fit(model, adapt_windows, fitting, mode=mode)
                    _, outputs["GRU tune " + mode] = predict(tuned, ds, training.batch_size)
                    torch.save(tuned.state_dict(), out / f"adapt_{seed}_{temp}_{mode}.pt")
                    (out / f"adapt_{seed}_{temp}_{mode}.json").write_text(json.dumps(history))
            if cfg.experiment == "E6":
                _, outputs["GRU int8"] = predict(quantized, ds, training.batch_size)
            for method, pred in outputs.items():
                record(rows, y, pred, seed=seed, ambient_c=temp, method=method)
            pd.DataFrame(
                dict(
                    file_id=[tests[j].file_id.iloc[0] for j, _ in ds.index],
                    time_s=[float(tests[j].time_s.iloc[e]) for j, e in ds.index],
                    soc=y,
                    **outputs,
                )
            ).to_csv(out / f"predictions_{seed}_{temp}.csv", index=False)
    save_rows(rows, out)
    if sizes:
        save_rows(sizes, out, "model_sizes.csv")


def robustness(cfg, out, source, training, frames, stats):
    train = [f for f in frames if f.split.iloc[0] == "train"]
    val = [f for f in frames if f.split.iloc[0] == "validation"]
    rows = []
    for seed in cfg.seeds:
        model = source_model(source, seed, training)
        changed = []
        for j, frame in enumerate(train):
            for kind, strength in cfg.augmentation.items():
                frame = perturb(frame, kind, float(strength), seed + j)
            changed.append(frame)
        seed_all(seed)
        augmented, history = fit(
            GRU(training.hidden),
            dataset(changed, stats, training, True),
            training,
            dataset(val, stats, training),
        )
        torch.save(augmented.state_dict(), out / f"augmented_{seed}.pt")
        (out / f"augmented_{seed}.json").write_text(json.dumps(history))
        for temp in cfg.temperatures:
            tests = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
            for kind, levels in cfg.noise.items():
                for level in levels:
                    changed = [perturb(f, kind, float(level), seed + j) for j, f in enumerate(tests)]
                    ds = dataset(changed, stats, training, stride=cfg.eval_stride)
                    for label, m in [("clean_training", model), ("augmented_training", augmented)]:
                        y, p = predict(m, ds, training.batch_size)
                        record(rows, y, p, seed=seed, ambient_c=temp, noise=kind, level=level, model=label)
            print(f"E4 seed {seed}, {temp} C complete", flush=True)
    save_rows(rows, out)
    pd.DataFrame(rows).groupby(["ambient_c", "noise", "level", "model", "scope"]).rmse.agg(
        ["mean", "std"]
    ).to_csv(out / "summary.csv")


def intervals(cfg, out, source, training, frames, stats):
    if len(set(cfg.seeds)) != cfg.ensemble_size:
        raise ValueError("Ensemble requires the configured number of distinct seeds")
    models = [source_model(source, seed, training) for seed in cfg.seeds]
    val = dataset([f for f in frames if f.split.iloc[0] == "validation"], stats, training)
    yv = predict(models[0], val, training.batch_size)[0]
    vp = np.stack([predict(m, val, training.batch_size)[1] for m in models])
    residual = abs(yv - vp.mean(0))
    pd.DataFrame(dict(soc=yv, **{f"seed_{s}": p for s, p in zip(cfg.seeds, vp)})).to_csv(
        out / "validation_predictions.csv", index=False
    )
    q = float(np.quantile(residual, cfg.coverage, method=cfg.quantile_method))
    z = NormalDist().inv_cdf((1 + cfg.coverage) / 2)
    rows, curves = [], []
    for temp in cfg.temperatures:
        tests = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
        ds = dataset(tests, stats, training)
        y = predict(models[0], ds, training.batch_size)[0]
        ps = np.stack([predict(m, ds, training.batch_size)[1] for m in models])
        center, spread = ps.mean(0), ps.std(0, ddof=1)
        pd.DataFrame(
            dict(
                file_id=[tests[j].file_id.iloc[0] for j, _ in ds.index],
                time_s=[float(tests[j].time_s.iloc[e]) for j, e in ds.index],
                soc=y,
                **{f"seed_{s}": p for s, p in zip(cfg.seeds, ps)},
            )
        ).to_csv(out / f"predictions_{temp}.csv", index=False)
        for scope, mask in [("full", np.ones(len(y), bool)), ("common", (y >= 0.4) & (y <= 1))]:
            for method, half in [("raw_ensemble", z * spread), ("validation_residual", np.full_like(y, q))]:
                rows.append(
                    dict(
                        ambient_c=temp,
                        scope=scope,
                        method=method,
                        ensemble_size=len(models),
                        nominal_coverage=cfg.coverage,
                        n=int(mask.sum()),
                        empirical_coverage=float(np.mean(abs(y[mask] - center[mask]) <= half[mask])),
                        mean_width_pp=float(200 * np.mean(half[mask])),
                    )
                )
            for level in cfg.curve_levels:
                width = float(np.quantile(residual, level, method=cfg.quantile_method))
                curves.append(
                    dict(
                        ambient_c=temp,
                        scope=scope,
                        nominal=level,
                        empirical_coverage=float(np.mean(abs(y[mask] - center[mask]) <= width)),
                        width_pp=200 * width,
                    )
                )
    save_rows(rows, out, "intervals.csv")
    save_rows(curves, out, "calibration.csv")
    (out / "limitations.txt").write_text(
        "Five models in one ensemble; not independent ensemble repeats. "
        "Source validation was reused for early stopping. Empirical coverage only; no independent calibration guarantee.\n"
    )


def online(cfg, out, source, training, frames, stats):
    rows = []
    fitting = copy.deepcopy(training)
    fitting.batch_size, fitting.adapt_steps = cfg.batch_size, cfg.steps_per_update
    fitting.learning_rate = cfg.learning_rate
    for seed in cfg.seeds:
        model = source_model(source, seed, training)
        for temp in cfg.temperatures:
            frame = next(f for f in frames if f.split.iloc[0] == "adapt" and f.ambient_c.iloc[0] == temp)
            if abs(float(frame.soc.iloc[0]) - 1) > cfg.anchor_tolerance:
                raise ValueError("Full-charge anchor unavailable")
            pseudo = coulomb_count(frame.time_s, frame.current_a, 1, training.capacity_ah)
            for supervision in cfg.supervision:
                updated = copy.deepcopy(model)
                for step in range(cfg.chunks + 1):
                    if step:
                        chunk = frame.loc[
                            (frame.time_s >= (step - 1) * cfg.chunk_seconds)
                            & (frame.time_s < step * cfg.chunk_seconds)
                        ].copy()
                        if supervision == "coulomb_pseudo":
                            chunk["soc"] = pseudo[chunk.index]
                        seed_all(seed + step)
                        updated, _ = fit(
                            updated, dataset([chunk], stats, training, stride=cfg.stride), fitting
                        )
                    for domain, t in [("source", 25), ("target", temp)]:
                        tests = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == t]
                        y, p = predict(updated, dataset(tests, stats, training), training.batch_size)
                        record(
                            rows,
                            y,
                            p,
                            seed=seed,
                            target_temp=temp,
                            supervision=supervision,
                            update=step,
                            elapsed_seconds=step * cfg.chunk_seconds,
                            evaluation_domain=domain,
                        )
                torch.save(updated.state_dict(), out / f"online_{seed}_{temp}_{supervision}.pt")
        print(f"E9 seed {seed} complete", flush=True)
    save_rows(rows, out)


def streaming_predictions(model, normalized, chunk_size):
    """Carry the GRU state across chunks; reset only at independent file boundaries."""
    if chunk_size < 1:
        raise ValueError("Streaming chunk size must be positive")
    state, values = None, []
    with torch.inference_mode():
        for start in range(0, len(normalized), chunk_size):
            x = torch.from_numpy(normalized[start : start + chunk_size]).unsqueeze(0)
            hidden, state = model.gru(model.input(x), state)
            values.append(model.head(hidden).squeeze(0).squeeze(-1).numpy())
    return np.concatenate(values)


def streaming(cfg, out, source, training, frames, stats):
    rows, differences = [], []
    for seed in cfg.seeds:
        model = source_model(source, seed, training)
        for temp in cfg.temperatures:
            tests = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
            ds = dataset(tests, stats, training)
            y, window = predict(model, ds, training.batch_size)
            continuous = [streaming_predictions(model, x, cfg.chunk_size) for x in ds.x]
            stream = np.array([continuous[j][e] for j, e in ds.index])
            # Matched endpoints exclude the identical initial warm-up interval in both modes.
            for mode, p in [("window_reset", window), ("stream_carry", stream)]:
                record(rows, y, p, seed=seed, ambient_c=temp, mode=mode)
            for j, f in enumerate(tests):
                indices = [k for k, (i, _) in enumerate(ds.index) if i == j]
                for mode, p in [("window_reset", window), ("stream_carry", stream)]:
                    record(
                        rows,
                        y[indices],
                        p[indices],
                        seed=seed,
                        ambient_c=temp,
                        mode=mode + "/" + str(f.file_id.iloc[0]),
                    )
                differences.append(
                    dict(
                        seed=seed,
                        ambient_c=temp,
                        file_id=str(f.file_id.iloc[0]),
                        samples_processed=len(f),
                        endpoints=len(indices),
                        max_abs_delta_pp=float(100 * np.max(abs(stream[indices] - window[indices]))),
                        mean_abs_delta_pp=float(100 * np.mean(abs(stream[indices] - window[indices]))),
                    )
                )
            pd.DataFrame(
                dict(
                    file_id=[tests[j].file_id.iloc[0] for j, _ in ds.index],
                    time_s=[float(tests[j].time_s.iloc[e]) for j, e in ds.index],
                    soc=y,
                    window_reset=window,
                    stream_carry=stream,
                )
            ).to_csv(out / f"predictions_{seed}_{temp}.csv", index=False)
        print(f"Streaming seed {seed} complete", flush=True)
    save_rows(rows, out)
    save_rows(differences, out, "history_difference.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.source:
        cfg.source = args.source
    out = Path(args.output or cfg.output)
    cfg.output = str(out)
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, out / "experiment.yaml")
    provenance = dict(
        config_sha256=digest(args.config),
        prepared_sha256=digest(Path(cfg.prepared) / "dataset.json"),
        code_sha256=digest(__file__),
    )
    provenance["source_sha256"] = {p.as_posix(): digest(p) for p in Path("src/battery_shift").glob("*.py")}
    if cfg.experiment == "train":
        train_source(cfg, out)
    else:
        source = Path(cfg.source)
        if not (source / "COMPLETE").exists():
            raise ValueError("Source training is incomplete")
        training = OmegaConf.load(source / "config.yaml")
        source_provenance = json.loads((source / "provenance.json").read_text())
        if source_provenance["prepared_sha256"] != provenance["prepared_sha256"]:
            raise ValueError("Prepared dataset differs from source-training artifact")
        torch.set_num_threads(training.threads)
        frames = load_panasonic(cfg.prepared, training)
        norms = json.loads((source / "normalization.json").read_text())
        stats = tuple(np.asarray(norms[k], dtype="float32") for k in ["mean", "std"])
        provenance.update(
            source=str(source),
            models_sha256=digest(source / "models.json"),
            normalization_sha256=digest(source / "normalization.json"),
        )
        if cfg.experiment in {"E1", "E2", "E3", "E6"}:
            evaluate_source(cfg, out, source, training, frames, stats)
        else:
            {"E4": robustness, "E8": intervals, "E9": online, "streaming": streaming}[cfg.experiment](
                cfg, out, source, training, frames, stats
            )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "COMPLETE").write_text(str(cfg.experiment) + " completed\n")


if __name__ == "__main__":
    main()
