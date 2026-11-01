"""Export trained weights, compile C99, verify real streams, benchmark this host."""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path
from battery_shift.artifacts import load_frames

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from battery_shift.models import GRU
from battery_shift.data import FEATURES


def compile_c(source, out):
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    library = out / ("battery.dll" if os.name == "nt" else "libbattery.so")
    if compiler:
        cmd = [compiler, "-std=c99", "-O2", "-shared", "-I", str(out), str(source), "-o", str(library)]
        if os.name != "nt":
            cmd += ["-fPIC", "-lm"]
        subprocess.run(cmd, check=True, capture_output=True)
    elif os.name == "nt":
        vswhere = (
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft Visual Studio/Installer/vswhere.exe"
        )
        install = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
        if not install:
            raise RuntimeError("Install a C99 compiler: MSVC, gcc or clang")
        setup = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
        batch = out / "build.cmd"
        batch.write_text(
            f'@echo off\ncall "{setup}"\nif errorlevel 1 exit /b 1\ncl /nologo /O2 /std:c11 /LD /I"{out}" "{source}" /Fe:"{library}" /Fo:"{out / "gru.obj"}" /link /IMPLIB:"{out / "battery.lib"}"\n'
        )
        result = subprocess.run(["cmd.exe", "/d", "/c", str(batch)], capture_output=True)
        if result.returncode:
            raise RuntimeError(
                result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
            )
        compiler = "MSVC /O2 /std:c11"
    else:
        raise RuntimeError("No C compiler available")
    return library, compiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/mvp-v1")
    parser.add_argument("--output", default="results/c-inference")
    args = parser.parse_args()
    run = Path(args.run)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(run / "config.yaml")
    stats = json.loads((run / "normalization.json").read_text())
    frames = load_frames("data/interim")
    rows = []
    torch.set_num_threads(1)
    for seed in cfg.seeds:
        sd = out / str(seed)
        sd.mkdir(exist_ok=True)
        model = GRU(cfg.hidden).eval()
        model.load_state_dict(torch.load(run / str(seed) / "float32.pt", weights_only=True))
        arrays = {name.replace(".", "_"): v.numpy().flatten() for name, v in model.state_dict().items()}
        arrays.update(
            feature_mean=np.array(stats["mean"]),
            feature_std=np.array(stats["std"]),
            feature_mask=np.array(cfg.get("feature_mask", [1, 1, 1])),
        )
        header = f"#define HIDDEN {cfg.hidden}\n"
        for name, a in arrays.items():
            header += (
                f"static const float {name}[{len(a)}] = {{" + ",".join(f"{float(v):.9e}f" for v in a) + "};\n"
            )
        (sd / "weights.h").write_text(header)
        library, compiler = compile_c(Path("embedded/gru.c").resolve(), sd)
        dll = ctypes.CDLL(str(library))
        state = (ctypes.c_float * cfg.hidden)()
        pointer = ctypes.POINTER(ctypes.c_float)
        dll.battery_step.argtypes = [pointer, pointer]
        dll.battery_step.restype = ctypes.c_float
        dll.battery_reset.argtypes = [pointer]
        dll.battery_state_bytes.restype = ctypes.c_size_t
        maximum = 0.0
        steps = 0
        for frame in frames:
            if frame.split.iloc[0] != "test":
                continue
            # Up to 1,000 consecutive steps at drive-current onset (real OOD inputs).
            current = np.flatnonzero(abs(frame.current_a.to_numpy()) > 0.05)
            start = max(0, int(current[0]) - 30) if len(current) else 0
            raw = np.ascontiguousarray(frame[FEATURES].to_numpy(dtype="float32")[start : start + 1000])
            dll.battery_reset(state)
            hidden = None
            with torch.inference_mode():
                for row in raw:
                    normalized = (
                        (row - np.array(stats["mean"], dtype="float32"))
                        / np.array(stats["std"], dtype="float32")
                        * np.array(cfg.get("feature_mask", [1, 1, 1]), dtype="float32")
                    )
                    y, hidden = model.step(torch.tensor(normalized).reshape(1, 1, 3), hidden)
                    c = dll.battery_step(state, row.ctypes.data_as(pointer))
                    maximum = max(maximum, abs(float(y.item()) - c))
                    steps += 1
        if maximum >= 1e-4:
            raise AssertionError(f"C parity failed: {maximum}")
        sample = np.array([3.7, 1.0, 25.0], dtype="float32")
        dll.battery_benchmark.argtypes = [pointer, ctypes.c_int]
        dll.battery_benchmark.restype = ctypes.c_double
        latency = dll.battery_benchmark(sample.ctypes.data_as(pointer), 100000)
        rows.append(
            dict(
                seed=seed,
                max_abs_error=maximum,
                real_stream_steps=steps,
                state_bytes=dll.battery_state_bytes(),
                weight_float_bytes=sum(len(a) * 4 for a in arrays.values()),
                explicit_step_float_workspace_bytes=(cfg.hidden + 3 + 3 + 6) * 4,
                host_seconds_per_step=latency,
                compiler=compiler,
                scope="host C float32 only; not MCU RAM peak or latency; workspace excludes compiler stack overhead",
            )
        )
    (out / "verification.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(out / "verification.csv", index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
