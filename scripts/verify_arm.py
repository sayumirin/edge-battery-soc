"""Back-to-back test of the C kernel on x86-64 and ARM against PyTorch, from one parameter file.

The parameter file is validated, turned into weights.h and into a PyTorch reference model.
The kernel is compiled natively and cross-compiled for aarch64 Linux (run under QEMU user mode),
and every build is fed all test drive cycles. A Cortex-M4F build is compiled but not run.

Linux only. Needs gcc, aarch64-linux-gnu-gcc, qemu-aarch64-static and arm-none-eabi-gcc, e.g.
    sudo apt-get install gcc-aarch64-linux-gnu qemu-user-static gcc-arm-none-eabi libnewlib-arm-none-eabi
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from battery_shift.artifacts import load_frames
from battery_shift.data import FEATURES
from battery_shift.params import identity, load, normalize, render_header, to_model

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-4  # same bound as scripts/verify_c.py
CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror"]
# Ubuntu 24.04 ships qemu-aarch64-static; newer releases ship a static qemu-aarch64 instead.
QEMU = shutil.which("qemu-aarch64-static") or shutil.which("qemu-aarch64") or "qemu-aarch64-static"
TARGETS = {
    "x86_64-linux": dict(cc="gcc", flags=[], runner=[]),
    "aarch64-linux": dict(cc="aarch64-linux-gnu-gcc", flags=["-static"], runner=[Path(QEMU).name]),
}
CORTEX_M4 = ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"]
TOOLS = ["gcc", "aarch64-linux-gnu-gcc", Path(QEMU).name, "arm-none-eabi-gcc", "arm-none-eabi-size"]


def run(cmd):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def compiler_version(cc):
    return run([cc, "--version"]).splitlines()[0]


def test_streams(frames):
    """Every test drive cycle in full, one recording per stream, in manifest order."""
    streams = []
    for frame in frames:
        if frame.split.iloc[0] != "test":
            continue
        streams.append(dict(
            cycle=str(frame.cycle.iloc[0]),
            ambient_c=float(frame.ambient_c.iloc[0]),
            raw=np.ascontiguousarray(frame[FEATURES].to_numpy(dtype=np.float32)),
        ))
    if not streams:
        raise ValueError("No test recordings in data/interim; run scripts/prepare_data.py first")
    return streams


def write_inputs(path, streams):
    with open(path, "wb") as f:
        for s in streams:
            f.write(np.uint32(len(s["raw"])).tobytes())
            f.write(s["raw"].astype("<f4").tobytes())


def reference(params, streams):
    model = to_model(params)
    torch.set_num_threads(1)
    outputs = []
    with torch.inference_mode():
        for s in streams:
            x = torch.from_numpy(normalize(params, s["raw"]))[None]
            hidden, _ = model.gru(model.input(x))
            outputs.append(model.head(hidden)[0, :, 0].numpy().astype(np.float32))
    return np.concatenate(outputs)


def check_target(name, target, out, inputs, expected):
    exe = out / f"b2b-{name}"
    run([target["cc"], *CFLAGS, *target["flags"], "-I", out, ROOT / "embedded/b2b_main.c", "-o", exe, "-lm"])
    outputs = out / f"outputs-{name}.bin"
    log = run([*target["runner"], exe, inputs, outputs]).strip()
    got = np.fromfile(outputs, dtype="<f4")
    if got.shape != expected.shape:
        raise AssertionError(f"{name}: {got.size} outputs for {expected.size} inputs")
    error = float(np.max(np.abs(got.astype(np.float64) - expected)))
    return got, dict(
        target=name,
        compiler=compiler_version(target["cc"]),
        flags=" ".join(CFLAGS + target["flags"]),
        runner=" ".join(target["runner"]) or "native",
        driver_log=log,
        max_abs_error_vs_pytorch=error,
        passed=error < TOLERANCE,
    )


def check_cortex_m4(out):
    obj = out / "gru-cortex-m4.o"
    run(["arm-none-eabi-gcc", *CFLAGS, *CORTEX_M4, "-I", out, "-c", ROOT / "embedded/gru.c", "-o", obj])
    text, data, bss = map(int, run(["arm-none-eabi-size", obj]).splitlines()[1].split()[:3])
    return dict(
        compiler=compiler_version("arm-none-eabi-gcc"),
        flags=" ".join(CFLAGS + CORTEX_M4),
        text_bytes=text,
        data_bytes=data,
        bss_bytes=bss,
        scope="object file of gru.c with weights, before linking libm; compiled, not run",
    )


def markdown(report):
    lines = [
        f"## Back-to-back test: {report['params']}",
        "",
        f"{len(report['streams'])} test drive cycles, {report['total_steps']:,} steps at 1 Hz, "
        f"tolerance {report['tolerance']:g} against PyTorch built from the same parameter file.",
        "",
        "| Target | Runs on | Max abs error vs PyTorch | Result |",
        "|---|---|---|---|",
    ]
    for t in report["targets"]:
        lines.append(f"| {t['target']} | {t['runner']} | {t['max_abs_error_vs_pytorch']:.2e} | "
                     f"{'pass' if t['passed'] else 'FAIL'} |")
    m4 = report["cortex_m4"]
    lines += [
        "",
        f"aarch64 vs x86-64 C outputs: max abs difference {report['max_abs_diff_aarch64_vs_x86_64']:.2e}.",
        "",
        f"Cortex-M4F ({m4['flags']}): {m4['text_bytes']:,} bytes of code and constant weights, "
        f"{m4['data_bytes']} bytes data, {m4['bss_bytes']} bytes bss ({m4['scope']}).",
        "",
        f"Parameter file sha256: `{report['params_sha256']}`",
        "",
        f"Scope: {report['scope']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params/gru-source-v3.json")
    parser.add_argument("--data", default="data/interim")
    parser.add_argument("--output", default="runs/arm-b2b")
    args = parser.parse_args()
    missing = [tool for tool in TOOLS if not shutil.which(tool)]
    if missing:
        sys.exit(f"Missing tools: {', '.join(missing)}. See the docstring of {Path(__file__).name}.")
    params = load(args.params)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.h").write_text(render_header(params), encoding="utf-8", newline="\n")
    streams = test_streams(load_frames(args.data))
    inputs = out / "inputs.bin"
    write_inputs(inputs, streams)
    expected = reference(params, streams)
    results, targets = {}, []
    for name, target in TARGETS.items():
        results[name], summary = check_target(name, target, out, inputs, expected)
        targets.append(summary)
    report = dict(
        params=identity(params),
        params_sha256=params["sha256"],
        params_file=str(Path(args.params).as_posix()),
        tolerance=TOLERANCE,
        streams=[dict(cycle=s["cycle"], ambient_c=s["ambient_c"], steps=len(s["raw"])) for s in streams],
        total_steps=int(sum(len(s["raw"]) for s in streams)),
        targets=targets,
        max_abs_diff_aarch64_vs_x86_64=float(np.max(np.abs(
            results["aarch64-linux"].astype(np.float64) - results["x86_64-linux"]))),
        cortex_m4=check_cortex_m4(out),
        scope=("QEMU user mode runs the aarch64 Linux build (ARM instruction set and libm), "
               "not an MCU. The Cortex-M4F build is compiled only. Parity is numerical agreement with "
               "PyTorch, not SoC accuracy."),
        passed=all(t["passed"] for t in targets),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "REPORT.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(markdown(report))
    if not report["passed"]:
        sys.exit("Back-to-back test failed")


if __name__ == "__main__":
    main()
