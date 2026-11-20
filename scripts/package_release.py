"""Package a parameter release after a passing back-to-back test.

The tag must be v<version> of the parameter file, and the test report must belong to the same
file (same sha256). The bundle holds the kernel source, the generated header, the parameter file
and the test report, as a reproducible .tar.gz plus SHA256SUMS.
"""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

from battery_shift.params import identity, load, render_header

ROOT = Path(__file__).resolve().parents[1]

USAGE = """{identity}

Portable float32 C99 SoC estimator (GRU) with its parameters.

Files
  gru.c          kernel source, no heap allocation
  weights.h      generated from params.json; do not edit
  params.json    versioned parameter file (weights, normalization, feature mask), sha256 inside
  b2b_main.c     back-to-back test driver used in CI
  REPORT.md      back-to-back test results for this parameter file
  report.json    the same results, machine readable

Use
  Put weights.h on the include path and include gru.c once in one translation unit.
  battery_reset(&state) at the start of a recording, then battery_step(&state, raw) per 1 Hz sample
  with raw = {{voltage V, discharge-positive current A, surface temperature degC}}.
  battery_params_id() returns "{identity}".

Scope
  {scope}
"""


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
    tar.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params/gru-source-v3.json")
    parser.add_argument("--check", required=True, help="output folder of scripts/verify_arm.py")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", default="dist")
    args = parser.parse_args()
    params = load(args.params)
    report = json.loads((Path(args.check) / "report.json").read_text())
    if args.tag != f"v{params['version']}":
        sys.exit(f"Tag {args.tag} does not match parameter version {params['version']}")
    if report["params_sha256"] != params["sha256"] or not report["passed"]:
        sys.exit("The back-to-back report is missing, failed or belongs to another parameter file")
    name = f"edge-soc-{args.tag}"
    files = {
        "gru.c": (ROOT / "embedded/gru.c").read_bytes(),
        "b2b_main.c": (ROOT / "embedded/b2b_main.c").read_bytes(),
        "weights.h": render_header(params).encode(),
        "params.json": Path(args.params).read_bytes(),
        "REPORT.md": (Path(args.check) / "REPORT.md").read_bytes(),
        "report.json": (Path(args.check) / "report.json").read_bytes(),
        "README.txt": USAGE.format(identity=identity(params), scope=report["scope"]).encode(),
    }
    out = Path(args.output)
    shutil.rmtree(out / name, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for file, data in sorted(files.items()):
            add(tar, f"{name}/{file}", data)
    archive = out / f"{name}.tar.gz"
    with open(archive, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(buffer.getvalue())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    print(f"{archive} {digest}")


if __name__ == "__main__":
    main()
