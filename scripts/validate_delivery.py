"""Check the actual local delivery artifacts; never treats blocked items as passed."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from pypdf import PdfReader


def main():
    root = Path(__file__).resolve().parents[1]
    # Protocol v3 results first; first-round (v1) tables remain as history and for unchanged baselines.
    expected = [
        "results/modular-v3/E1/metrics.csv",
        "results/modular-v3/E2/metrics.csv",
        "results/modular-v3/E3/metrics.csv",
        "results/modular-v3/E4/metrics.csv",
        "results/modular-v3/E6/metrics.csv",
        "results/modular-v3/E7/metrics.csv",
        "results/modular-v3/E7/capacity_sensitivity.csv",
        "results/modular-v3/E8/intervals.csv",
        "results/modular-v3/E9/metrics.csv",
        "results/modular-v3/streaming/metrics.csv",
        "results/cnn-v2/metrics.csv",
        "results/no-temperature-v1/metrics.csv",
        "results/soh-v1/metrics.csv",
        "results/physics-v1/metrics.csv",
        "results/mvp-v1/metrics.csv",
        "results/extensions-v1/e9_metrics.csv",
        "results/cross-cell-v1/metrics.csv",
    ]
    for name in expected:
        frame = pd.read_csv(root / name)
        if frame.empty or frame.isna().any().any():
            raise AssertionError("Incomplete result: " + name)
    fix = pd.read_csv(root / "results/fix-check-v1/metrics.csv")
    assert len(fix) == 156 and fix.rmse.notna().all() and set(fix.seed) == {17, 42, 2026}
    c = json.loads((root / "results/c-inference-v3/verification.json").read_text())
    assert len(c) == 5 and sum(r["real_stream_steps"] for r in c) == 75000
    assert max(r["max_abs_error"] for r in c) < 1e-4
    assert len(PdfReader(root / "report/battery-shift-report.pdf").pages) == 6
    for temp in [25, 10, 0, -10, -20]:
        assert (root / f"results/diagnostics-v1/eda_{temp}.png").exists()
    assert (root / "results/modular-v3/E7/COMPLETE").exists()
    lg = json.loads((root / "results/modular-v3/E7/data_audit.json").read_text())
    assert len(lg["files"]) == 24 and len(lg["excluded"]) == 1
    assert sum(f["split"] == "adapt" for f in lg["files"]) == 5
    assert len(pd.read_csv(root / "results/modular-v3/E7/metrics.csv")) == 150
    official = json.loads((root / "results/diagnostics-v1/official_archive_audit.json").read_text())
    assert len(official["verified_mirror_files"]) == 26 and len(official["c20_files"]) == 1
    pipeline = json.loads((root / "results/full_pipeline_verification.json").read_text())
    assert pipeline["status"] == "completed" and pipeline["e7_included"]
    assert (Path(pipeline["run"]) / "COMPLETE").exists()
    platform = json.loads((root / "results/cross_platform_verification.json").read_text())
    assert platform["status"] == "completed" and platform["c_max_abs_error"] < 1e-4
    modular = json.loads((root / "results/modular-v3/verification.json").read_text())
    assert modular["status"] == "passed" and len(modular["ensemble_members"]) == 5
    assert modular["protocol"]["temperature_scale"] == [25.0, 20.0]
    verification = json.loads(
        (root / "results/diagnostics-v1/full_environment_verification.json").read_text()
    )
    assert verification["exact_numeric_equality"]
    suites = ET.parse(root / "results/test_results.xml").getroot().findall(".//testsuite")
    assert suites and all(int(t.get("failures", 0)) == 0 and int(t.get("errors", 0)) == 0 for t in suites)
    result = dict(
        status="local_artifacts_verified",
        result_tables=expected,
        post_hoc_checks=["results/fix-check-v1/metrics.csv"],
        pdf_pages=6,
        c_max_abs_error=max(r["max_abs_error"] for r in c),
        tests_passed=sum(int(t.get("tests", 0)) for t in suites),
        modular_stage_checks=modular,
        remaining=[
            "real MCU measurements",
            "independent physical-machine validation",
            "Linux/WSL re-validation of protocol v3 (Linux evidence covers v1)",
            "end-to-end reproduce_all.py run with the protocol-v3 stages",
            "remote GitHub CI (deferred by user)",
        ],
    )
    (root / "results/delivery_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
