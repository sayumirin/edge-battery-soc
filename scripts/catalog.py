"""Build a pinned download inventory. Run only when deliberately updating the dataset."""

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = "6d720489c9b5b95624e448d44e8e2c0e48bdacaf"
REPO = "fededc88/Panasonic-18650PF-Data"


def main():
    with urllib.request.urlopen(
        f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1", timeout=60
    ) as r:
        tree = json.load(r)["tree"]
    entries = []
    for f in tree:
        p = f["path"]
        if f["type"] != "blob":
            continue
        match = re.search(r"(?:^|/)(-?\d+)degC/", p)
        cycle = re.search(r"degC_(Cycle_[1-4]|US06|HWFTa|HWFTb|HWFET|UDDS|LA92|NN)_Pan18650PF\.mat$", p)
        if match and cycle:
            temp = int(match[1])
            name = cycle[1]
            if name.startswith("HWFT"):
                name = "HWFET"
            split = (
                (
                    "train"
                    if name in ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN"]
                    else "validation"
                    if name == "HWFET"
                    else "test"
                )
                if temp == 25
                else (
                    "adapt" if name == "Cycle_1" else "test" if name in ["US06", "UDDS", "LA92"] else "unused"
                )
            )
            if split == "unused":
                continue
            entries.append(
                dict(
                    path=p,
                    git_blob=f["sha"],
                    size=f["size"],
                    ambient_c=temp,
                    cycle=name,
                    split=split,
                    file_id=f["sha"],
                )
            )
        elif "C20" in p or p.endswith(".txt"):
            print("Reference:", p)
    out = ROOT / "data/splits/manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(repository=REPO, revision=REV, files=entries), indent=2), encoding="utf-8")
    print(f"Selected {len(entries)} files, {sum(e['size'] for e in entries) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
